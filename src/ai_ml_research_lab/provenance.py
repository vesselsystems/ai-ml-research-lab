"""Validate tracked provenance metadata against an ignored local snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_PROVENANCE_PATH = Path("data/provenance.json")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()
_PENDING_VALUES = {
    "pending",
    "unknown",
    "not established",
    "not_established",
    "not reviewed",
    "not_reviewed",
    "unverified",
}
_REVIEW_STATUS_VALUES = frozenset(
    {
        "pending",
        "unknown",
        "not established",
        "not_established",
        "not reviewed",
        "not_reviewed",
        "unverified",
        "verified",
        "reviewed",
        "established",
        "complete",
        "approved",
        "not applicable",
        "not_applicable",
    }
)
_RESOLVED_STATUS_VALUES = {
    "verified",
    "reviewed",
    "established",
    "complete",
    "approved",
}
_REVIEW_VALUE_FIELDS = (
    "license_or_terms",
    "license",
    "license_review",
    "permission_review",
    "license_permission_review",
)


class ProvenanceMetadataError(ValueError):
    """Raised when tracked provenance metadata cannot be validated safely."""


@dataclass(frozen=True)
class ProvenanceValidationResult:
    """The result of comparing provenance measurements with a local snapshot.

    ``measurements_match`` reports only the local byte/schema/row-count check.
    ``passed`` is stricter: it is false when those measurements are unavailable
    or drifted, and also when review metadata is explicitly pending.  This keeps
    a measured local match distinct from an unresolved retrieval or terms review.
    """

    passed: bool
    snapshot_available: bool
    provenance_path: Path
    snapshot_path: Path
    errors: tuple[str, ...] = ()
    pending_metadata: tuple[str, ...] = ()
    measurements_match: bool = False
    status: str = "invalid"

    @property
    def metadata_pending(self) -> bool:
        """Whether the provenance record contains unresolved metadata fields."""
        return bool(self.pending_metadata)

    @property
    def pending(self) -> bool:
        """Alias for :attr:`metadata_pending` for callers reporting status."""
        return self.metadata_pending

    @property
    def metadata_status(self) -> str:
        """Return ``pending`` or ``complete`` for review metadata."""
        return "pending" if self.metadata_pending else "complete"

    @property
    def integrity_match(self) -> bool:
        """Alias for the measured local snapshot result."""
        return self.measurements_match


def validate_provenance(
    provenance_path: Path = DEFAULT_PROVENANCE_PATH,
    *,
    raw_path: Path | None = None,
    project_root: Path | None = None,
) -> ProvenanceValidationResult:
    """Compare tracked measurements with an available local raw CSV.

    This function performs local filesystem reads only.  It never downloads a
    replacement snapshot.  A missing snapshot is reported as unavailable
    evidence; the command-line entry point can explicitly allow that state for
    a clean CI checkout.  Hash, size, schema, and row-count mismatches always
    remain failures.
    """
    provenance_path = Path(provenance_path).resolve()
    metadata = _read_metadata(provenance_path)
    snapshot_metadata = _snapshot_metadata(metadata)
    _validate_review_metadata(metadata)
    pending_metadata = _pending_metadata(metadata)
    root = _project_root(provenance_path, project_root)
    snapshot_path = _resolve_snapshot_path(
        snapshot_metadata["path"],
        root,
        raw_path,
    )
    expected_hash = snapshot_metadata["sha256"]
    expected_size = snapshot_metadata["size_bytes"]
    expected_columns = snapshot_metadata["source_columns"]
    expected_row_count = snapshot_metadata.get("row_count")

    if not snapshot_path.exists():
        return ProvenanceValidationResult(
            passed=False,
            snapshot_available=False,
            provenance_path=provenance_path,
            snapshot_path=snapshot_path,
            errors=(f"raw snapshot unavailable: {snapshot_path}",),
            pending_metadata=pending_metadata,
            status="missing",
        )

    if not snapshot_path.is_file():
        return ProvenanceValidationResult(
            passed=False,
            snapshot_available=True,
            provenance_path=provenance_path,
            snapshot_path=snapshot_path,
            errors=(f"raw snapshot is not a regular file: {snapshot_path}",),
            pending_metadata=pending_metadata,
            status="drift",
        )

    errors: list[str] = []
    try:
        actual_hash = _sha256(snapshot_path)
        actual_size = snapshot_path.stat().st_size
    except OSError as exc:
        return ProvenanceValidationResult(
            passed=False,
            snapshot_available=True,
            provenance_path=provenance_path,
            snapshot_path=snapshot_path,
            errors=(f"could not read raw snapshot: {exc}",),
            pending_metadata=pending_metadata,
            status="drift",
        )

    if actual_hash != expected_hash:
        errors.append("sha256 mismatch between tracked metadata and raw snapshot")
    if actual_size != expected_size:
        errors.append(
            f"size_bytes mismatch (tracked {expected_size}, local {actual_size})"
        )

    try:
        actual_columns, actual_row_count = _csv_measurements(snapshot_path)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        errors.append(f"could not inspect CSV schema: {exc}")
    else:
        if actual_columns != expected_columns:
            errors.append("source_columns schema mismatch between metadata and raw snapshot")
        if expected_row_count is not None and actual_row_count != expected_row_count:
            errors.append(
                f"row_count mismatch (tracked {expected_row_count}, local {actual_row_count})"
            )

    measurements_match = not errors
    if errors:
        status = "drift"
    elif pending_metadata:
        status = "pending"
    else:
        status = "matched"

    return ProvenanceValidationResult(
        passed=measurements_match and not pending_metadata,
        snapshot_available=True,
        provenance_path=provenance_path,
        snapshot_path=snapshot_path,
        errors=tuple(errors),
        pending_metadata=pending_metadata,
        measurements_match=measurements_match,
        status=status,
    )


def _read_metadata(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProvenanceMetadataError(f"cannot read provenance metadata: {exc}") from exc

    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProvenanceMetadataError(f"provenance metadata is not valid JSON: {exc}") from exc

    if not isinstance(metadata, Mapping):
        raise ProvenanceMetadataError("provenance metadata must contain a JSON object")
    return metadata


def _snapshot_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = metadata.get("local_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ProvenanceMetadataError("provenance metadata lacks a local_snapshot object")

    path = _required_string(snapshot, "path")
    digest = _required_string(snapshot, "sha256").lower()
    if not _SHA256.fullmatch(digest):
        raise ProvenanceMetadataError("local_snapshot.sha256 must be a 64-character hex digest")

    size_bytes = _required_nonnegative_int(snapshot, "size_bytes")
    source_columns = snapshot.get("source_columns", snapshot.get("columns"))
    if (
        not isinstance(source_columns, list)
        or not source_columns
        or not all(isinstance(column, str) and column for column in source_columns)
    ):
        raise ProvenanceMetadataError(
            "local_snapshot.source_columns must be a non-empty list of strings"
        )

    normalized: dict[str, Any] = {
        "path": path,
        "sha256": digest,
        "size_bytes": size_bytes,
        "source_columns": source_columns,
    }
    if "row_count" in snapshot:
        normalized["row_count"] = _required_nonnegative_int(snapshot, "row_count")
    return normalized


def _validate_review_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate optional review statuses and timestamps without filling them in."""
    schema_version = metadata.get("provenance_schema_version", _MISSING)
    if schema_version is not _MISSING:
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ProvenanceMetadataError(
                "provenance_schema_version must be an integer when provided"
            )
        if schema_version != 1:
            raise ProvenanceMetadataError(
                "provenance_schema_version must be exactly 1"
            )

    source = metadata.get("source", {})
    if not isinstance(source, Mapping):
        raise ProvenanceMetadataError("provenance metadata source must be an object")
    snapshot = metadata.get("local_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ProvenanceMetadataError("provenance metadata lacks a local_snapshot object")

    status_values: dict[str, str] = {}
    for section_name, values, names in (
        (
            "source",
            source,
            (
                "license_review_status",
                "permission_review_status",
                "license_permission_review_status",
            ),
        ),
        ("local_snapshot", snapshot, ("retrieval_date_status",)),
    ):
        for name in names:
            if name in values:
                status_values[f"{section_name}.{name}"] = _review_status(
                    values[name], f"{section_name}.{name}"
                )

    for name in _REVIEW_VALUE_FIELDS:
        if name in source:
            _optional_review_text(source[name], f"source.{name}")

    for name in ("license_or_terms_url", "license_url"):
        if name in source:
            _optional_review_text(source[name], f"source.{name}")

    for name in ("retrieved_at_utc", "metadata_generated_at_utc"):
        if name in snapshot and snapshot[name] is not None:
            _parse_utc_timestamp(snapshot[name], f"local_snapshot.{name}")

    if "retrieval_date" in snapshot and snapshot["retrieval_date"] is not None:
        retrieval_date = snapshot["retrieval_date"]
        if not _is_pending(retrieval_date):
            if not isinstance(retrieval_date, str):
                raise ProvenanceMetadataError(
                    "local_snapshot.retrieval_date must be an ISO date or null"
                )
            try:
                date.fromisoformat(retrieval_date.strip())
            except ValueError as error:
                raise ProvenanceMetadataError(
                    "local_snapshot.retrieval_date must be an ISO date or null"
                ) from error

    _reject_resolved_without_value(
        source,
        status_values,
        status_name="source.license_review_status",
        value_name="license_or_terms",
    )
    _reject_resolved_without_value(
        source,
        status_values,
        status_name="source.permission_review_status",
        value_name="permission_review",
    )
    retrieval_status = status_values.get("local_snapshot.retrieval_date_status")
    if retrieval_status in _RESOLVED_STATUS_VALUES and not any(
        snapshot.get(name) not in (None, _MISSING)
        for name in ("retrieved_at_utc", "retrieval_date")
    ):
        raise ProvenanceMetadataError(
            "local_snapshot.retrieval_date_status is resolved but no retrieval date "
            "or timestamp is recorded"
        )


def _review_status(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceMetadataError(f"{field} must be a recognized status string")
    normalized = " ".join(value.strip().lower().split())
    if normalized not in _REVIEW_STATUS_VALUES:
        raise ProvenanceMetadataError(f"{field} has unsupported status {value!r}")
    return normalized


def _optional_review_text(value: object, field: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ProvenanceMetadataError(f"{field} must be non-empty text or null")


def _parse_utc_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceMetadataError(f"{field} must be a timezone-aware ISO timestamp or null")
    normalized_value = value.strip()
    normalized = (
        normalized_value[:-1] + "+00:00"
        if normalized_value.endswith("Z")
        else normalized_value
    )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ProvenanceMetadataError(
            f"{field} must be a timezone-aware ISO timestamp or null"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProvenanceMetadataError(
            f"{field} must be a timezone-aware ISO timestamp or null"
        )
    if parsed.utcoffset() != timedelta(0):
        raise ProvenanceMetadataError(f"{field} must use the UTC offset")
    return parsed


def _reject_resolved_without_value(
    values: Mapping[str, Any],
    statuses: Mapping[str, str],
    *,
    status_name: str,
    value_name: str,
) -> None:
    if statuses.get(status_name) in _RESOLVED_STATUS_VALUES and (
        value_name not in values or values[value_name] is None
    ):
        raise ProvenanceMetadataError(
            f"{status_name} is resolved but {value_name} is null or missing"
        )


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise ProvenanceMetadataError(f"local_snapshot.{name} must be a non-empty string")
    return value


def _required_nonnegative_int(values: Mapping[str, Any], name: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProvenanceMetadataError(f"local_snapshot.{name} must be a non-negative integer")
    return value


def _pending_metadata(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect explicit unresolved retrieval and terms/permission fields."""
    pending: list[str] = []
    source = metadata.get("source")
    snapshot = metadata.get("local_snapshot")

    if isinstance(snapshot, Mapping):
        for name in ("retrieved_at_utc", "retrieval_date"):
            if name in snapshot and _is_pending(snapshot[name]):
                pending.append(f"local_snapshot.{name}")
        if "retrieval_date_status" in snapshot and _is_pending(
            snapshot["retrieval_date_status"]
        ):
            pending.append("local_snapshot.retrieval_date_status")

    if isinstance(source, Mapping):
        for name in (
            "license_or_terms",
            "license",
            "license_review",
            "permission_review",
            "license_permission_review",
        ):
            if name in source and _is_pending(source[name]):
                pending.append(f"source.{name}")
        for name in (
            "license_review_status",
            "permission_review_status",
            "license_permission_review_status",
        ):
            if name in source and _is_pending(source[name]):
                pending.append(f"source.{name}")

    return tuple(dict.fromkeys(pending))


def _is_pending(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = " ".join(value.strip().lower().split())
        return not normalized or normalized in _PENDING_VALUES
    return False


def _project_root(provenance_path: Path, project_root: Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()

    # The repository layout is data/provenance.json.  Falling back to the
    # metadata directory keeps temporary metadata fixtures convenient to use.
    if provenance_path.parent.name == "data":
        return provenance_path.parent.parent
    return provenance_path.parent


def _resolve_snapshot_path(
    metadata_path: str,
    project_root: Path,
    raw_path: Path | None,
) -> Path:
    if raw_path is not None:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()

    candidate = Path(metadata_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (project_root / candidate).resolve()

    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ProvenanceMetadataError(
            "local_snapshot.path must stay within the project root"
        ) from exc
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_measurements(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV has no header row") from exc
        row_count = sum(1 for row in reader if row)
    return columns, row_count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare tracked provenance metadata with the ignored local raw CSV. "
            "No data is downloaded."
        )
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=DEFAULT_PROVENANCE_PATH,
        help="tracked provenance JSON path (default: data/provenance.json)",
    )
    parser.add_argument(
        "--raw",
        dest="raw_path",
        type=Path,
        help="optional local raw CSV path override",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="root used to resolve a repository-relative snapshot path",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "treat an unavailable ignored snapshot as an explicit skip; metadata "
            "errors and mismatches still fail"
        ),
    )
    parser.add_argument(
        "--allow-pending",
        "--allow-pending-metadata",
        dest="allow_pending",
        action="store_true",
        help=(
            "allow explicitly pending retrieval or terms metadata after local "
            "measurements match; mismatches still fail"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the provenance validator and return a process exit code."""
    args = _parser().parse_args(argv)
    try:
        result = validate_provenance(
            args.provenance,
            raw_path=args.raw_path,
            project_root=args.project_root,
        )
    except ProvenanceMetadataError as exc:
        print(f"ERROR: invalid provenance metadata: {exc}", file=sys.stderr)
        return 1

    if not result.snapshot_available:
        message = result.errors[0]
        if args.allow_missing:
            print(
                f"SKIP: {message}; no local snapshot validation was performed. "
                "This is unavailable evidence, not a measured match."
            )
            return 0
        print(f"ERROR: {message}", file=sys.stderr)
        print(
            "Provide the pinned snapshot locally before validating it; this validator "
            "does not download data.",
            file=sys.stderr,
        )
        return 1

    if result.errors:
        print("ERROR: provenance validation failed:", file=sys.stderr)
        for error in result.errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if result.metadata_pending:
        fields = ", ".join(result.pending_metadata)
        message = (
            "local snapshot measurements match, but provenance metadata remains "
            f"pending ({fields})"
        )
        if args.allow_pending:
            print(f"PENDING: {message}; pending state was explicitly allowed.")
            return 0
        print(
            f"ERROR: {message}; use --allow-pending only for an explicit pending state.",
            file=sys.stderr,
        )
        return 1

    print(
        "PASS: tracked provenance matches the local raw snapshot "
        f"({result.snapshot_path})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
