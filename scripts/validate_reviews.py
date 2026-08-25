#!/usr/bin/env python3
"""Validate the human-review queue without making any review decisions."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_ml_research_lab.decision_policy import validate_decision_policy

DEFAULT_QUEUE_PATH = Path("reviews/review_queue.v1.json")
DEFAULT_POLICY_PATH = Path("policies/decision_policy.v1.json")
REQUIRED_REVIEW_IDS = frozenset(
    {"retrieval-date", "license-terms", "permission", "operational-use"}
)
REQUIRED_FIELDS = frozenset(
    {
        "id",
        "question",
        "evidence_refs",
        "status",
        "decision",
        "reviewer",
        "reviewed_at_utc",
        "rationale",
    }
)
OPTIONAL_FIELDS = frozenset({"notes", "blocks"})
STATUSES = frozenset({"pending", "approved", "rejected"})
_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ReviewQueueError(ValueError):
    """Raised when the review queue cannot be trusted as a v1 queue."""


@dataclass(frozen=True)
class ReviewQueueResult:
    """Validated queue facts used by the command-line gate."""

    queue_path: Path
    pending: tuple[tuple[str, str], ...]
    operational_use_approved: bool


def validate_review_queue(
    queue_path: Path = DEFAULT_QUEUE_PATH,
    *,
    project_root: Path | None = None,
) -> ReviewQueueResult:
    """Validate a queue and return its pending items and operational gate state."""
    queue_path = Path(queue_path).resolve()
    root = Path(project_root).resolve() if project_root is not None else _default_root(queue_path)
    payload = _read_json(queue_path)
    reviews = _validate_structure(payload, root)
    _validate_gate_consistency(reviews)

    pending = tuple(
        (review["id"], review["question"])
        for review in reviews
        if review["status"] == "pending"
    )
    operational = next(review for review in reviews if review["id"] == "operational-use")
    return ReviewQueueResult(
        queue_path=queue_path,
        pending=pending,
        operational_use_approved=operational["status"] == "approved",
    )


# A short, discoverable alias for callers that use the script name as the API.
validate_reviews = validate_review_queue


def _default_root(queue_path: Path) -> Path:
    if queue_path.parent.name == "reviews":
        return queue_path.parent.parent
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewQueueError(f"cannot read review queue: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReviewQueueError(f"review queue is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ReviewQueueError("review queue must contain a JSON object")
    return payload


def _validate_structure(payload: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    if set(payload) != {"review_queue_schema_version", "reviews"}:
        raise ReviewQueueError(
            "review queue must contain only review_queue_schema_version and reviews"
        )
    version = payload.get("review_queue_schema_version")
    if isinstance(version, bool) or version != 1:
        raise ReviewQueueError("review_queue_schema_version must be exactly 1")
    raw_reviews = payload.get("reviews")
    if not isinstance(raw_reviews, list) or not raw_reviews:
        raise ReviewQueueError("reviews must be a non-empty list")

    reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_review in enumerate(raw_reviews):
        field = f"reviews[{index}]"
        if not isinstance(raw_review, Mapping):
            raise ReviewQueueError(f"{field} must be an object")
        keys = set(raw_review)
        missing = REQUIRED_FIELDS - keys
        unknown = keys - REQUIRED_FIELDS - OPTIONAL_FIELDS
        if missing:
            raise ReviewQueueError(f"{field} is missing required fields: {sorted(missing)}")
        if unknown:
            raise ReviewQueueError(f"{field} has unknown fields: {sorted(unknown)}")

        review = dict(raw_review)
        review_id = _required_text(review["id"], f"{field}.id")
        if not _ID_PATTERN.fullmatch(review_id):
            raise ReviewQueueError(
                f"{field}.id must be a stable lowercase kebab-case identifier"
            )
        if review_id in seen:
            raise ReviewQueueError(f"duplicate review id: {review_id}")
        seen.add(review_id)
        _required_text(review["question"], f"{field}.question")
        _evidence_refs(review["evidence_refs"], root, f"{field}.evidence_refs")
        status = review["status"]
        if not isinstance(status, str) or status not in STATUSES:
            raise ReviewQueueError(
                f"{field}.status must be exactly one of pending, approved, rejected"
            )
        _optional_list(review, "notes", field)
        _optional_list(review, "blocks", field)
        _validate_decision_fields(review, field)
        if review_id == "retrieval-date" and status == "approved":
            _iso_date(review["decision"], f"{field}.decision")
        if review_id == "operational-use" and status == "approved":
            notes = review.get("notes", [])
            if not notes:
                raise ReviewQueueError(
                    f"{field}.notes must record operational approval details when approved"
                )
        if status == "approved" and review.get("blocks", []):
            raise ReviewQueueError(
                f"{field} claims approval while blocks are still recorded"
            )
        reviews.append(review)

    if seen != REQUIRED_REVIEW_IDS:
        missing = sorted(REQUIRED_REVIEW_IDS - seen)
        extra = sorted(seen - REQUIRED_REVIEW_IDS)
        details = []
        if missing:
            details.append(f"missing review ids: {missing}")
        if extra:
            details.append(f"unsupported review ids: {extra}")
        raise ReviewQueueError("review queue scope is incomplete (" + "; ".join(details) + ")")
    return reviews


def _validate_decision_fields(review: Mapping[str, Any], field: str) -> None:
    status = review["status"]
    decision = review["decision"]
    reviewer = review["reviewer"]
    reviewed_at = review["reviewed_at_utc"]
    rationale = review["rationale"]
    if status == "pending":
        for name, value in (
            ("decision", decision),
            ("reviewer", reviewer),
            ("reviewed_at_utc", reviewed_at),
            ("rationale", rationale),
        ):
            if value not in (None, ""):
                raise ReviewQueueError(
                    f"{field}.{name} must be null or empty while status is pending"
                )
        return

    for name, value in (("decision", decision), ("reviewer", reviewer), ("rationale", rationale)):
        _required_text(value, f"{field}.{name}")
    _utc_timestamp(reviewed_at, f"{field}.reviewed_at_utc")
    _check_decision_direction(status, decision, f"{field}.decision")


def _check_decision_direction(status: str, decision: object, field: str) -> None:
    normalized = str(decision).strip().lower()
    rejection_claim = bool(
        re.search(r"\brejected?\b|\bdenied\b|\bnot approved\b|\bnot permitted\b", normalized)
    )
    approval_claim = not rejection_claim and bool(
        re.search(r"\bapproved?\b|\bpermitted\b|\ballowed\b", normalized)
    )
    if approval_claim and rejection_claim:
        raise ReviewQueueError(f"{field} contains contradictory approval and rejection claims")
    if status == "approved" and rejection_claim:
        raise ReviewQueueError(f"{field} contradicts status approved")
    if status == "rejected" and approval_claim and not rejection_claim:
        raise ReviewQueueError(f"{field} contradicts status rejected")


def _validate_gate_consistency(reviews: Sequence[Mapping[str, Any]]) -> None:
    by_id = {review["id"]: review for review in reviews}
    operational = by_id["operational-use"]
    if operational["status"] == "approved":
        blockers = [
            review_id
            for review_id in ("retrieval-date", "license-terms", "permission")
            if by_id[review_id]["status"] != "approved"
        ]
        if blockers:
            raise ReviewQueueError(
                "operational-use claims approval while prerequisite reviews are not approved: "
                + ", ".join(blockers)
            )


def _evidence_refs(value: object, root: Path, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ReviewQueueError(f"{field} must be a non-empty list of tracked file paths")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ReviewQueueError(f"{field} must contain non-empty file paths")
    if len(set(value)) != len(value):
        raise ReviewQueueError(f"{field} must not contain duplicate paths")
    tracked = _tracked_files(root)
    for reference in value:
        path_text = reference.strip()
        path = Path(path_text)
        if path.is_absolute() or ".." in path.parts or "\\" in path_text:
            raise ReviewQueueError(f"{field} contains an unsafe path: {reference!r}")
        normalized = path.as_posix()
        if normalized not in tracked or not (root / path).is_file():
            raise ReviewQueueError(
                f"{field} references missing or untracked file: {reference}"
            )


def _tracked_files(root: Path) -> set[str]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
            text=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReviewQueueError(f"cannot determine tracked files with git: {exc}") from exc
    if completed.returncode != 0:
        raise ReviewQueueError(
            f"cannot determine tracked files with git (exit code {completed.returncode})"
        )
    return {item.decode("utf-8") for item in completed.stdout.split(b"\0") if item}


def _validate_policy(policy_path: Path) -> None:
    try:
        text = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewQueueError(f"cannot read decision policy: {exc}") from exc
    try:
        policy = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReviewQueueError(f"decision policy is not valid JSON: {exc}") from exc

    try:
        validate_decision_policy(policy)
    except ValueError as exc:
        raise ReviewQueueError(f"decision policy is not complete and approved: {exc}") from exc


def _optional_list(review: Mapping[str, Any], name: str, field: str) -> None:
    if name not in review:
        return
    value = review[name]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ReviewQueueError(f"{field}.{name} must be a list of non-empty strings")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewQueueError(f"{field} must be non-empty text")
    return value.strip()


def _utc_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ReviewQueueError(f"{field} must be a timezone-aware UTC ISO timestamp")
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReviewQueueError(f"{field} must be a timezone-aware UTC ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ReviewQueueError(f"{field} must be a timezone-aware UTC ISO timestamp")
    return parsed


def _iso_date(value: object, field: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ReviewQueueError(f"{field} must be an ISO date (YYYY-MM-DD)")
    text = value.strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ReviewQueueError(f"{field} must be an ISO date (YYYY-MM-DD)") from exc
    if parsed.isoformat() != text:
        raise ReviewQueueError(f"{field} must be an ISO date (YYYY-MM-DD)")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the tracked human-review queue.")
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE_PATH,
        help="review queue JSON path (default: reviews/review_queue.v1.json)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="repository root used to resolve evidence_refs (default: inferred)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="decision policy JSON path (default: policies/decision_policy.v1.json)",
    )
    parser.add_argument(
        "--require-operational-use",
        action="store_true",
        help="fail unless the operational-use review and decision policy are approved",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy_error: ReviewQueueError | None = None
    if args.require_operational_use:
        try:
            _validate_policy(args.policy)
        except ReviewQueueError as exc:
            policy_error = exc

    try:
        result = validate_review_queue(args.queue, project_root=args.project_root)
    except ReviewQueueError as exc:
        print(f"ERROR: invalid review queue: {exc}", file=sys.stderr)
        return 1

    print("PASS: review queue structure and evidence references are valid.")
    if result.pending:
        print("PENDING reviews:")
        for review_id, question in result.pending:
            print(f"- {review_id}: {question}")
    else:
        print("PENDING reviews: none")
    if result.operational_use_approved:
        print("Operational-use gate: approved.")
    else:
        print("Operational-use gate: not approved.")
    if args.require_operational_use:
        failed = False
        if not result.operational_use_approved:
            print(
                "ERROR: --require-operational-use requires an approved operational-use review.",
                file=sys.stderr,
            )
            failed = True
        if policy_error is not None:
            print(
                "ERROR: --require-operational-use requires a complete approved decision "
                f"policy ({args.policy}): {policy_error}",
                file=sys.stderr,
            )
            failed = True
        if failed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
