import copy
import hashlib
import json
from pathlib import Path

import pytest

from ai_ml_research_lab.provenance import (
    ProvenanceMetadataError,
    main,
    validate_provenance,
)

COLUMNS = ["id", "value"]


def _write_metadata(
    tmp_path: Path,
    *,
    raw_name: str = "snapshot.csv",
    pending: bool = False,
) -> Path:
    raw = tmp_path / raw_name
    raw.write_text(",".join(COLUMNS) + "\n1,10\n2,20\n", encoding="utf-8")
    source: dict[str, object] = {}
    if pending:
        source = {
            "license_or_terms": None,
            "license_review_status": "pending",
            "permission_review": None,
            "permission_review_status": "pending",
        }
    metadata = {
        "source": source,
        "local_snapshot": {
            "path": raw.name,
            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "size_bytes": raw.stat().st_size,
            "source_columns": COLUMNS,
            "row_count": 2,
            **(
                {"retrieved_at_utc": None, "retrieval_date_status": "pending"}
                if pending
                else {"retrieved_at_utc": "2025-01-01T00:00:00Z"}
            ),
        },
    }
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(metadata), encoding="utf-8")
    return provenance


def test_provenance_matches_local_snapshot(tmp_path: Path) -> None:
    provenance = _write_metadata(tmp_path)

    result = validate_provenance(provenance, project_root=tmp_path)

    assert result.passed is True
    assert result.snapshot_available is True
    assert result.measurements_match is True
    assert result.status == "matched"
    assert result.errors == ()


def test_provenance_fails_on_snapshot_drift(tmp_path: Path) -> None:
    provenance = _write_metadata(tmp_path)
    (tmp_path / "snapshot.csv").write_text(
        ",".join(COLUMNS) + "\n1,10\n2,20\n3,30\n",
        encoding="utf-8",
    )

    result = validate_provenance(provenance, project_root=tmp_path)

    assert result.passed is False
    assert result.snapshot_available is True
    assert result.measurements_match is False
    assert result.status == "drift"
    assert any("sha256 mismatch" in error for error in result.errors)
    assert any("size_bytes mismatch" in error for error in result.errors)
    assert any("row_count mismatch" in error for error in result.errors)


def test_provenance_reports_missing_snapshot_without_downloading(
    tmp_path: Path,
) -> None:
    provenance = _write_metadata(tmp_path, raw_name="missing.csv")
    (tmp_path / "missing.csv").unlink()

    result = validate_provenance(provenance, project_root=tmp_path)

    assert result.passed is False
    assert result.snapshot_available is False
    assert result.measurements_match is False
    assert result.status == "missing"
    assert "unavailable" in result.errors[0]


def test_cli_requires_explicit_allow_missing_for_unavailable_snapshot(
    tmp_path: Path, capsys
) -> None:
    provenance = _write_metadata(tmp_path, raw_name="missing.csv")
    (tmp_path / "missing.csv").unlink()

    failed = main(["--provenance", str(provenance)])
    failure_output = capsys.readouterr()
    skipped = main(["--provenance", str(provenance), "--allow-missing"])
    skip_output = capsys.readouterr()

    assert failed == 1
    assert "unavailable" in failure_output.err
    assert skipped == 0
    assert "not a measured match" in skip_output.out


def test_pending_metadata_is_distinct_from_measured_match(
    tmp_path: Path, capsys
) -> None:
    provenance = _write_metadata(tmp_path, pending=True)

    result = validate_provenance(provenance, project_root=tmp_path)

    assert result.passed is False
    assert result.measurements_match is True
    assert result.metadata_pending is True
    assert result.metadata_status == "pending"
    assert result.status == "pending"
    assert "source.license_or_terms" in result.pending_metadata
    assert "local_snapshot.retrieved_at_utc" in result.pending_metadata

    blocked = main(["--provenance", str(provenance)])
    blocked_output = capsys.readouterr()
    allowed = main(["--provenance", str(provenance), "--allow-pending"])
    allowed_output = capsys.readouterr()

    assert blocked == 1
    assert "measurements match" in blocked_output.err
    assert allowed == 0
    assert "PENDING" in allowed_output.out


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("local_snapshot", "retrieved_at_utc", "2025-01-01T00:00:00"),
        ("local_snapshot", "retrieved_at_utc", "not-a-timestamp"),
        ("local_snapshot", "retrieved_at_utc", "2025-01-01T00:00:00+01:00"),
        ("local_snapshot", "retrieval_date_status", "finished"),
        ("source", "license_review_status", "finished"),
    ],
)
def test_provenance_rejects_invalid_status_or_timestamp(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    provenance = _write_metadata(tmp_path)
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    metadata = copy.deepcopy(metadata)
    metadata[section][field] = value
    provenance.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ProvenanceMetadataError):
        validate_provenance(provenance, project_root=tmp_path)


def test_provenance_rejects_resolved_status_without_claiming_a_value(
    tmp_path: Path,
) -> None:
    provenance = _write_metadata(tmp_path, pending=True)
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    metadata["source"]["license_review_status"] = "verified"
    provenance.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ProvenanceMetadataError, match="resolved"):
        validate_provenance(provenance, project_root=tmp_path)


def test_allow_flags_never_turn_drift_into_a_pass(tmp_path: Path) -> None:
    provenance = _write_metadata(tmp_path)
    (tmp_path / "snapshot.csv").write_text(",".join(COLUMNS) + "\n1,10\n", encoding="utf-8")

    result = main(
        [
            "--provenance",
            str(provenance),
            "--allow-missing",
            "--allow-pending",
        ]
    )

    assert result == 1
