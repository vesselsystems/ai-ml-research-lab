"""Run the experiment and write the reproducible result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from ai_ml_research_lab.data import load_dataset
from ai_ml_research_lab.experiment import (
    run_calibration_analysis,
    run_error_analysis,
    run_experiments,
    run_threshold_analysis,
)
from ai_ml_research_lab.provenance import ProvenanceMetadataError, validate_provenance

if __name__ == "__main__":
    root = Path(__file__).parents[1]
    source = root / "data" / "raw" / "telco_churn.csv"
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise SystemExit(
            f"Missing {source}. Run `python scripts/download_data.py` first "
            "(the raw CSV is intentionally not tracked)."
        )

    try:
        provenance_result = validate_provenance(
            root / "data" / "provenance.json",
            raw_path=source,
            project_root=root,
        )
    except ProvenanceMetadataError as error:
        raise SystemExit(f"Invalid tracked provenance metadata: {error}") from error
    if not provenance_result.measurements_match:
        detail = "; ".join(provenance_result.errors) or "snapshot measurements do not match"
        raise SystemExit(f"Refusing to analyze an unverified local snapshot: {detail}")
    if provenance_result.metadata_pending:
        print(
            "WARNING: local snapshot measurements match, but provenance review remains "
            f"pending: {', '.join(provenance_result.pending_metadata)}"
        )

    frame = load_dataset(source)
    summary, fitted = run_experiments(frame)
    threshold_report = run_threshold_analysis(frame, fitted)
    calibration_report = run_calibration_analysis(frame, fitted)
    error_report = run_error_analysis(frame, fitted)
    summary.to_csv(report_dir / "model_comparison.csv", index=False)
    threshold_report.to_csv(report_dir / "threshold_analysis.csv", index=False)
    calibration_report.to_csv(report_dir / "calibration_analysis.csv", index=False)
    error_report.to_csv(report_dir / "error_analysis.csv", index=False)
    (report_dir / "metrics.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote model, threshold, calibration, and error reports to {report_dir}")
