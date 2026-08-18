"""Run the experiment and write the reproducible result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from ai_ml_research_lab.data import load_dataset
from ai_ml_research_lab.experiment import run_experiments, run_threshold_analysis

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

    frame = load_dataset(source)
    summary, fitted = run_experiments(frame)
    threshold_report = run_threshold_analysis(frame, fitted)
    summary.to_csv(report_dir / "model_comparison.csv", index=False)
    threshold_report.to_csv(report_dir / "threshold_analysis.csv", index=False)
    (report_dir / "metrics.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote model and threshold reports to {report_dir}")
