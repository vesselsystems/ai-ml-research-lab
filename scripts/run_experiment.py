"""Run the full experiment and write portfolio-friendly artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from ai_ml_research_lab.data import load_dataset
from ai_ml_research_lab.experiment import run_experiments

if __name__ == "__main__":
    root = Path(__file__).parents[1]
    source = root / "data" / "raw" / "telco_churn.csv"
    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)

    frame = load_dataset(source)
    summary, _ = run_experiments(frame)
    summary.to_csv(report_dir / "model_comparison.csv", index=False)
    (report_dir / "metrics.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote reports to {report_dir}")
