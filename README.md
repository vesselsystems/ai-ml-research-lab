# Churn Prediction: An Evidence-First Comparison

[![CI](https://github.com/vesselsystems/ai-ml-research-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/vesselsystems/ai-ml-research-lab/actions/workflows/ci.yml)

I compared a regularized logistic regression against a random forest on the IBM Telco churn data under one fixed protocol, with a majority-class baseline for scale.

## Why I built it

I wanted to know whether the bigger model actually wins here. It mostly does not. On this data the two land close enough that the honest answer is "it depends on the threshold," and I wanted a run that says that out loud instead of picking a winner.

## What it does

Seed 42, stratified 80/20 split, five-fold stratified cross-validation on the training rows. The numbers below are the 1,409-row holdout, positive rate 26.54%.

| Model | ROC-AUC | 95% bootstrap | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Majority class | 0.5000 | 0.5000–0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 |
| Logistic regression | 0.8413 | 0.8181–0.8629 | 0.6326 | 0.5043 | 0.7834 | 0.6136 |
| Random forest | 0.8366 | 0.8123–0.8586 | 0.6445 | 0.5621 | 0.6898 | 0.6194 |

The ROC-AUC intervals overlap, so this split does not crown a winner. The majority row is there to show that 73% accuracy means nothing when the model predicts no positives at all.

## How it works

- scikit-learn pipelines: median impute plus `StandardScaler` for numeric, most-frequent impute plus `OneHotEncoder` for categorical. `customerID` is dropped; blank `TotalCharges` becomes a missing value.
- Preprocessing is fit inside each pipeline, so nothing leaks from the holdout.
- The 0.2–0.7 threshold grid and the ten equal-width calibration bins are reporting conventions, not a threshold-selection procedure. No recalibration model is fitted.
- `policies/decision_policy.v1.json` lists what a real decision needs: intended action, eligible population, cost per false positive and false negative, capacity, owner, approval. The dataset supplies none of it, so those fields are blank rather than guessed.

## Run it

The raw CSV is gitignored.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Then download and run:

```bash
python scripts/download_data.py
python scripts/run_experiment.py
```

That regenerates `model_comparison.csv`, `metrics.json`, `threshold_analysis.csv`, `calibration_analysis.csv`, and `error_analysis.csv` under `reports/`.

## Tests and CI

52 tests plus Ruff, on Python 3.11 and 3.12. CI does not need the CSV.

```bash
pytest
ruff check .
```

## Limits and what's next

This is an offline study of historical churn labels. It says nothing about whether an outreach action would have changed any of those outcomes, and there is no cost model, no fairness analysis, and no temporal split. The license and permission fields in `data/provenance.json` are still open.

Next: close out the license review, add a subgroup analysis with groups someone has actually approved, and re-run against a decision set with real costs behind the threshold.

## License

MIT for the code and docs. The dataset stays with its publisher.
