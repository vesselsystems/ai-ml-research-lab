# Churn prediction: an evidence-first comparison

[![CI](https://github.com/vesselsystems/ai-ml-research-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/vesselsystems/ai-ml-research-lab/actions/workflows/ci.yml)

This repository asks a narrow question: on the IBM Telco Customer Churn CSV, does a random forest rank churn cases better than a regularized logistic-regression model? It also includes a majority-class reference and a held-out threshold table so that ranking metrics are not mistaken for an operating decision.

The result is an offline experiment. It does not estimate the effect of a retention action, test an intervention, or support decisions about individual customers.

## Current result

The current run used seed 42, a stratified 80/20 split, and five-fold stratified cross-validation on the training rows. Values below are from the 1,409-row holdout; the positive rate in that holdout is 26.54%.

| Model | ROC-AUC | 95% bootstrap interval | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.5000 | 0.5000–0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic regression | 0.8413 | 0.8181–0.8629 | 0.6326 | 0.5043 | 0.7834 | 0.6136 | 0.7381 |
| Random forest | 0.8366 | 0.8123–0.8586 | 0.6445 | 0.5621 | 0.6898 | 0.6194 | 0.7750 |

The logistic model has the higher holdout ROC-AUC, while the forest has slightly higher average precision, F1, precision, and accuracy at threshold 0.5. The ROC-AUC intervals overlap, and this single split does not establish a generally superior model. The complete table is in `reports/first_run.md`; the threshold counts are in `reports/threshold_analysis.csv` after running the experiment.

## Run it from a clean clone

The raw CSV is intentionally ignored by Git. Download it before running the data-dependent experiment; CI runs only the tests and does not need the CSV.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: use `source .venv/bin/activate` instead
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_experiment.py
pytest
ruff check .
```

`run_experiment.py` writes `model_comparison.csv`, `metrics.json`, and `threshold_analysis.csv` under `reports/`. The first two are generated files ignored by Git. The threshold report records fixed thresholds (0.2 through 0.7) for the same holdout used in the summary; it is descriptive, not a threshold-selection procedure.

## Data and method

- **Source:** [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv), pinned to commit `d5371f5d83a446ad5673cbcca3b814b926491f8a`, 7,043 rows in the current local file.
- **Label:** `Churn=Yes` is 1 and `Churn=No` is 0; the full-file positive rate is 26.54%.
- **Features:** `customerID` is removed. Numeric values use median imputation and scaling; categorical values use most-frequent imputation and one-hot encoding. `TotalCharges` is parsed as numeric, with blank values becoming missing values for imputation.
- **Candidates:** a majority-class `DummyClassifier`, class-weighted logistic regression, and a class-weighted random forest. The majority classifier is a reference point, not a useful churn scorer.
- **Evaluation:** the preprocessing is fitted inside each pipeline. The holdout is used once for the reported comparison; cross-validation is run on the training rows. Metrics are ROC-AUC, average precision, precision, recall, F1, and accuracy. Held-out ROC-AUC intervals are percentile intervals from 500 bootstrap resamples.

The full protocol, including controls and interpretation boundaries, is in [`reports/experiment_protocol.md`](reports/experiment_protocol.md). The model card is in [`reports/model_card.md`](reports/model_card.md). [`docs/methodology.md`](docs/methodology.md) explains the design in more detail.

## Scope and limitations

The source records describe historical churn labels, not whether any particular outreach would have changed an outcome. Contract, payment, tenure, demographic, and service fields may encode unequal access or act as proxies for protected characteristics. The experiment does not include subgroup performance, calibration assessment, privacy or legal review, drift checks, or an intervention study. Those omissions are reasons not to use these scores for customer treatment, eligibility, denial, or prioritization without separate evidence and review.

## License

The code and documentation are released under the [MIT License](LICENSE). The source dataset remains subject to its publisher's terms.
