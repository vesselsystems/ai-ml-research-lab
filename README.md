# Churn prediction: an evidence-first comparison

[![CI](https://github.com/vesselsystems/ai-ml-research-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/vesselsystems/ai-ml-research-lab/actions/workflows/ci.yml)

This repository asks a narrow question: on the IBM Telco Customer Churn CSV, does a random forest rank churn cases better than a regularized logistic-regression model? It also includes a majority-class reference, held-out threshold and calibration reports, and aggregate score-band error analysis so that ranking metrics are not mistaken for an operating decision.

The result is an offline experiment. It does not estimate the effect of a retention action, test an intervention, or support decisions about individual customers.

## Current result

The current run used seed 42, a stratified 80/20 split, and five-fold stratified cross-validation on the training rows. Values below are from the 1,409-row holdout; the positive rate in that holdout is 26.54%.

| Model | ROC-AUC | 95% bootstrap interval | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.5000 | 0.5000–0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic regression | 0.8413 | 0.8181–0.8629 | 0.6326 | 0.5043 | 0.7834 | 0.6136 | 0.7381 |
| Random forest | 0.8366 | 0.8123–0.8586 | 0.6445 | 0.5621 | 0.6898 | 0.6194 | 0.7750 |

The logistic model has the higher holdout ROC-AUC, while the forest has slightly higher average precision, F1, precision, and accuracy at threshold 0.5. The ROC-AUC intervals overlap, and this single split does not establish a generally superior model. The complete table is in `reports/first_run.md`; threshold, calibration, and score-band error counts are generated under `reports/` by the experiment script.

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

`run_experiment.py` writes `model_comparison.csv`, `metrics.json`, `threshold_analysis.csv`, `calibration_analysis.csv`, and `error_analysis.csv` under `reports/`. The model summary and JSON are generated files ignored by Git; the three analysis CSVs are reproducible evidence artifacts. Thresholds 0.2 through 0.7 are fixed for the same holdout used in the summary and are descriptive, not a threshold-selection procedure. Calibration uses ten equal-width probability bins without fitting a recalibration model. Error analysis aggregates confusion counts by those score bands and exports no identifiers or feature values.

## Data and method

- **Source:** [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv), pinned to commit `d5371f5d83a446ad5673cbcca3b814b926491f8a`, 7,043 rows in the current local file.
- **Label:** `Churn=Yes` is 1 and `Churn=No` is 0; the full-file positive rate is 26.54%.
- **Features:** `customerID` is removed. Numeric values use median imputation and scaling; categorical values use most-frequent imputation and one-hot encoding. `TotalCharges` is parsed as numeric, with blank values becoming missing values for imputation.
- **Candidates:** a majority-class `DummyClassifier`, class-weighted logistic regression, and a class-weighted random forest. The majority classifier is a reference point, not a useful churn scorer.
- **Evaluation:** the preprocessing is fitted inside each pipeline. The holdout is used once for the reported comparison and the descriptive threshold, calibration, and error analyses; cross-validation is run on the training rows. Metrics are ROC-AUC, average precision, Brier score, log loss, precision, recall, F1, and accuracy. Held-out ROC-AUC intervals are percentile intervals from 500 bootstrap resamples. Calibration metrics are descriptive estimates on this holdout, not evidence that scores are valid for a future population or action.

The full protocol, including controls and interpretation boundaries, is in [`reports/experiment_protocol.md`](reports/experiment_protocol.md). The model card is in [`reports/model_card.md`](reports/model_card.md). [`docs/methodology.md`](docs/methodology.md) explains the design in more detail.

## Provenance evidence boundary

[`data/provenance.json`](data/provenance.json) is the tracked provenance record. It pins the source URL and upstream revision and records measurements made from the currently available ignored `data/raw/telco_churn.csv`: its SHA-256, byte size, CSV header, and row count. Those measurements establish only the state of that local snapshot when recorded; they do not prove freshness, that an upstream response would be byte-identical, or that the data are representative.

The retrieval date and license/permission review are explicitly `null`/`pending` in the record. This project therefore makes no retrieval-date, license, permission, or redistribution claim. The raw CSV remains untracked. `scripts/validate_provenance.py` performs local-only checks and never downloads; it validates recognized review statuses and timezone-aware ISO timestamps without filling missing facts. A missing file is unavailable evidence, a pending review is reported separately from a measured match, and any hash, size, schema, or row-count drift fails. `--allow-missing` and `--allow-pending` are explicit CI skips/allowances only; they never allow a mismatch. For a clean checkout, CI runs `python scripts/validate_provenance.py --allow-missing --allow-pending`.

## Threshold and decision-policy boundary

The threshold grid is an analysis convention, not an operating policy. `REPORTING_THRESHOLD = 0.5` is used only for the reported precision/recall/F1/accuracy and error-analysis conventions; no production threshold is recommended. The dataset supplies neither a retention action nor business costs, so this repository does not invent them or report expected savings, ROI, or net benefit.

The versioned [decision-policy template](policies/decision_policy.v1.json) and its [boundary documentation](docs/decision_policy.md) make the missing inputs explicit. A future, separately governed action must supply an intended action, eligible population, `cost_unit`, hard capacity with an explicit period and `highest_score_then_input_order` selection rule, reviewed false-positive and false-negative costs, owner, non-expired review date, excluded-use list, schema/policy versions, and approval status. `policy_threshold_analysis` accepts only an explicitly supplied, approved policy and reports cost/capacity tradeoffs; it has no cost or capacity defaults. No approved policy is present here, so threshold selection remains blocked.

The existing threshold, calibration, and error tables are descriptive reporting only. They do not select an operating threshold. If a policy is later approved, threshold selection must use training/validation data or a separate decision set, with the holdout reserved for confirmation. The policy cost expression is `total_cost = false_positive_cost * FP + false_negative_cost * FN`.

## Calibration and error analysis

`reports/calibration_analysis.csv` evaluates the same holdout in ten fixed equal-width probability bins. It reports bin counts, observed positive rates, mean predicted probabilities, Brier score, log loss, expected calibration error (ECE), and maximum absolute bin gap. ECE is the row-weighted mean absolute observed-minus-predicted gap over nonempty bins. Current holdout values are:

| Model | Brier score | Log loss | ECE | Max absolute bin gap |
|---|---:|---:|---:|---:|
| Majority class | 0.2654 | 9.5673 | 0.2654 | 0.2654 |
| Logistic regression | 0.1688 | 0.4969 | 0.1504 | 0.3171 |
| Random forest | 0.1491 | 0.4524 | 0.0860 | 0.1826 |

Lower Brier score, log loss, and calibration gaps are better within this holdout, but these estimates are not a calibration guarantee and no recalibration model was fitted. `reports/error_analysis.csv` breaks each model's threshold-0.5 holdout errors into the same score bands. At threshold 0.5, logistic regression has 288 false positives and 81 false negatives; the forest has 201 false positives and 116 false negatives. These are aggregate label errors, not evidence about causes, customer subgroups, or intervention response.

## Scope and limitations

The source records describe historical churn labels, not whether any particular outreach would have changed an outcome. Contract, payment, tenure, demographic, and service fields may encode unequal access or act as proxies for protected characteristics. No subgroup or fairness result is claimed: the available fields have not been approved as evaluation groups, and no such analysis was run. The source also does not provide a defensible temporal evaluation design here, so no time-based performance or drift claim is made. Privacy/legal review and an intervention study are absent. Those limitations are reasons not to use these scores for customer treatment, eligibility, denial, or prioritization without separate evidence and review.

## License

The code and documentation are released under the [MIT License](LICENSE). The source dataset remains subject to its publisher's terms.
