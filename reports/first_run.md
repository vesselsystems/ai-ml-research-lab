# Current run results

This report records the run generated from the local [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv), pinned to commit `d5371f5d83a446ad5673cbcca3b814b926491f8a`. The input had SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`, 7,043 rows, and a 26.54% positive rate. Seed 42 produced a stratified 80/20 split; the holdout had 1,409 rows. Five-fold stratified cross-validation used training rows only.

## Holdout comparison

Precision, recall, F1, and accuracy use the default probability threshold of 0.5. ROC-AUC intervals are 2.5th–97.5th percentile intervals from 500 bootstrap resamples of the holdout.

| Model | Test ROC-AUC | 95% bootstrap interval | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.5000 | 0.5000–0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic regression | 0.8413 | 0.8181–0.8629 | 0.6326 | 0.5043 | 0.7834 | 0.6136 | 0.7381 |
| Random forest | 0.8366 | 0.8123–0.8586 | 0.6445 | 0.5621 | 0.6898 | 0.6194 | 0.7750 |

The majority reference shows why accuracy is insufficient for this target: it predicts no positives. Logistic regression has the higher ROC-AUC and recall in this split. The forest has higher average precision, precision, F1, and accuracy at 0.5. Their ROC-AUC intervals overlap, so this run does not support a general winner.

## Threshold view

`threshold_analysis.csv` contains fixed threshold rows for each model. At threshold 0.5, logistic regression flags 581 of 1,409 holdout rows (precision 0.5043, recall 0.7834); the forest flags 459 (precision 0.5621, recall 0.6898). At threshold 0.6, logistic regression has precision 0.5399 and recall 0.7059; the forest has precision 0.6261 and recall 0.5775.

This is a descriptive error-cost view, not threshold tuning. The file does not show whether a customer would benefit from an action, and no action or cost matrix was provided.

## Boundaries

The source label is observed churn, not the result of a controlled retention intervention. These scores cannot establish causality or expected business impact. Contract, payment, tenure, demographic, and service fields may be proxies for protected characteristics; this run has no subgroup, calibration, privacy/legal, drift, or intervention evaluation. See [`model_card.md`](model_card.md) and [`experiment_protocol.md`](experiment_protocol.md) for the data provenance, uncertainty, and excluded uses.
