# First experiment run

Configuration: random seed 42, stratified 80/20 split, five-fold stratified cross-validation, class-weighted models.

| Model | Test ROC-AUC | 95% bootstrap CI | Avg. precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.8413 | 0.8181–0.8629 | 0.6326 | 0.5043 | 0.7834 | 0.6136 | 0.7381 |
| Random forest | 0.8366 | 0.8123–0.8586 | 0.6445 | 0.5621 | 0.6898 | 0.6194 | 0.7750 |

## Initial interpretation

- The logistic baseline had slightly higher ROC-AUC and substantially higher recall.
- The random forest had higher accuracy, precision, average precision, and F1 at the default threshold.
- The confidence intervals overlap, so this run does not justify claiming that one model is universally superior.
- The right choice depends on the intervention: missing a customer who needs help and contacting too many customers have different costs.

## What this does not prove

This is an educational portfolio experiment, not evidence that a telecom provider should target customers. Before use, the team would need threshold/cost analysis, calibration, subgroup evaluation, privacy/legal review, causal or intervention testing, and monitoring for drift.
