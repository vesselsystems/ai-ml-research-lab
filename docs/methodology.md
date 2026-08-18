# Methodology and research controls

## Design

The study compares two model families under the same data split and preprocessing contract:

1. A class-weighted logistic regression as an interpretable baseline.
2. A class-weighted random forest as a nonlinear comparison.

All imputers, encoders, and scalers are inside a scikit-learn `Pipeline` so test information cannot leak into preprocessing. The identifier is removed before modeling.

## Evaluation

- The target is stratified into a fixed 80/20 train/test split using seed 42.
- The training set is evaluated with five-fold stratified cross-validation.
- The held-out test set is used once for the final comparison.
- ROC-AUC measures ranking across thresholds; average precision is useful for an imbalanced positive class; precision/recall/F1 show a thresholded tradeoff.
- A bootstrap percentile interval is reported for held-out ROC-AUC. It describes uncertainty in this sample; it is not a guarantee of future performance.

## Governance questions

Before using a churn score, a real team would need to answer:

- Is the prediction used to offer support fairly, or to deny service?
- Are contract, payment, tenure, or demographic fields proxies for protected attributes or unequal treatment?
- Does the organization have consent and a documented retention intervention?
- Is the probability calibrated well enough for the decision threshold?
- How will drift, missingness, and changes in customer behavior be monitored?

## Reproduction

Run `python scripts/download_data.py` followed by `python scripts/run_experiment.py`. The raw data and generated reports are excluded from Git by default; the commands and code are the reproducible source of truth.
