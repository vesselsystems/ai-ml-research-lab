# Methodology

## Question and scope

The experiment compares a transparent linear model with a nonlinear model for the binary label `Churn`. The question is predictive and dataset-specific: do the models rank the observed positive labels differently on the same held-out rows? A churn prediction is not a treatment-effect estimate, and this repository does not test whether contacting a customer changes churn.

## Data contract

The input is the public [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv), pinned to commit `d5371f5d83a446ad5673cbcca3b814b926491f8a`. The current local file has 7,043 rows and 21 columns, with 1,869 `Yes` labels (26.54%). Its SHA-256 is `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.

`Churn` is mapped from `Yes`/`No` to 1/0. `customerID` is excluded before the split. `TotalCharges` is converted to numeric; values that cannot be parsed become missing and are imputed within the model pipeline. The other columns are divided by pandas' numeric-vs-nonnumeric types.

This is a historical public dataset. The repository records the source and local checksum, but it does not claim that the file is current, representative of another telecom population, or free of measurement and labeling problems.

## Split and candidate models

Seed 42 creates one stratified 80/20 train/holdout split. Five-fold stratified cross-validation is run only on the training rows. The holdout is not used to fit preprocessing, choose a model, or tune a threshold.

Three references are evaluated:

1. **Majority class:** predicts the most common training label. This shows how much accuracy can be obtained without using features; its zero recall is expected on this imbalanced target.
2. **Logistic regression:** regularized, class-weighted, and preceded by median/mode imputation, standardization of numeric values, and one-hot encoding of categorical values.
3. **Random forest:** class-weighted, with 200 trees and a minimum leaf size of 3, using the same preprocessing contract.

Every learned preprocessing step is inside the candidate's scikit-learn `Pipeline`. Cross-validation therefore fits preprocessing separately within each training fold rather than reusing information from validation rows.

## Measures and uncertainty

ROC-AUC measures ranking across thresholds. Average precision describes ranking with the positive-class prevalence in mind. Precision, recall, F1, and accuracy use the model's default 0.5 probability threshold. The holdout ROC-AUC interval is a 2.5th–97.5th percentile interval from 500 bootstrap resamples. It describes sampling variation in these holdout rows; it is not a forecast interval and does not cover population shift.

`reports/threshold_analysis.csv` reports counts and precision/recall/F1 at fixed thresholds from 0.2 to 0.7. It is a decision-useful view of the tradeoff between false positives and false negatives, not a claim that 0.5 or any other value is an appropriate operating threshold. A threshold used outside this analysis would need to be fixed using a validation process and an explicit, reviewed cost or intervention policy.

## Reproduction and controls

From a clean checkout:

```bash
python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_experiment.py
pytest
ruff check .
```

The download is separate because the raw CSV is ignored. CI runs the tests with a small in-memory fixture and does not require network access or ignored data. The experiment script fails with a clear instruction when the CSV has not been downloaded. The seed, split, model settings, source URL, and generated artifacts are recorded in `reports/experiment_protocol.md`.

## Interpretation boundaries

The results can support a cautious comparison of these model families on this file. They cannot support a claim that one model is universally better, that a score is calibrated for a particular action, that a feature causes churn, or that an intervention would improve retention. The dataset includes contract, payment, tenure, demographic, and service fields that may be proxies for protected characteristics. No subgroup, fairness, calibration, privacy/legal, drift, or intervention evaluation is included in this study.
