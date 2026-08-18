# Experiment protocol and run record

## Question

On the IBM Telco Customer Churn CSV, does a random forest provide better predictive ranking than regularized logistic regression? The comparison also asks how the apparent choice changes when a probability threshold trades false positives against false negatives.

This is a predictive comparison. It is not a test of a retention intervention and does not identify causes of churn.

## Data and provenance

- Source: [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv), pinned to commit `d5371f5d83a446ad5673cbcca3b814b926491f8a`.
- Local file: `data/raw/telco_churn.csv` (downloaded by `scripts/download_data.py`; ignored by Git).
- Current file checksum: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` (SHA-256).
- Size: 7,043 rows, 21 columns, 1,869 `Churn=Yes` labels (26.54%).
- Target: `Yes` becomes 1 and `No` becomes 0. `customerID` is removed. `TotalCharges` is parsed as numeric so its blank values can be imputed.

The file is a historical public dataset. The checksum makes this run identifiable; it does not establish that the data represent another population or that the labels are free of measurement bias.

## Fixed design

- Random seed: 42.
- Split: one stratified 80/20 train/holdout split; the holdout contains 1,409 rows and 374 positives (26.54%).
- Cross-validation: five shuffled, stratified folds on the training rows only.
- Preprocessing: numeric median imputation and standardization; categorical most-frequent imputation and one-hot encoding with unknown categories ignored. Each step is inside the candidate pipeline.
- Candidates: majority-class reference, class-weighted regularized logistic regression, and class-weighted random forest (`n_estimators=200`, `min_samples_leaf=3`).
- Primary comparison measures: ROC-AUC and average precision. Brier score and log loss assess probabilistic predictions. Precision, recall, F1, and accuracy are reported at probability threshold 0.5.
- Uncertainty: 500 bootstrap resamples of the holdout for a percentile interval on ROC-AUC. Cross-validation spread is reported as the fold standard deviation.
- Additional artifacts: `reports/threshold_analysis.csv` applies fixed thresholds 0.2, 0.3, 0.4, 0.5, 0.6, and 0.7 to the holdout and records confusion counts and thresholded metrics; `reports/calibration_analysis.csv` uses ten fixed equal-width probability bins; and `reports/error_analysis.csv` decomposes threshold-0.5 errors by score band. None of these artifacts selects an operating threshold or fits a calibration model.

No hyperparameter search, recalibration, or threshold tuning was performed against the holdout. The threshold grid is a descriptive convention. No action, capacity, or business cost is provided; a future threshold must be selected with a reviewed action and supplied false-positive/false-negative costs on training/validation data or a separate decision set. Run `python scripts/run_experiment.py` after downloading the CSV to regenerate the machine-readable summaries.

## Hypothesis

A nonlinear forest may improve ranking or thresholded classification over a transparent linear baseline. The protocol does not assume that one metric defines a universally best model: the relevant tradeoff depends on a documented action and its costs, neither of which is supplied by this dataset.

## Results from the current run

| Model | Test ROC-AUC | 95% bootstrap interval | CV ROC-AUC mean ± SD | Test average precision | Test precision | Test recall | Test F1 | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.5000 | 0.5000–0.5000 | 0.5000 ± 0.0000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic regression | 0.8413 | 0.8181–0.8629 | 0.8459 ± 0.0124 | 0.6326 | 0.5043 | 0.7834 | 0.6136 | 0.7381 |
| Random forest | 0.8366 | 0.8123–0.8586 | 0.8404 ± 0.0089 | 0.6445 | 0.5621 | 0.6898 | 0.6194 | 0.7750 |

The logistic model ranked slightly higher by holdout and cross-validation ROC-AUC. The forest had higher holdout average precision, precision, F1, and accuracy at 0.5, while the logistic model recovered more positives. The bootstrap intervals overlap, and the differences are small enough that this run does not justify a universal winner.

The threshold artifact makes the tradeoff concrete. At threshold 0.5, logistic regression flagged 581 of 1,409 rows (precision 0.5043, recall 0.7834); the forest flagged 459 (precision 0.5621, recall 0.6898). At threshold 0.6, the logistic model reached precision 0.5399 and recall 0.7059, while the forest reached precision 0.6261 and recall 0.5775. These are holdout descriptions, not evidence that either threshold is appropriate for customer action. The error artifact retains the same threshold-0.5 errors by score band without exporting identifiers or feature values.

The calibration artifact reports these same-holdout diagnostics:

| Model | Brier score | Log loss | ECE | Maximum absolute bin gap |
|---|---:|---:|---:|---:|
| Majority class | 0.2654 | 9.5673 | 0.2654 | 0.2654 |
| Logistic regression | 0.1688 | 0.4969 | 0.1504 | 0.3171 |
| Random forest | 0.1491 | 0.4524 | 0.0860 | 0.1826 |

The forest has lower holdout Brier score, log loss, ECE, and maximum bin gap in this run, but the single random holdout does not establish future calibration or a universally preferred model.

## Leakage and governance checks

- The identifier is removed before modeling.
- Imputation, scaling, and one-hot encoding are learned within each training fold or training split.
- The target is not used as a feature.
- The source contains contract, payment, tenure, demographic, and service fields. They may be useful predictors while also acting as proxies for protected characteristics or unequal treatment; no subgroup or fairness result is claimed and this run does not evaluate that risk.
- The label records observed churn, not the outcome under a specified intervention. A high score should not be described as a cause or as proof that outreach will work.

## What would change the conclusion

A different representative sample, a defensible time-based validation design, a separately evaluated recalibration process, an agreed cost matrix, and a separately evaluated intervention could change the model or threshold choice. This study makes no time-based, subgroup, or fairness claim. Drift, missingness changes, privacy/legal requirements, and consent would also need evidence before any use beyond this offline comparison.
