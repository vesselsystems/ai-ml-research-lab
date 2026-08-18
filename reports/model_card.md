# Model card: offline churn prediction comparison

## Summary and status

This card describes the three candidates evaluated in this repository: a majority-class reference, class-weighted regularized logistic regression, and a class-weighted random forest. They predict the observed `Churn` label in the IBM Telco Customer Churn CSV. The models were evaluated offline; no fitted model is supplied for operational use, and no intervention was tested.

## Intended use

- Reproduce the split, preprocessing, metrics, and uncertainty calculation in this repository.
- Compare a simple reference, a transparent linear model, and a nonlinear model on this source file.
- Inspect how different fixed probability thresholds change the observed holdout precision/recall tradeoff.

## Excluded use

Do not use these results to deny service, change eligibility or pricing, rank individual customers for treatment, or claim that outreach will prevent churn. Do not treat a score as a causal effect, a fairness assessment, or evidence of performance on a different customer population. The repository does not establish an appropriate action, cost matrix, consent basis, or governance process.

## Data and provenance

- **Source:** [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv), pinned to commit `d5371f5d83a446ad5673cbcca3b814b926491f8a`.
- **Run input:** `data/raw/telco_churn.csv`, SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
- **Size and label:** 7,043 rows; 1,869 `Churn=Yes` labels (26.54%). The holdout had 1,409 rows and 374 positives.
- **Processing:** `customerID` is excluded. `Churn` is mapped from `Yes`/`No`. `TotalCharges` is parsed as numeric; 11 blank/unparseable values in this file become missing and are imputed. Other numeric features use median imputation and scaling; categorical features use most-frequent imputation and one-hot encoding.
- **Access:** the raw CSV is ignored by Git and is downloaded by `scripts/download_data.py`; CI uses an in-memory test fixture instead.

The source is historical and its collection, label definition, representativeness, and consent context are not established by this project.

## Evaluation and current results

Seed 42 produced one stratified 80/20 split. Five-fold stratified cross-validation was run on training rows only. The holdout was not used to fit preprocessing or tune a threshold. Metrics below are current holdout values; precision, recall, F1, and accuracy use threshold 0.5.

| Model | ROC-AUC | 95% bootstrap interval | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.5000 | 0.5000–0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic regression | 0.8413 | 0.8181–0.8629 | 0.6326 | 0.5043 | 0.7834 | 0.6136 | 0.7381 |
| Random forest | 0.8366 | 0.8123–0.8586 | 0.6445 | 0.5621 | 0.6898 | 0.6194 | 0.7750 |

The logistic model has the higher ROC-AUC in this run; the forest has higher average precision and F1 at 0.5. The 95% holdout ROC-AUC intervals overlap. That pattern is compatible with a metric-dependent tradeoff, not a reliable claim that one model is better in general.

## Uncertainty and decision artifact

The intervals are percentile intervals from 500 bootstrap resamples of one holdout. They describe uncertainty in these rows and do not account for sampling bias, time drift, label changes, or deployment behavior. Cross-validation ROC-AUC was 0.8459 ± 0.0124 for logistic regression and 0.8404 ± 0.0089 for the forest (mean ± fold standard deviation).

`reports/threshold_analysis.csv` records confusion counts and precision/recall/F1 at fixed thresholds from 0.2 to 0.7. At 0.5, logistic regression flagged 581 rows (precision 0.5043, recall 0.7834), while the forest flagged 459 (precision 0.5621, recall 0.6898). These numbers illustrate a tradeoff on the holdout; they do not select a threshold or establish that either model's probabilities are calibrated.

## Risks and limitations

- The target is observed churn, not response to an intervention, so predictions should not be described as causes or treatment effects.
- Contract, payment, tenure, demographic, and service fields may encode access patterns or proxy protected characteristics. No subgroup or fairness evaluation was run.
- No calibration, time-based validation, drift analysis, privacy/legal review, consent review, or intervention evaluation was run.
- A single public historical file and one random holdout may not represent another population or future period.
- Class imbalance makes accuracy alone misleading; the majority reference reaches 0.7346 accuracy while finding no positive cases.
