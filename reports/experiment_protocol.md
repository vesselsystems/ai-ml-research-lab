# Experiment protocol

This file is the research log for the first reproducible run.

## Hypothesis

A nonlinear random forest may improve ranking performance over a transparent logistic-regression baseline, but the best model should be selected using the metric and intervention that a real stakeholder can justify—not ROC-AUC alone.

## Planned comparison

- Same source data
- Same fixed stratified split
- Same cross-validation folds
- Same imputation/encoding contract
- Logistic regression baseline vs. random forest comparison
- Metrics: ROC-AUC, average precision, precision, recall, F1, and accuracy
- Bootstrap 95% interval for held-out ROC-AUC

## Interpretation template

After running the experiment, record:

- Which model ranked first and by which metric?
- Did cross-validation and held-out results agree?
- What is the precision/recall tradeoff at the default threshold?
- Which features may be proxies or create an unfair intervention?
- What additional data or experiment would change the conclusion?

Do not present a model score without its split strategy, metric definition, uncertainty, and limitations.
