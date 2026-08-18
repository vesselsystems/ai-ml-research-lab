# AI/ML Research Lab: Churn Prediction and Responsible Evaluation

A reproducible research-style ML project following the AI Career Training Plan's second project:

> Predictive model (churn, fraud, or demand) with a full write-up.

The project compares a regularized logistic-regression baseline with a random-forest model on a public telecom churn dataset. It emphasizes experimental discipline rather than leaderboard chasing: a fixed split, stratified cross-validation, multiple metrics, a leakage review, and a model-card-style limitations section.

## Research questions

1. Does a nonlinear tree ensemble improve ranking performance over a transparent linear baseline?
2. How do model choices change precision, recall, and F1 on an imbalanced target?
3. Which data-quality and governance decisions must be documented before a churn score could be used responsibly?

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_experiment.py
pytest
ruff check .
```

Outputs are written to `reports/` and are intentionally reproducible with the configured random seed.

## Method

- Public dataset: IBM Telco Customer Churn CSV.
- Target: `Churn` converted to a binary label.
- Identifier: `customerID` is excluded from modeling.
- Numeric fields: median imputation and standardization for the linear model.
- Categorical fields: most-frequent imputation and one-hot encoding.
- Evaluation: stratified train/test split plus 5-fold stratified cross-validation.
- Metrics: ROC-AUC, average precision, precision, recall, F1, and accuracy.
- Responsible-use review: leakage, missingness, class imbalance, proxy variables, calibration, and the fact that correlation is not a causal retention intervention.

## Repository structure

```text
.
├── docs/methodology.md
├── reports/experiment_protocol.md
├── scripts/
│   ├── download_data.py
│   └── run_experiment.py
├── src/ai_ml_research_lab/
│   ├── data.py
│   └── experiment.py
└── tests/
```

## Portfolio deliverable

The finished repository should contain:

- a model comparison table
- a short research note explaining the question, design, results, and limitations
- a model card describing intended use, non-use, data, metrics, and risks
- reproducible commands and tests

This is a portfolio study, not a production retention system. A real deployment would require stakeholder review, consent/legal review, drift monitoring, calibration, and an intervention policy that is tested separately from prediction quality.
