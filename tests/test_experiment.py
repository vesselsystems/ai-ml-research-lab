import pandas as pd

from ai_ml_research_lab.data import split_features_target
from ai_ml_research_lab.experiment import run_experiments


def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(20)],
            "tenure": list(range(20)),
            "MonthlyCharges": [50 + i for i in range(20)],
            "TotalCharges": [50 * (i + 1) for i in range(20)],
            "Contract": ["Month-to-month", "One year"] * 10,
            "Churn": ["Yes", "No"] * 10,
        }
    )


def test_identifier_is_excluded_and_target_is_binary() -> None:
    features, target = split_features_target(tiny_frame())

    assert "customerID" not in features.columns
    assert set(target.unique()) == {0, 1}


def test_experiment_returns_both_models_and_metrics() -> None:
    summary, fitted = run_experiments(
        tiny_frame(),
        n_splits=2,
        forest_estimators=10,
        test_size=0.25,
    )

    assert set(summary["model"]) == {"logistic_regression", "random_forest"}
    assert set(fitted) == set(summary["model"])
    assert summary["test_roc_auc"].notna().all()
    assert summary["test_roc_auc_ci_low"].notna().all()
