import numpy as np
import pandas as pd

from ai_ml_research_lab.data import split_features_target
from ai_ml_research_lab.experiment import (
    calibration_analysis,
    error_analysis,
    run_calibration_analysis,
    run_error_analysis,
    run_experiments,
    run_threshold_analysis,
    threshold_analysis,
)


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

    assert set(summary["model"]) == {
        "majority_class",
        "logistic_regression",
        "random_forest",
    }
    assert set(fitted) == set(summary["model"])
    assert summary["test_roc_auc"].notna().all()
    assert summary["test_roc_auc_ci_low"].notna().all()
    assert summary[["test_brier_score", "test_log_loss"]].notna().all().all()

    majority = summary.loc[summary["model"] == "majority_class"].iloc[0]
    assert majority["test_roc_auc"] == 0.5
    assert fitted["majority_class"].named_steps["model"].strategy == "most_frequent"


def test_threshold_analysis_reports_decision_tradeoff() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])

    report = threshold_analysis(y_true, probabilities, thresholds=(0.3, 0.8))

    assert list(report["threshold"]) == [0.3, 0.8]
    assert list(report["predicted_positive_count"]) == [3, 2]
    assert list(report["true_positive"]) == [2, 2]
    assert list(report["false_positive"]) == [1, 0]
    assert report.loc[0, "recall"] == 1.0
    assert report.loc[1, "precision"] == 1.0


def test_calibration_analysis_uses_fixed_bins_and_reports_summary_metrics() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])

    report = calibration_analysis(y_true, probabilities, n_bins=4)

    assert list(report["rows"]) == [1, 1, 0, 2]
    assert report.loc[0, "observed_positive_rate"] == 0.0
    assert report.loc[3, "observed_positive_rate"] == 1.0
    assert np.isclose(report["expected_calibration_error"].iloc[0], 0.2)
    assert np.isclose(report["brier_score"].iloc[0], 0.055)
    assert report["log_loss"].notna().all()


def test_error_analysis_reconciles_confusion_counts_by_score_band() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])

    report = error_analysis(y_true, probabilities, threshold=0.5, n_bins=4)

    assert report["rows"].sum() == 4
    assert report["true_positive"].sum() == 2
    assert report["true_negative"].sum() == 2
    assert report["false_positive"].sum() == 0
    assert report["false_negative"].sum() == 0
    assert report["errors"].sum() == 0


def test_calibration_and_error_reports_use_all_fitted_models() -> None:
    frame = tiny_frame()
    _, fitted = run_experiments(
        frame,
        n_splits=2,
        forest_estimators=10,
        test_size=0.25,
    )

    calibration = run_calibration_analysis(
        frame,
        fitted,
        test_size=0.25,
        n_bins=4,
    )
    errors = run_error_analysis(
        frame,
        fitted,
        test_size=0.25,
        n_bins=4,
    )

    assert set(calibration["model"]) == set(fitted)
    assert len(calibration) == len(fitted) * 4
    assert calibration.groupby("model")["rows"].sum().eq(5).all()
    assert set(errors["model"]) == set(fitted)
    assert len(errors) == len(fitted) * 4
    assert errors.groupby("model")["rows"].sum().eq(5).all()


def test_threshold_report_uses_all_fitted_models() -> None:
    frame = tiny_frame()
    _, fitted = run_experiments(
        frame,
        n_splits=2,
        forest_estimators=10,
        test_size=0.25,
    )

    report = run_threshold_analysis(
        frame,
        fitted,
        test_size=0.25,
        thresholds=(0.5,),
    )

    assert set(report["model"]) == set(fitted)
    assert len(report) == len(fitted)
    assert set(report["rows"]) == {5}
