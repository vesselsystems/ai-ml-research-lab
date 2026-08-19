import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ai_ml_research_lab.experiment as experiment_module
from ai_ml_research_lab.data import split_features_target
from ai_ml_research_lab.decision_policy import (
    policy_threshold_analysis,
    validate_decision_policy,
)
from ai_ml_research_lab.experiment import (
    REPORTING_THRESHOLD,
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


def approved_policy(max_actions: int = 10) -> dict[str, object]:
    # Synthetic, test-only policy fixture; no approved policy is checked in.
    return {
        "policy_schema_version": 1,
        "policy_version": "1.0",
        "intended_action": "Offer a reviewed retention contact",
        "eligible_population": "Rows in the separately reviewed evaluation population",
        "capacity_constraints": {
            "max_actions": max_actions,
            "period": "per review period",
            "selection_rule": "highest_score_then_input_order",
        },
        "cost_unit": "reviewed cost units per action",
        "false_positive_cost": 10.0,
        "false_negative_cost": 25.0,
        "owner": "retention-program-owner",
        "review_date": "2099-12-31",
        "excluded_uses": ["Do not deny service or change pricing"],
        "status": "approved",
        "approved_by": "independent-reviewer",
        "approved_at": "2025-01-01T00:00:00Z",
    }


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


def test_run_experiments_uses_reporting_threshold_at_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_train = pd.DataFrame({"feature": [0, 1]})
    x_test = pd.DataFrame({"feature": [2, 3]})
    y_train = pd.Series([0, 1])
    y_test = pd.Series([1, 0])
    probabilities = np.array([REPORTING_THRESHOLD, np.nextafter(REPORTING_THRESHOLD, 0)])

    class BoundaryPipeline:
        def fit(self, features: pd.DataFrame, target: pd.Series) -> "BoundaryPipeline":
            return self

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            raise AssertionError("run_experiments should use its explicit reporting comparator")

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.column_stack((1 - probabilities, probabilities))

    monkeypatch.setattr(
        experiment_module,
        "_holdout_split",
        lambda frame, test_size, random_state: (x_train, x_test, y_train, y_test),
    )
    monkeypatch.setattr(
        experiment_module,
        "build_pipelines",
        lambda features, random_state, forest_estimators: {"boundary": BoundaryPipeline()},
    )
    monkeypatch.setattr(
        experiment_module,
        "cross_validate",
        lambda pipeline, features, target, cv, scoring: {
            f"test_{name}": np.array([1.0]) for name in scoring
        },
    )
    monkeypatch.setattr(
        experiment_module,
        "bootstrap_roc_auc_ci",
        lambda *args, **kwargs: (1.0, 1.0),
    )

    summary, _ = run_experiments(pd.DataFrame({"feature": [0, 1, 2, 3]}), n_splits=2)

    row = summary.iloc[0]
    assert row["test_accuracy"] == 1.0
    assert row["test_precision"] == 1.0
    assert row["test_recall"] == 1.0
    assert row["test_f1"] == 1.0


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


def test_policy_analysis_requires_an_explicit_policy() -> None:
    y_true = pd.Series([0, 1])
    probabilities = np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="approved decision policy"):
        policy_threshold_analysis(y_true, probabilities)


def test_policy_template_is_machine_readable_and_stays_unapproved() -> None:
    template_path = Path(__file__).parents[1] / "policies" / "decision_policy.v1.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))

    assert {
        "policy_schema_version",
        "policy_version",
        "intended_action",
        "eligible_population",
        "capacity_constraints",
        "cost_unit",
        "false_positive_cost",
        "false_negative_cost",
        "owner",
        "review_date",
        "excluded_uses",
        "status",
        "approved_by",
        "approved_at",
    } <= template.keys()
    assert template["policy_schema_version"] == 1
    assert template["capacity_constraints"]["period"] is None
    assert template["capacity_constraints"]["selection_rule"] is None
    with pytest.raises(ValueError, match="approved"):
        validate_decision_policy(template)


def test_valid_supplied_policy_is_accepted() -> None:
    policy = validate_decision_policy(approved_policy())

    assert policy.status == "approved"
    assert policy.max_actions == 10
    assert policy.capacity_period == "per review period"
    assert policy.selection_rule == "highest_score_then_input_order"
    assert policy.cost_unit == "reviewed cost units per action"
    assert policy.false_positive_cost == 10.0
    assert policy.false_negative_cost == 25.0


def test_policy_analysis_handles_capacity_without_inventing_actions() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])

    report = policy_threshold_analysis(
        y_true,
        probabilities,
        policy=approved_policy(max_actions=2),
        thresholds=(0.3,),
    )

    row = report.iloc[0]
    assert row["candidate_positive_count"] == 3
    assert row["predicted_positive_count"] == 2
    assert row["action_count"] == 2
    assert bool(row["capacity_exceeded"])
    assert row["capacity"] == 2


def test_policy_analysis_reports_supplied_threshold_cost_tradeoffs() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])

    report = policy_threshold_analysis(
        y_true,
        probabilities,
        policy=approved_policy(),
        thresholds=(0.3, 0.9),
    )

    assert list(report["false_positive"]) == [1, 0]
    assert list(report["false_negative"]) == [0, 1]
    assert list(report["total_cost"]) == [10.0, 25.0]
    assert list(report["cost_unit"]) == ["reviewed cost units per action"] * 2


def test_policy_capacity_rule_is_deterministic_for_equal_scores() -> None:
    policy = approved_policy(max_actions=2)
    report = policy_threshold_analysis(
        pd.Series([0, 1, 1]),
        np.array([0.9, 0.9, 0.8]),
        policy=policy,
        thresholds=(0.5,),
    )

    row = report.iloc[0]
    assert row["candidate_positive_count"] == 3
    assert row["predicted_positive_count"] == 2
    assert row["true_positive"] == 1
    assert row["false_positive"] == 1


@pytest.mark.parametrize("missing_field", ["period", "selection_rule"])
def test_policy_capacity_requires_explicit_period_and_selection_rule(
    missing_field: str,
) -> None:
    policy = copy.deepcopy(approved_policy())
    del policy["capacity_constraints"][missing_field]  # type: ignore[index]

    with pytest.raises(ValueError, match=missing_field):
        validate_decision_policy(policy)


def test_policy_capacity_rejects_unsupported_selection_rule() -> None:
    policy = copy.deepcopy(approved_policy())
    policy["capacity_constraints"]["selection_rule"] = "input_order"  # type: ignore[index]

    with pytest.raises(ValueError, match="highest_score_then_input_order"):
        validate_decision_policy(policy)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_schema_version", 2),
        ("policy_schema_version", 1.0),
        ("approved_at", "2025-01-01T00:00:00"),
        ("review_date", "2000-01-01"),
    ],
)
def test_policy_rejects_invalid_version_and_dates(field: str, value: object) -> None:
    policy = copy.deepcopy(approved_policy())
    policy[field] = value

    with pytest.raises(ValueError):
        validate_decision_policy(policy)


def test_policy_requires_both_version_fields() -> None:
    for field in ("policy_schema_version", "policy_version"):
        policy = copy.deepcopy(approved_policy())
        del policy[field]

        with pytest.raises(ValueError, match="required"):
            validate_decision_policy(policy)


def test_policy_rejects_string_excluded_uses() -> None:
    policy = copy.deepcopy(approved_policy())
    policy["excluded_uses"] = "Do not deny service"

    with pytest.raises(ValueError, match="excluded_uses"):
        validate_decision_policy(policy)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("status",), "draft", "approved"),
        (("false_positive_cost",), float("nan"), "finite"),
        (("false_negative_cost",), -1.0, "negative"),
        (("capacity_constraints", "max_actions"), float("inf"), "finite"),
        (("capacity_constraints", "max_actions"), -1, "non-negative"),
    ],
)
def test_policy_analysis_rejects_malformed_values(
    path: tuple[str, ...], value: object, message: str
) -> None:
    policy = copy.deepcopy(approved_policy())
    target: dict[str, object] = policy
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_decision_policy(policy)
