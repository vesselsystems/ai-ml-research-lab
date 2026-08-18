"""Model pipelines and reproducible evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42



def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a leakage-safe preprocessing graph from the training schema."""
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_pipelines(
    features: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    forest_estimators: int = 200,
) -> dict[str, Pipeline]:
    """Return the transparent baseline and the nonlinear comparison model."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor(features)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor(features)),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        n_estimators=forest_estimators,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def bootstrap_roc_auc_ci(
    y_true: pd.Series,
    scores: np.ndarray,
    n_bootstraps: int = 500,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Estimate a percentile confidence interval for ROC-AUC."""
    rng = np.random.default_rng(random_state)
    y_array = np.asarray(y_true)
    score_array = np.asarray(scores)
    estimates: list[float] = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_array), len(y_array))
        sample_y = y_array[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        estimates.append(roc_auc_score(sample_y, score_array[indices]))

    if not estimates:
        return float("nan"), float("nan")
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def run_experiments(
    frame: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    test_size: float = 0.2,
    n_splits: int = 5,
    forest_estimators: int = 200,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Fit each candidate and return a comparable test/CV summary plus fitted models."""
    from .data import split_features_target

    features, target = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []
    for name, pipeline in build_pipelines(
        x_train,
        random_state=random_state,
        forest_estimators=forest_estimators,
    ).items():
        cv_result = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        test_metrics = _metrics(y_test, predictions, probabilities)
        auc_low, auc_high = bootstrap_roc_auc_ci(
            y_test,
            probabilities,
            random_state=random_state,
        )
        fitted[name] = pipeline
        rows.append(
            {
                "model": name,
                "test_rows": len(y_test),
                "positive_rate": float(target.mean()),
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "test_roc_auc_ci_low": auc_low,
                "test_roc_auc_ci_high": auc_high,
                **{
                    f"cv_{key}_mean": float(cv_result[f"test_{key}"].mean())
                    for key in scoring
                },
                **{
                    f"cv_{key}_std": float(cv_result[f"test_{key}"].std())
                    for key in scoring
                },
            }
        )

    return pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False), fitted
