"""Validation and analysis for explicitly approved decision policies.

The ordinary threshold reports in :mod:`ai_ml_research_lab.experiment` remain
 descriptive. This module is a separate boundary for a reviewed policy: it
requires an approved policy record and never supplies costs or capacity on its
own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .experiment import DEFAULT_THRESHOLDS, _validate_binary_scores

_MISSING = object()
CAPACITY_SELECTION_RULE = "highest_score_then_input_order"


@dataclass(frozen=True)
class DecisionPolicy:
    """The validated fields needed for policy-driven threshold analysis."""

    policy_schema_version: int
    policy_version: str
    intended_action: str
    eligible_population: str
    max_actions: int
    capacity_period: str
    selection_rule: str
    cost_unit: str
    false_positive_cost: float
    false_negative_cost: float
    owner: str
    review_date: str
    excluded_uses: tuple[str, ...]
    status: str
    approved_by: str
    approved_at: str

    @property
    def capacity_selection_rule(self) -> str:
        """Return the deterministic capacity rule used by the report."""
        return self.selection_rule


def _required_text(policy: Mapping[str, Any], field: str) -> str:
    value = policy.get(field, _MISSING)
    if value is _MISSING or value is None:
        raise ValueError(f"decision policy field {field!r} is required")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"decision policy field {field!r} must be non-empty text")
    return value.strip()


def _approval_sources(policy: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = [policy]
    approval = policy.get("approval", _MISSING)
    if approval is not _MISSING:
        if not isinstance(approval, Mapping):
            raise ValueError("decision policy field 'approval' must be an object")
        sources.append(approval)
    return sources


def _approval_value(
    policy: Mapping[str, Any],
    fields: tuple[str, ...],
    label: str,
) -> list[Any]:
    values: list[Any] = []
    for source in _approval_sources(policy):
        for field in fields:
            if field in source:
                values.append(source[field])
    if not values:
        raise ValueError(f"decision policy {label} is required")
    return values


def _timezone_aware_timestamp(value: str, field: str) -> datetime:
    """Parse an ISO timestamp and require an explicit UTC offset."""
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    return parsed


def _validated_approval(policy: Mapping[str, Any]) -> tuple[str, str, str]:
    statuses = _approval_value(policy, ("status", "approval_status"), "approval status")
    normalized_statuses: list[str] = []
    for status in statuses:
        if not isinstance(status, str) or not status.strip():
            raise ValueError("decision policy approval status must be non-empty text")
        normalized_statuses.append(status.strip().lower())
    if any(status != "approved" for status in normalized_statuses):
        raise ValueError("decision policy must be approved before analysis")
    if len(set(normalized_statuses)) != 1:
        raise ValueError("decision policy approval status fields disagree")

    approvers = _approval_value(
        policy,
        ("approved_by", "approver"),
        "approved_by",
    )
    approval_times = _approval_value(
        policy,
        ("approved_at", "approval_date"),
        "approved_at",
    )

    for label, values in (("approved_by", approvers), ("approved_at", approval_times)):
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"decision policy {label} must be non-empty text")
        if len({value.strip() for value in values}) != 1:
            raise ValueError(f"decision policy {label} fields disagree")

    approved_at = approval_times[0].strip()
    _timezone_aware_timestamp(approved_at, "decision policy approved_at")

    return "approved", approvers[0].strip(), approved_at


def _numeric_value(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Integral, np.integer, np.floating)):
        raise ValueError(f"decision policy field {field!r} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"decision policy field {field!r} must be finite")
    if number < 0:
        raise ValueError(f"decision policy field {field!r} cannot be negative")
    return number


def _cost(policy: Mapping[str, Any], field: str) -> float:
    value = policy.get(field, _MISSING)
    if value is _MISSING or value is None:
        raise ValueError(f"decision policy field {field!r} is required")
    if isinstance(value, Mapping):
        nested_values = [value[key] for key in ("value", "amount", "cost") if key in value]
        if not nested_values:
            raise ValueError(f"decision policy field {field!r} needs a numeric value")
        if len(nested_values) > 1 and any(item != nested_values[0] for item in nested_values[1:]):
            raise ValueError(f"decision policy field {field!r} values disagree")
        value = nested_values[0]
    return _numeric_value(value, field)


def _capacity(policy: Mapping[str, Any]) -> tuple[int, str, str]:
    constraints = policy.get("capacity_constraints", _MISSING)
    if constraints is _MISSING or constraints is None:
        raise ValueError("decision policy field 'capacity_constraints' is required")
    if not isinstance(constraints, Mapping):
        raise ValueError("capacity_constraints must be an object")

    capacity_values = [
        constraints[key]
        for key in ("max_actions", "maximum_actions", "max_actions_per_period")
        if key in constraints
    ]
    if not capacity_values:
        raise ValueError("capacity_constraints must include max_actions")
    if len(capacity_values) > 1 and any(
        item != capacity_values[0] for item in capacity_values[1:]
    ):
        raise ValueError("capacity constraint values disagree")
    raw_capacity = capacity_values[0]

    raw_period = constraints.get("period", _MISSING)
    if raw_period is _MISSING or raw_period is None:
        raise ValueError("capacity_constraints period is required")
    if not isinstance(raw_period, str) or not raw_period.strip():
        raise ValueError("capacity_constraints period must be non-empty text")
    period = raw_period.strip()

    raw_selection_rule = constraints.get("selection_rule", _MISSING)
    if raw_selection_rule is _MISSING or raw_selection_rule is None:
        raise ValueError("capacity_constraints selection_rule is required")
    if not isinstance(raw_selection_rule, str) or raw_selection_rule != CAPACITY_SELECTION_RULE:
        raise ValueError(
            "capacity_constraints selection_rule must be "
            f"{CAPACITY_SELECTION_RULE!r}"
        )
    selection_rule = CAPACITY_SELECTION_RULE

    if isinstance(raw_capacity, bool) or not isinstance(
        raw_capacity, (Real, Integral, np.integer, np.floating)
    ):
        raise ValueError("capacity max_actions must be a finite non-negative integer")
    capacity_number = float(raw_capacity)
    if not isfinite(capacity_number):
        raise ValueError("capacity max_actions must be finite")
    if capacity_number < 0 or not capacity_number.is_integer():
        raise ValueError("capacity max_actions must be a non-negative integer")
    return int(capacity_number), period, selection_rule


def _excluded_uses(policy: Mapping[str, Any]) -> tuple[str, ...]:
    value = policy.get("excluded_uses", _MISSING)
    if value is _MISSING or value is None:
        raise ValueError("decision policy field 'excluded_uses' is required")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(value)
    else:
        raise ValueError("decision policy excluded_uses must be a non-empty list of text")
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError("decision policy excluded_uses must contain non-empty text")
    return tuple(item.strip() for item in values)


def _review_date(policy: Mapping[str, Any]) -> str:
    value = _required_text(policy, "review_date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("decision policy review_date must be an ISO date") from error
    if parsed < date.today():
        raise ValueError("decision policy review_date is expired")
    return value


def _policy_versions(policy: Mapping[str, Any]) -> tuple[int, str]:
    schema_version = policy.get("policy_schema_version", _MISSING)
    if schema_version is _MISSING:
        raise ValueError("decision policy policy_schema_version is required")
    if isinstance(schema_version, bool) or not isinstance(
        schema_version, (Integral, np.integer)
    ):
        raise ValueError("decision policy policy_schema_version must be an integer")
    if int(schema_version) != 1:
        raise ValueError("decision policy policy_schema_version must be exactly 1")

    policy_version = policy.get("policy_version", _MISSING)
    if policy_version is _MISSING or policy_version is None:
        raise ValueError("decision policy policy_version is required")
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError("decision policy policy_version must be non-empty text")
    return 1, policy_version.strip()


def validate_decision_policy(policy: Mapping[str, Any] | None) -> DecisionPolicy:
    """Validate and normalize a policy for cost/capacity analysis.

    A policy is intentionally stricter than the descriptive reporting helpers.
    It must identify the action and population, provide non-negative finite
    costs and an explicit non-negative integer capacity, and carry approval
    evidence with an ``approved`` status.  No value is inferred when a field is
    absent.
    """
    if policy is None:
        raise ValueError("an approved decision policy is required")
    if not isinstance(policy, Mapping):
        raise ValueError("decision policy must be an object")

    schema_version, policy_version = _policy_versions(policy)
    status, approved_by, approved_at = _validated_approval(policy)
    max_actions, capacity_period, selection_rule = _capacity(policy)
    return DecisionPolicy(
        policy_schema_version=schema_version,
        policy_version=policy_version,
        intended_action=_required_text(policy, "intended_action"),
        eligible_population=_required_text(policy, "eligible_population"),
        max_actions=max_actions,
        capacity_period=capacity_period,
        selection_rule=selection_rule,
        cost_unit=_required_text(policy, "cost_unit"),
        false_positive_cost=_cost(policy, "false_positive_cost"),
        false_negative_cost=_cost(policy, "false_negative_cost"),
        owner=_required_text(policy, "owner"),
        review_date=_review_date(policy),
        excluded_uses=_excluded_uses(policy),
        status=status,
        approved_by=approved_by,
        approved_at=approved_at,
    )


def _threshold_values(thresholds: Sequence[float]) -> np.ndarray:
    if isinstance(thresholds, (str, bytes, bytearray)):
        raise ValueError("thresholds must be a sequence of numbers")
    try:
        values = list(thresholds)
    except TypeError as error:
        raise ValueError("thresholds must be a sequence of numbers") from error
    if not values:
        raise ValueError("thresholds cannot be empty")

    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(
            value, (Real, Integral, np.integer, np.floating)
        ):
            raise ValueError("thresholds must be numeric")
        number = float(value)
        if not isfinite(number):
            raise ValueError("thresholds must be finite")
        if not 0 <= number <= 1:
            raise ValueError("thresholds must be between 0 and 1")
        normalized.append(number)
    return np.asarray(normalized, dtype=float)


def policy_threshold_analysis(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    policy: Mapping[str, Any] | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Calculate cost and capacity tradeoffs for an approved policy.

    ``y_true`` and ``probabilities`` must represent the policy's eligible
    population.  Scores at or above a threshold are candidates.  When the
    candidate count exceeds the policy's hard ``max_actions`` capacity, the
    highest scores are selected, with stable input order breaking ties.  The
    returned confusion counts and costs describe the selected actions, while
    ``candidate_positive_count`` exposes how many rows the threshold would
    otherwise flag.
    """
    validated = validate_decision_policy(policy)
    y_array, score_array = _validate_binary_scores(y_true, probabilities)
    threshold_array = _threshold_values(thresholds)

    columns = [
        "threshold",
        "rows",
        "actual_positive_rate",
        "candidate_positive_count",
        "predicted_positive_count",
        "action_count",
        "capacity",
        "capacity_exceeded",
        "capacity_utilization",
        "cost_unit",
        "predicted_positive_rate",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "precision",
        "recall",
        "f1",
        "false_positive_cost",
        "false_negative_cost",
        "total_cost",
    ]
    rows: list[dict[str, Any]] = []
    for threshold in threshold_array:
        candidate_predictions = (score_array >= threshold).astype("int64")
        candidate_indices = np.flatnonzero(candidate_predictions)
        candidate_count = int(candidate_indices.size)
        capacity_exceeded = candidate_count > validated.max_actions

        if capacity_exceeded:
            ordered = candidate_indices[
                np.argsort(-score_array[candidate_indices], kind="mergesort")
            ]
            selected_indices = ordered[: validated.max_actions]
        else:
            selected_indices = candidate_indices

        predictions = np.zeros(len(y_array), dtype="int64")
        predictions[selected_indices] = 1
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            y_array,
            predictions,
            labels=[0, 1],
        ).ravel()
        action_count = int(predictions.sum())
        total_cost = float(
            validated.false_positive_cost * false_positive
            + validated.false_negative_cost * false_negative
        )
        capacity_utilization = (
            float(action_count / validated.max_actions) if validated.max_actions else 0.0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "rows": int(len(y_array)),
                "actual_positive_rate": float(np.mean(y_array)),
                "candidate_positive_count": candidate_count,
                "predicted_positive_count": action_count,
                "action_count": action_count,
                "capacity": validated.max_actions,
                "capacity_exceeded": capacity_exceeded,
                "capacity_utilization": capacity_utilization,
                "cost_unit": validated.cost_unit,
                "predicted_positive_rate": float(predictions.mean()),
                "true_positive": int(true_positive),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_negative": int(true_negative),
                "precision": float(precision_score(y_array, predictions, zero_division=0)),
                "recall": float(recall_score(y_array, predictions, zero_division=0)),
                "f1": float(f1_score(y_array, predictions, zero_division=0)),
                "false_positive_cost": validated.false_positive_cost,
                "false_negative_cost": validated.false_negative_cost,
                "total_cost": total_cost,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def decision_policy_analysis(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    policy: Mapping[str, Any] | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Alias for :func:`policy_threshold_analysis` with an explicit name."""
    return policy_threshold_analysis(
        y_true,
        probabilities,
        policy=policy,
        thresholds=thresholds,
    )
