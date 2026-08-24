# Decision-policy boundary

`policies/decision_policy.v1.json` is a versioned, machine-readable template for
an action that might be considered separately from this offline model study. It
is intentionally a **draft**: its `null` values are required inputs, not
suggested defaults. No completed or approved policy is checked into this
repository. An approved policy would be necessary but not sufficient: pending
retrieval, license/terms, permission, privacy, or consent review would still
block redistribution or operational use. This helper does not resolve those
reviews.

## Required policy record

A completed policy must document all of the following:

- `intended_action`: the concrete action that a positive model result would
  trigger.
- `eligible_population`: the population to which the action and evaluation
  apply. The arrays supplied to analysis must already represent this population.
- `capacity_constraints.max_actions`: a non-negative integer hard limit on
  actions.
- `capacity_constraints.period`: the required non-empty text describing the
  capacity window.
- `capacity_constraints.selection_rule`: exactly
  `highest_score_then_input_order`. This is the only supported rule: select the
  highest scores first and preserve input order for equal scores.
- `cost_unit`: the required shared unit or basis for both costs (for example, a
  reviewed currency-per-action basis). The validator does not invent or
  convert units.
- `false_positive_cost` and `false_negative_cost`: finite, non-negative numeric
  costs in `cost_unit`. Zero is allowed when it is an explicit, reviewed value;
  a missing value is not zero.
- `owner`: the accountable owner.
- `review_date`: a non-expired ISO date for the next review.
- `excluded_uses`: a non-empty list of uses that are out of scope; a single
  string is not accepted as a list.
- `status`, `approved_by`, and `approved_at`: approval evidence. Policy-driven
  analysis accepts only `status: "approved"` with a named approver and a
  timezone-aware ISO timestamp. Draft, pending, retired, or otherwise
  unapproved records are rejected.

`policy_schema_version` is required and must be exactly `1`. `policy_version`
is also required and must be non-empty text. A policy can place approval fields
in an `approval` object; the validator accepts that equivalent representation,
but the checked-in template uses the top-level fields above. An expired
`review_date` is rejected.

## Policy-driven analysis

`ai_ml_research_lab.decision_policy.policy_threshold_analysis` is separate from
the descriptive threshold report. It requires the completed approved policy and
never supplies an action, owner, capacity, or cost. It reports the supplied
false-positive/false-negative cost for each threshold as:

```text
total_cost = false_positive_cost * false_positives
           + false_negative_cost * false_negatives
```

Scores at or above a threshold are candidates. If candidate count exceeds
`max_actions`, the documented `highest_score_then_input_order` rule selects the
highest scores up to that hard cap; stable input order breaks equal-score ties.
The report keeps both the candidate count and the selected action count, so
capacity-limited results are visible, and includes the shared `cost_unit`.
This is a tradeoff table, not causal evidence or an automatic authorization to
act.

For example, a caller must supply a policy mapping rather than relying on an
implicit cost matrix:

```python
from ai_ml_research_lab.decision_policy import policy_threshold_analysis

report = policy_threshold_analysis(
    y_true,
    probabilities,
    policy=approved_policy,
    thresholds=(0.3, 0.5, 0.7),
)
```

With no owner, reviewed costs, capacity, and approval, threshold selection
remains blocked. The repository's existing `threshold_analysis` grid and
`REPORTING_THRESHOLD = 0.5` are descriptive reporting conventions only; they
do not select an operating threshold and do not imply that retention outreach
works.
