# Human-review workflow

[`reviews/review_queue.v1.json`](../reviews/review_queue.v1.json) is the only
place for the remaining human inputs in this repository. It does not change
`data/provenance.json`, the decision-policy template, or any report. The
checked-in queue is intentionally unresolved: every item is `pending` and its
unresolved fields are `null` or empty.

## Field-by-field instructions

The top-level `review_queue_schema_version` is `1`. `reviews` is the fixed
four-item scope below. Do not add privacy, fairness, causal, or other review
claims to this queue; those are outside this workflow.

Each review item has:

- `id`: the stable lowercase identifier. The four identifiers are
  `retrieval-date`, `license-terms`, `permission`, and `operational-use`.
- `question`: the question the human reviewer must answer.
- `evidence_refs`: one or more repository-relative paths to tracked evidence.
  References are pointers, not decisions; do not cite an ignored raw CSV.
- `status`: exactly `pending`, `approved`, or `rejected`.
- `decision`: the concise answer supported by the evidence. Leave it `null`
  while pending. For `retrieval-date`, an approved decision must be an exact
  ISO date (`YYYY-MM-DD`). Do not infer a date from a file timestamp or hash.
- `reviewer`: the accountable human reviewer, not an automated tool. Leave it
  `null` while pending.
- `reviewed_at_utc`: the review time as a timezone-aware UTC ISO timestamp (for
  example, `2025-01-02T03:04:05Z`). Leave it `null` while pending.
- `rationale`: a concise explanation tied to `evidence_refs`. Leave it `null`
  while pending.
- `notes` (optional): a list of non-empty strings for bounded context. For an
  approved `operational-use` item, this must record the approval details:
  action, eligible population, accountable owner, policy/version reference,
  and any expiry or next-review condition.
- `blocks` (optional): a list of unresolved blockers. An approved item cannot
  retain blockers.

An approved or rejected item must have a decision, reviewer, UTC review time,
rationale, and evidence references. A pending item must leave decision,
reviewer, review time, and rationale null or empty. A rejection is not an
authorization, and an approval must not claim the opposite in its decision.
The validator also rejects duplicate IDs, unknown fields/statuses, malformed
dates/timestamps, and evidence references that are not tracked files.

## Evidence packets and recommended authority

These packets identify available tracked evidence and unresolved fields; they do
not make a decision or assign a reviewer. The recommended authorities below are
roles for human confirmation, not assigned reviewers. The ignored raw CSV is not
an evidence path.

### `retrieval-date`

- **Available tracked evidence:** [`data/provenance.json`](../data/provenance.json)
and [`scripts/download_data.py`](../scripts/download_data.py). The provenance
  record pins the source and records snapshot measurements, while the script is
the download entry point; neither supplies a verified retrieval receipt.
- **Missing evidence fields:** `local_snapshot.retrieved_at_utc` is `null` and
  `local_snapshot.retrieval_date_status` is `pending` in
  `data/provenance.json`.
- **Recommended authority:** a data custodian or publisher record authority
  who can verify the retrieval date. This is a recommendation, not an
  assignment.

### `license-terms`

- **Available tracked evidence:** [`data/provenance.json`](../data/provenance.json)
and [`README.md`](../README.md). The README distinguishes the code license
  from the source dataset's publisher terms.
- **Missing evidence fields:** `source.license_or_terms` and
  `source.license_or_terms_url` are `null`, and
  `source.license_review_status` is `pending`.
- **Recommended authority:** a licensing authority or data custodian who can
  verify the publisher's terms for the intended handling and redistribution.
  This is a recommendation, not an assignment.

### `permission`

- **Available tracked evidence:** [`data/provenance.json`](../data/provenance.json),
  [`README.md`](../README.md), and [`docs/methodology.md`](methodology.md).
- **Missing evidence fields:** `source.permission_review` is `null` and
  `source.permission_review_status` is `pending`.
- **Recommended authority:** the institutional permission authority for the
  intended research handling, redistribution, or use. This is a
  recommendation, not an assignment.

### `operational-use`

- **Available tracked evidence:** [`README.md`](../README.md),
  [`docs/decision_policy.md`](decision_policy.md),
  [`policies/decision_policy.v1.json`](../policies/decision_policy.v1.json),
  and [`reports/model_card.md`](../reports/model_card.md).
- **Missing evidence fields:** the policy template still has no values for
  `intended_action`, `eligible_population`,
  `capacity_constraints.max_actions`, `capacity_constraints.period`,
  `capacity_constraints.selection_rule`, `cost_unit`,
  `false_positive_cost`, `false_negative_cost`, `owner`, `review_date`,
  `approved_by`, or `approved_at`; `excluded_uses` is empty. Its `status` is
  `draft`, so no approved operational-use evidence is present.
- **Recommended authority:** the accountable operational owner, with privacy,
  legal, and fairness governance review as applicable. This is a recommendation
  of roles, not an assignment of reviewers.

## Research gate versus operational gate

The research result is conditional offline evidence about the available local
snapshot. A matching local snapshot can be analyzed as research evidence, but
that is not a human approval to redistribute data or act on people. The
retrieval-date, license/terms, and permission items document facts and
permissions; unresolved or rejected terms/permission still block
redistribution and operational use.

Operational use is a separate, higher gate. The `operational-use` item cannot
be approved unless retrieval date, license/terms, and permission are all
approved. Its approval also needs the concrete details listed above and does
not replace the separately required approved decision policy in
[`policies/decision_policy.v1.json`](../policies/decision_policy.v1.json).
Approval of the policy alone is likewise not approval of operational use.

Run the structural check with:

```bash
python scripts/validate_reviews.py
```

The command prints every pending item. CI accepts a valid queue with pending
items, so unresolved human work remains visible without being silently
converted into a pass. A deployment or other explicitly operational check can
run:

```bash
python scripts/validate_reviews.py --require-operational-use
```

That command fails until the operational-use item is approved and all of its
prerequisites pass the gate consistency checks.
