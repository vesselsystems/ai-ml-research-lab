# Contributing

This repository favors reproducible experiments and explicit limitations.

Before opening a change:

1. Create an isolated Python 3.11+ environment and install `.[dev]`.
2. Run `pytest`, `ruff check .`, and `python scripts/validate_reviews.py`.
3. Keep preprocessing inside the evaluated pipeline and record changes to the protocol.
4. Do not commit downloaded source data, fitted models, credentials, or personal information.

Human review changes belong only in `reviews/review_queue.v1.json`; follow [`docs/review_workflow.md`](docs/review_workflow.md) and do not fill unresolved values from inference.

A new metric or model should include a reason for adding it, a test or reproducible artifact, and an explanation of how it changes the interpretation.
