import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
QUEUE_PATH = REPO_ROOT / "reviews" / "review_queue.v1.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_reviews.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("validate_reviews_script", SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
validate_reviews_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules[_SCRIPT_SPEC.name] = validate_reviews_script
_SCRIPT_SPEC.loader.exec_module(validate_reviews_script)


def _queue() -> dict[str, object]:
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def _approved_policy() -> dict[str, object]:
    return {
        "policy_schema_version": 1,
        "policy_version": "1.0",
        "intended_action": "Offer a reviewed retention contact",
        "eligible_population": "Rows in the synthetic evaluation population",
        "capacity_constraints": {
            "max_actions": 10,
            "period": "per review period",
            "selection_rule": "highest_score_then_input_order",
        },
        "cost_unit": "reviewed cost units per action",
        "false_positive_cost": 10.0,
        "false_negative_cost": 25.0,
        "owner": "synthetic-policy-owner",
        "review_date": "2099-12-31",
        "excluded_uses": ["Do not deny service or change pricing"],
        "status": "approved",
        "approved_by": "synthetic-reviewer",
        "approved_at": "2025-01-01T00:00:00Z",
    }


def _run(queue: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--queue", str(queue), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_template_is_valid_and_prints_every_pending_review() -> None:
    result = _run(QUEUE_PATH)

    assert result.returncode == 0
    assert "PENDING reviews:" in result.stdout
    for review_id in ("retrieval-date", "license-terms", "permission", "operational-use"):
        assert review_id in result.stdout


def test_require_operational_use_fails_for_pending_template() -> None:
    result = _run(QUEUE_PATH, "--require-operational-use")

    assert result.returncode == 1
    assert "requires an approved operational-use review" in result.stderr


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda queue: queue["reviews"].append(queue["reviews"][0]), "duplicate review id"),
        (lambda queue: queue["reviews"][0].__setitem__("status", "done"), "status"),
    ],
)
def test_validator_rejects_malformed_queue(tmp_path: Path, change, message: str) -> None:
    queue = _queue()
    change(queue)
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue), encoding="utf-8")

    result = _run(path, "--project-root", str(REPO_ROOT))

    assert result.returncode == 1
    assert message in result.stderr


def test_validator_rejects_invalid_utc_timestamp(tmp_path: Path) -> None:
    queue = _queue()
    review = queue["reviews"][0]
    review.update(
        {
            "status": "approved",
            "decision": "2025-01-01",
            "reviewer": "human-reviewer",
            "reviewed_at_utc": "2025-01-01T00:00:00",
            "rationale": "The date is recorded in the evidence.",
        }
    )
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue), encoding="utf-8")

    result = _run(path, "--project-root", str(REPO_ROOT))

    assert result.returncode == 1
    assert "UTC ISO timestamp" in result.stderr


def test_validator_rejects_missing_tracked_evidence(tmp_path: Path) -> None:
    queue = _queue()
    queue["reviews"][0]["evidence_refs"] = ["does-not-exist.md"]
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue), encoding="utf-8")

    result = _run(path, "--project-root", str(REPO_ROOT))

    assert result.returncode == 1
    assert "missing or untracked" in result.stderr


def test_operational_gate_requires_all_prerequisites(tmp_path: Path) -> None:
    queue = _queue()
    operational = queue["reviews"][3]
    operational.update(
        {
            "status": "approved",
            "decision": "approved for the documented use",
            "reviewer": "human-reviewer",
            "reviewed_at_utc": "2025-01-01T00:00:00Z",
            "rationale": "The separate approval is recorded.",
            "notes": ["Owner, action, population, policy, and review date recorded."],
        }
    )
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue), encoding="utf-8")

    result = _run(path, "--project-root", str(REPO_ROOT))

    assert result.returncode == 1
    assert "prerequisite reviews are not approved" in result.stderr


def test_approved_queue_requires_an_approved_policy(tmp_path: Path) -> None:
    queue = _queue()
    for review in queue["reviews"]:
        review.update(
            {
                "status": "approved",
                "reviewer": "human-reviewer",
                "reviewed_at_utc": "2025-01-01T00:00:00Z",
                "rationale": "Synthetic test fixture only.",
            }
        )
    queue["reviews"][0]["decision"] = "2025-01-01"
    queue["reviews"][1]["decision"] = "approved for the reviewed terms"
    queue["reviews"][2]["decision"] = "approved for the reviewed permission"
    queue["reviews"][3]["decision"] = "approved for the documented use"
    queue["reviews"][3]["notes"] = ["Synthetic operational details for this test fixture."]
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(copy.deepcopy(queue)), encoding="utf-8")

    draft_result = _run(
        queue_path,
        "--project-root",
        str(REPO_ROOT),
        "--require-operational-use",
    )

    assert draft_result.returncode == 1
    assert "complete approved decision policy" in draft_result.stderr

    policy_path = tmp_path / "approved-policy.json"
    policy_path.write_text(json.dumps(_approved_policy()), encoding="utf-8")
    result = _run(
        queue_path,
        "--project-root",
        str(REPO_ROOT),
        "--policy",
        str(policy_path),
        "--require-operational-use",
    )

    assert result.returncode == 0
    assert "Operational-use gate: approved." in result.stdout


def test_tracked_files_fails_closed_when_git_fails(tmp_path: Path, monkeypatch) -> None:
    def fail_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git", "ls-files"])

    monkeypatch.setattr(validate_reviews_script.subprocess, "run", fail_git)

    with pytest.raises(validate_reviews_script.ReviewQueueError, match="tracked files"):
        validate_reviews_script._tracked_files(tmp_path)
