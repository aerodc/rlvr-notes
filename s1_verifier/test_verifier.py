"""Tests for a verifier. If these pass, the verifier is shippable.

The important ones are `test_deterministic` and `test_talked_only_fails`:
the first is the property everyone assumes, the second is the failure mode
that makes an environment worthless without anyone noticing.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pytest

from tasks import GOOD_STATE, TALKED_ONLY_STATE, TASK_001, TASK_002, TASK_003, demo_trajectories
from verifier import TaskSpec, Trajectory, explain, replay, verify


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec,traj", list(zip((TASK_001, TASK_002, TASK_003), demo_trajectories())))
def test_deterministic(spec: TaskSpec, traj: Trajectory):
    """Same trajectory, 100 replays, exactly one distinct score."""
    distinct, rate = replay(spec, traj, n=100)
    assert distinct == 1, f"{spec.task_id}: {distinct} distinct scores over 100 replays"
    assert rate == 0.0


def test_score_payload_is_stable():
    """Two calls produce identical payloads, including the spec hash."""
    a = verify(TASK_001, demo_trajectories()[0]).as_dict()
    b = verify(TASK_001, demo_trajectories()[0]).as_dict()
    assert a == b


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


def test_evidence_covers_every_requirement():
    traj = demo_trajectories()[0]
    score = verify(TASK_001, traj)
    assert len(score.evidence) == len(TASK_001.requirements)
    for e in score.evidence:
        assert e.detail, "every check must explain itself"
        assert e.expected is not None or e.passed, "a failure must carry an expectation"


def test_failure_is_explainable_without_the_trajectory():
    """The report alone should say what went wrong and what was expected."""
    traj = demo_trajectories()[1]  # talked, did not act
    report = explain(verify(TASK_001, traj))
    assert "FAIL" in report
    assert "high" in report
    assert "alice" in report


# --------------------------------------------------------------------------- #
# Reward shape
# --------------------------------------------------------------------------- #


def test_reward_in_unit_interval():
    for spec, traj in zip((TASK_001, TASK_002, TASK_003), demo_trajectories()):
        s = verify(spec, traj)
        assert 0.0 <= s.reward <= 1.0
        assert all(0.0 <= v <= 1.0 for v in s.subscores.values())


def test_gating_failure_zeroes_the_reward():
    """Partial credit on a task whose core requirement failed is worse than useless."""
    bad = {"issues": [
        {"id": 1, "priority": "medium", "assignee": "alice"},
        {"id": 2, "priority": "medium"},
        {"id": 3, "priority": "low"},
    ], "comments": []}
    score = verify(TASK_001, Trajectory(state=bad))
    assert score.reward == 0.0
    # ...and the constraints still passed, which is exactly why gating is needed.
    assert score.subscores.get("constraint") == 1.0


def test_talked_only_fails():
    """The model claimed success and changed nothing. The verifier must catch it."""
    score = verify(TASK_001, Trajectory(state=TALKED_ONLY_STATE, output="All done!"))
    assert score.reward == 0.0
    failed = [e.requirement for e in score.evidence if not e.passed]
    assert "issue_1_priority_is_high" in failed


def test_correct_trajectory_scores_one():
    assert verify(TASK_001, demo_trajectories()[0]).reward == 1.0


# --------------------------------------------------------------------------- #
# Spec identity
# --------------------------------------------------------------------------- #


def test_spec_hash_is_stable_and_sensitive():
    assert TASK_001.spec_hash() == TASK_001.spec_hash()

    from verifier import Requirement

    altered = TaskSpec(
        task_id=TASK_001.task_id,
        instruction=TASK_001.instruction,
        requirements=TASK_001.requirements[:-1]
        + (Requirement(name="issue_3_untouched", path="issues.2.priority", expected="high"),),
    )
    assert altered.spec_hash() != TASK_001.spec_hash(), (
        "changing a requirement must change the hash; otherwise scores are not attributable"
    )


def test_missing_path_is_a_failure_not_a_crash():
    """Malformed state must never raise — a crashed verifier is a stalled training run."""
    assert verify(TASK_001, Trajectory(state={})).reward == 0.0
    assert verify(TASK_002, Trajectory(state={"comments": "not-a-list"})).reward == 0.0


# --------------------------------------------------------------------------- #
# The two states that decide whether the environment is worth anything
# --------------------------------------------------------------------------- #


def test_good_and_bad_states_are_distinguishable():
    """Sanity: the verifier separates the two worlds it is supposed to separate."""
    good = verify(TASK_001, Trajectory(state=GOOD_STATE))
    bad = verify(TASK_001, Trajectory(state=TALKED_ONLY_STATE))
    assert good.reward == 1.0
    assert bad.reward == 0.0
