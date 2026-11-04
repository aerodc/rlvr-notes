"""Three task specs and matching trajectories, for the verifier demo and tests.

The world is a minimal issue tracker. The point is not the domain — it is that
success conditions are written down as data, so they can be versioned, hashed,
diffed and argued about, rather than living in someone's head.
"""

from __future__ import annotations

from verifier import Requirement, TaskSpec, Trajectory


# A world in which the agent did the task correctly.
GOOD_STATE = {
    "issues": [
        {"id": 1, "title": "Fix login redirect", "priority": "high", "assignee": "alice", "status": "open"},
        {"id": 2, "title": "Tune search latency", "priority": "medium", "assignee": "bob", "status": "open"},
        {"id": 3, "title": "Deprecate legacy API", "priority": "low", "assignee": None, "status": "open"},
    ],
    "comments": [{"issue_id": 1, "body": "Reproduced on staging."}],
}

# A world in which the agent wrote the right words but changed nothing.
TALKED_ONLY_STATE = {
    "issues": [
        {"id": 1, "title": "Fix login redirect", "priority": "medium", "assignee": None, "status": "open"},
        {"id": 2, "title": "Tune search latency", "priority": "medium", "assignee": "bob", "status": "open"},
        {"id": 3, "title": "Deprecate legacy API", "priority": "low", "assignee": None, "status": "open"},
    ],
    "comments": [],
}


TASK_001 = TaskSpec(
    task_id="triage-001",
    instruction=(
        "Set issue #1 to priority high and assign it to alice. "
        "Do not change any other issue."
    ),
    requirements=(
        Requirement(
            name="issue_1_priority_is_high",
            path="issues.0.priority",
            expected="high",
            gating=True,
        ),
        Requirement(
            name="issue_1_assigned_to_alice",
            path="issues.0.assignee",
            expected="alice",
            gating=True,
        ),
        # The easy half: models love changing everything else too.
        Requirement(
            name="issue_2_untouched",
            path="issues.1.priority",
            expected="medium",
            category="constraint",
        ),
        Requirement(
            name="issue_3_untouched",
            path="issues.2.priority",
            expected="low",
            category="constraint",
        ),
    ),
)

TASK_002 = TaskSpec(
    task_id="comment-002",
    instruction="Add a comment to issue #1 explaining what you found.",
    requirements=(
        Requirement(
            name="a_comment_exists",
            path="comments",
            expected=1,
            kind="length",
            gating=True,
        ),
        Requirement(
            name="comment_is_on_issue_1",
            path="comments.0.issue_id",
            expected=1,
            gating=True,
        ),
        Requirement(
            name="comment_is_not_empty",
            path="comments.0.body",
            expected=5,
            kind="predicate",
            predicate=lambda v: isinstance(v, str) and len(v.strip()) >= 5,
        ),
    ),
)

TASK_003 = TaskSpec(
    task_id="audit-003",
    instruction="Mark every high-priority issue as needing review.",
    requirements=(
        Requirement(
            name="all_high_priority_flagged",
            path="issues",
            expected=True,
            kind="predicate",
            gating=True,
            predicate=lambda issues: all(
                i.get("needs_review") is True
                for i in (issues or [])
                if i.get("priority") == "high"
            ),
        ),
        Requirement(
            name="low_priority_not_flagged",
            path="issues",
            expected=True,
            kind="predicate",
            category="constraint",
            predicate=lambda issues: all(
                "needs_review" not in i
                for i in (issues or [])
                if i.get("priority") != "high"
            ),
        ),
    ),
)

DEMO_TASKS = (TASK_001, TASK_002, TASK_003)


def demo_trajectories() -> list[Trajectory]:
    good = Trajectory(
        actions=({"tool": "update_issue", "id": 1, "priority": "high", "assignee": "alice"},),
        state=GOOD_STATE,
        output="Done. Issue #1 is now high priority and assigned to alice.",
    )
    talked = Trajectory(
        actions=(),
        state=TALKED_ONLY_STATE,
        output="I have set issue #1 to high priority and assigned it to alice.",
    )
    reviewed_state = {
        "issues": [
            {"id": 1, "title": "Fix login redirect", "priority": "high", "needs_review": True},
            {"id": 2, "title": "Tune search latency", "priority": "medium"},
        ],
        "comments": [],
    }
    return [good, talked, Trajectory(state=reviewed_state)]
