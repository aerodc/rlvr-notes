"""
A verifier you could hand to a lab.

The contract is deliberately boring and deliberately strict:

    verify(spec, trajectory) -> Score(reward, subscores, evidence, version, spec_hash)

Three properties do the work:

  * **Deterministic.** Same trajectory in, same score out. Always. Tested.
  * **Evidence on every check.** Not logging — the output. A verifier that
    returns 0 and nothing else costs whoever debugs it an hour per task.
  * **Versioned and hashed.** A score without a verifier version is not
    reproducible, and a score without a spec hash is not attributable to a
    task definition. Both go in the payload so they end up in the training log.

Gating checks are the fourth, quieter idea: if a core requirement fails, the
reward is 0 regardless of partial credit. Half-marks on a task whose whole
point was missed teach a model to do the easy half.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

VERIFIER_VERSION = "0.1.0"


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Requirement:
    """One thing that must be true in the world for the task to count as done."""

    name: str
    path: str  # dotted path into the world state, e.g. "issues.0.priority"
    expected: Any
    kind: str = "equals"  # equals | contains | length | count | predicate
    category: str = "core"
    weight: float = 1.0
    gating: bool = False  # fail this and the reward is 0, whatever else passed
    predicate: Callable[[Any], bool] | None = None


@dataclass(frozen=True)
class CheckResult:
    """The evidence for one requirement. This is what makes a 0 debuggable."""

    requirement: str
    passed: bool
    expected: Any
    actual: Any
    detail: str
    category: str
    weight: float
    gating: bool


@dataclass(frozen=True)
class Score:
    task_id: str
    reward: float
    subscores: dict[str, float]
    evidence: tuple[CheckResult, ...]
    verifier_version: str
    spec_hash: str

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "reward": self.reward,
            "subscores": self.subscores,
            "verifier_version": self.verifier_version,
            "spec_hash": self.spec_hash,
            "evidence": [
                {
                    "requirement": e.requirement,
                    "passed": e.passed,
                    "expected": e.expected,
                    "actual": e.actual,
                    "detail": e.detail,
                    "category": e.category,
                    "weight": e.weight,
                    "gating": e.gating,
                }
                for e in self.evidence
            ],
        }


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    requirements: tuple[Requirement, ...]

    def spec_hash(self) -> str:
        payload = {
            "task_id": self.task_id,
            "requirements": [
                {
                    "name": r.name,
                    "path": r.path,
                    "kind": r.kind,
                    "expected": r.expected,
                    "category": r.category,
                    "weight": r.weight,
                    "gating": r.gating,
                }
                for r in self.requirements
            ],
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True)
class Trajectory:
    """What the agent did, and what the world looked like afterwards."""

    actions: tuple[dict, ...] = ()
    state: dict = field(default_factory=dict)
    output: str = ""


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #


def get_path(state: Any, path: str) -> Any:
    """Resolve "issues.0.priority" against nested dicts and lists."""
    current = state
    if not path:
        return current
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _check(req: Requirement, actual: Any) -> tuple[bool, str]:
    if req.kind == "equals":
        return actual == req.expected, f"{req.path} == {req.expected!r}"
    if req.kind == "contains":
        ok = actual is not None and req.expected in actual
        return ok, f"{req.expected!r} in {actual!r}"
    if req.kind == "length":
        ok = actual is not None and hasattr(actual, "__len__") and len(actual) == req.expected
        got = len(actual) if actual is not None and hasattr(actual, "__len__") else None
        return ok, f"len({req.path}) == {req.expected} (got {got})"
    if req.kind == "count":
        ok = actual is not None and hasattr(actual, "__len__") and len(actual) == req.expected
        got = len(actual) if actual is not None and hasattr(actual, "__len__") else None
        return ok, f"count({req.path}) == {req.expected} (got {got})"
    if req.kind == "predicate":
        if req.predicate is None:
            return False, "predicate kind used without a predicate"
        try:
            return bool(req.predicate(actual)), f"predicate on {req.path}"
        except Exception as exc:  # a broken predicate is a failed check, never a crash
            return False, f"predicate raised {type(exc).__name__}: {exc}"
    return False, f"unknown kind {req.kind!r}"


def verify(spec: TaskSpec, trajectory: Trajectory) -> Score:
    """Score one trajectory. Pure function: no I/O, no clock, no randomness."""
    evidence: list[CheckResult] = []
    gating_failed = False

    for req in spec.requirements:
        actual = get_path(trajectory.state, req.path)
        passed, detail = _check(req, actual)
        if req.gating and not passed:
            gating_failed = True
        evidence.append(
            CheckResult(
                requirement=req.name,
                passed=passed,
                expected=req.expected,
                actual=actual,
                detail=detail,
                category=req.category,
                weight=req.weight,
                gating=req.gating,
            )
        )

    total_weight = sum(r.weight for r in spec.requirements) or 1.0
    earned = sum(e.weight for e in evidence if e.passed)

    # Subscores per category, so a failure is localisable before you read evidence.
    by_category: dict[str, list[CheckResult]] = {}
    for e in evidence:
        by_category.setdefault(e.category, []).append(e)
    subscores = {
        cat: round(sum(x.weight for x in xs if x.passed) / (sum(x.weight for x in xs) or 1.0), 4)
        for cat, xs in by_category.items()
    }

    reward = 0.0 if gating_failed else round(earned / total_weight, 4)

    return Score(
        task_id=spec.task_id,
        reward=reward,
        subscores=subscores,
        evidence=tuple(evidence),
        verifier_version=VERIFIER_VERSION,
        spec_hash=spec.spec_hash(),
    )


# --------------------------------------------------------------------------- #
# Determinism: the property people assume rather than test
# --------------------------------------------------------------------------- #


def replay(spec: TaskSpec, trajectory: Trajectory, n: int = 100) -> tuple[int, float]:
    """Score the same trajectory n times. Returns (distinct_scores, nondeterminism_rate)."""
    seen: set[tuple] = set()
    for _ in range(n):
        s = verify(spec, trajectory)
        seen.add((s.reward, tuple((e.requirement, e.passed) for e in s.evidence)))
    distinct = len(seen)
    return distinct, round((distinct - 1) / n, 6)


def explain(score: Score) -> str:
    """Human-readable failure report. If this is not enough to debug, evidence is thin."""
    lines = [
        f"task={score.task_id}  reward={score.reward:.3f}  "
        f"verifier={score.verifier_version}  spec={score.spec_hash}",
        f"subscores: {score.subscores}",
    ]
    for e in score.evidence:
        mark = "PASS" if e.passed else "FAIL"
        gate = " [gating]" if e.gating else ""
        lines.append(f"  {mark}{gate} {e.requirement}: {e.detail}")
        if not e.passed:
            lines.append(f"        actual={e.actual!r}  expected={e.expected!r}")
    return "\n".join(lines)


if __name__ == "__main__":
    from tasks import DEMO_TASKS, demo_trajectories

    for spec, traj in zip(DEMO_TASKS, demo_trajectories()):
        print(explain(verify(spec, traj)))
        print()
