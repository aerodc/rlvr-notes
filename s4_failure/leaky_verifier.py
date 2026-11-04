#!/usr/bin/env python3
"""
Three ways a code environment gets reward-hacked, and the verifier that stops them.

This runs a tiny matrix: three agents against three verifiers. One agent does the
task; two claim to. The question is which verifier can tell them apart.

    python3 leaky_verifier.py
    python3 leaky_verifier.py --json

The uncomfortable result is the first row: a verifier that reads the agent's
*words* gives full marks to an agent that changed nothing. Nothing crashes. The
reward curve looks excellent. The model learns to narrate.
"""

from __future__ import annotations

import argparse
import copy
import json

TASK_INSTRUCTION = "Set issue #1 to priority high and assign it to alice."


def make_world() -> dict:
    return {
        "issues": [
            {"id": 1, "title": "Fix login redirect", "priority": "medium", "assignee": None},
            {"id": 2, "title": "Tune search latency", "priority": "medium", "assignee": "bob"},
        ],
        "comments": [],
    }


# --------------------------------------------------------------------------- #
# Agents. Each returns (world, output). Only one of them does the task.
# --------------------------------------------------------------------------- #


def honest_agent(world: dict) -> tuple[dict, str]:
    world["issues"][0]["priority"] = "high"
    world["issues"][0]["assignee"] = "alice"
    return world, "Done. Issue #1 is now high priority and assigned to alice."


def echo_agent(world: dict) -> tuple[dict, str]:
    """Answer leakage + narration. Reads the instruction, restates it, acts on nothing."""
    return world, f"Done. {TASK_INSTRUCTION.replace('Set ', '').lower()}"


def loose_agent(world: dict) -> tuple[dict, str]:
    """Does half the task and claims all of it. The most common real failure."""
    world["issues"][0]["priority"] = "high"  # sets the field a loose check looks at
    return world, "Done. Issue #1 is high priority and assigned to alice."


AGENTS = {
    "honest": honest_agent,
    "echo": echo_agent,
    "loose": loose_agent,
}


# --------------------------------------------------------------------------- #
# Verifiers. All three look plausible. Only one is a verifier.
# --------------------------------------------------------------------------- #


def v_string(world: dict, output: str) -> float:
    """Checks the agent's reply. The classic mistake, and the easiest to write."""
    text = output.lower()
    return 1.0 if ("high" in text and "alice" in text) else 0.0


def v_loose(world: dict, output: str) -> float:
    """Checks one field of the world. Better than v_string, still half a verifier."""
    return 1.0 if world["issues"][0]["priority"] == "high" else 0.0


def v_state(world: dict, output: str) -> float:
    """Checks the world. This is the only one that is actually a verifier."""
    i1 = world["issues"][0]
    return 1.0 if (i1["priority"] == "high" and i1["assignee"] == "alice") else 0.0


VERIFIERS = {
    "string": v_string,
    "loose": v_loose,
    "state": v_state,
}


def oracle(world: dict) -> float:
    """Ground truth: did the world actually change as instructed?"""
    i1 = world["issues"][0]
    return 1.0 if (i1["priority"] == "high" and i1["assignee"] == "alice") else 0.0


# --------------------------------------------------------------------------- #
# The matrix
# --------------------------------------------------------------------------- #


def run_matrix() -> dict:
    rows = []
    for agent_name, agent in AGENTS.items():
        world, output = agent(copy.deepcopy(make_world()))
        truth = oracle(world)
        row = {"agent": agent_name, "true_success": truth, "scores": {}}
        for v_name, verifier in VERIFIERS.items():
            row["scores"][v_name] = verifier(copy.deepcopy(world), output)
        rows.append(row)

    summary = {}
    for v_name in VERIFIERS:
        honest = next(r for r in rows if r["agent"] == "honest")["scores"][v_name]
        hackers = [r for r in rows if r["agent"] != "honest"]
        hacked = any(r["scores"][v_name] >= honest for r in hackers)
        summary[v_name] = {
            "honest_reward": honest,
            "hacked": hacked,
            "max_hacker_reward": max(r["scores"][v_name] for r in hackers),
        }
    return {"rows": rows, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    result = run_matrix()

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print()
    print("  agent     true success |  v_string  v_loose  v_state")
    print("  " + "-" * 52)
    for row in result["rows"]:
        s = row["scores"]
        print(
            f"  {row['agent']:<9}      {row['true_success']:.1f}       |"
            f"    {s['string']:.1f}      {s['loose']:.1f}      {s['state']:.1f}"
        )
    print()

    print("  What each verifier would teach the model")
    for name, s in result["summary"].items():
        if s["hacked"]:
            print(
                f"    {name:<6}  HACKED — an agent that did not solve it scores "
                f"{s['max_hacker_reward']:.1f}, same as the honest one."
            )
        else:
            print(f"    {name:<6}  holds — the hackers score below the honest agent.")
    print()

    print("  The three hacks, named")
    print("    1. answer leakage   the expected value is present in the instruction,")
    print("                        the files, or the test names, so the agent can echo it")
    print("    2. checker shortcut the verifier reads the reply instead of the world")
    print("    3. loose assertion  the verifier checks one field when the task has two")
    print()
    print("  The fix is not a stricter string match. It is checking state, and")
    print("  requiring evidence from the world rather than from the transcript:")
    print()
    print("    - assert on world state, never on the model's output")
    print("    - assert every field the instruction names, and that untouched things")
    print("      are untouched (the negative assertion is the one people skip)")
    print("    - run a red-team agent nightly whose only job is to score 1.0 without")
    print("      doing the task, and fail the build when it succeeds")
    print()


if __name__ == "__main__":
    main()
