#!/usr/bin/env python3
"""
GRPO in ~130 lines, no dependencies.

This is not a language model. It is the smallest thing that still behaves like one
under RLVR: a policy that picks one of N answers, and a verifier that says whether
the answer was right.

Everything that matters is on display:

  * a group of G rollouts is sampled from the same prompt
  * the advantage is computed *relative to that group*
  * the policy-gradient update is weighted by that advantage

Two things to watch, in this order:

  1. `--task learnable` — the policy climbs, then saturates. Once it saturates,
     every reward in the group is 1.0, std(r) = 0, the advantage is 0, and
     learning stops dead even though nothing is broken.

  2. `--task flat` — zero advantage from step one. A task where every answer is
     correct and a task that has already been solved are *indistinguishable* from
     the gradient's point of view. Same cost to run. Same zero.

Usage:
    python3 toy_grpo.py --task learnable
    python3 toy_grpo.py --task flat
    python3 toy_grpo.py --task hard
    python3 toy_grpo.py --task learnable --json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Task: what counts as correct
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    """A verifier, in the smallest possible form: action -> reward."""

    name: str
    n_actions: int
    correct: frozenset[int]
    description: str
    # Probability that the verifier returns the WRONG answer. 0.0 is a good
    # verifier; anything above 0 is a flaky one, and it poisons the gradient.
    verifier_noise: float = 0.0

    def reward(self, action: int, rng: random.Random) -> float:
        truthful = 1.0 if action in self.correct else 0.0
        if self.verifier_noise <= 0.0:
            return truthful
        if truthful:
            return 0.0 if rng.random() < self.verifier_noise else 1.0
        return 1.0 if rng.random() < self.verifier_noise else 0.0


def build_task(name: str) -> Task:
    if name == "learnable":
        # 2 of 10 answers are right -> pass rate ~0.20 at the start.
        return Task(
            name="learnable",
            n_actions=10,
            correct=frozenset({2, 5}),
            description="20% of answers correct — the walkable regime",
        )
    if name == "flat":
        # Every answer is right. A saturated environment from step zero.
        return Task(
            name="flat",
            n_actions=10,
            correct=frozenset(range(10)),
            description="100% correct from the start — no spread, no signal",
        )
    if name == "hard":
        # 1 in 1000. At G=8 the chance that a group contains a correct answer is
        # under 1%, so the vast majority of steps produce std(r) = 0.
        return Task(
            name="hard",
            n_actions=1000,
            correct=frozenset({42}),
            description="0.1% correct — groups almost never contain a success",
        )
    if name == "noisy":
        # The task is learnable, but the verifier is wrong 20% of the time.
        return Task(
            name="noisy",
            n_actions=10,
            correct=frozenset({2, 5}),
            description="20% correct, but the verifier flips 20% of the time",
            verifier_noise=0.20,
        )
    raise ValueError(f"unknown task {name!r}")


# --------------------------------------------------------------------------- #
# Policy: softmax over actions
# --------------------------------------------------------------------------- #


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 1e-12)


def sample(probs: list[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


# --------------------------------------------------------------------------- #
# The GRPO step
# --------------------------------------------------------------------------- #


@dataclass
class StepStats:
    step: int
    pass_rate: float          # what the verifier reported
    true_pass_rate: float     # what actually happened
    std_reward: float
    mean_abs_advantage: float
    entropy: float


def grpo_step(
    logits: list[float],
    task: Task,
    *,
    group_size: int,
    lr: float,
    rng: random.Random,
    eps: float = 1e-8,
) -> StepStats:
    """One GRPO update. Returns what happened, so you can watch it."""

    # 1. Roll out a group: G independent samples from the SAME policy.
    probs = softmax(logits)
    actions = [sample(probs, rng) for _ in range(group_size)]

    # 2. Score them.
    rewards = [task.reward(a, rng) for a in actions]

    # 3. Group-relative advantage. This is the line that matters.
    mean_r = sum(rewards) / group_size
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / group_size)
    advantages = [(r - mean_r) / (std_r + eps) for r in rewards]

    # 4. Policy gradient: average over the group of  A_i * grad log pi(a_i).
    #    grad log softmax(z)[a] = onehot(a) - probs
    grad = [0.0] * len(logits)
    for a, adv in zip(actions, advantages):
        for j in range(len(logits)):
            grad[j] += adv * ((1.0 if j == a else 0.0) - probs[j])
    for j in range(len(logits)):
        logits[j] += lr * grad[j] / group_size

    true_pass = sum(1.0 if a in task.correct else 0.0 for a in actions) / group_size

    return StepStats(
        step=0,
        pass_rate=mean_r,
        true_pass_rate=true_pass,
        std_reward=std_r,
        mean_abs_advantage=sum(abs(a) for a in advantages) / group_size,
        entropy=entropy(probs),
    )


# --------------------------------------------------------------------------- #
# Running it
# --------------------------------------------------------------------------- #


def run(
    task_name: str, steps: int, group_size: int, lr: float, seed: int, noise: float | None = None
) -> dict:
    rng = random.Random(seed)
    task = build_task(task_name)
    if noise is not None:
        task.verifier_noise = noise
    logits = [0.0] * task.n_actions

    history: list[StepStats] = []
    for step in range(1, steps + 1):
        stats = grpo_step(logits, task, group_size=group_size, lr=lr, rng=rng)
        stats.step = step
        history.append(stats)

    window = max(1, steps // 40)  # first/last ~5 of 200 steps
    head = history[:window]
    tail = history[-window:]

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    return {
        "task": task_name,
        "description": task.description,
        "steps": steps,
        "group_size": group_size,
        "start_pass_rate": round(avg([s.pass_rate for s in head]), 4),
        "end_pass_rate": round(avg([s.pass_rate for s in tail]), 4),
        "start_entropy": round(avg([s.entropy for s in head]), 4),
        "end_entropy": round(avg([s.entropy for s in tail]), 4),
        "start_advantage": round(avg([s.mean_abs_advantage for s in head]), 6),
        "end_advantage": round(avg([s.mean_abs_advantage for s in tail]), 6),
        "peak_advantage": round(max(s.mean_abs_advantage for s in history), 6),
        "start_true_pass_rate": round(avg([s.true_pass_rate for s in head]), 4),
        "end_true_pass_rate": round(avg([s.true_pass_rate for s in tail]), 4),
        "signal_fraction": round(
            sum(1 for s in history if s.mean_abs_advantage > 1e-6) / len(history), 4
        ),
        "history": [
            {
                "step": s.step,
                "pass_rate": round(s.pass_rate, 4),
                "true_pass_rate": round(s.true_pass_rate, 4),
                "std_reward": round(s.std_reward, 4),
                "mean_abs_advantage": round(s.mean_abs_advantage, 6),
                "entropy": round(s.entropy, 4),
            }
            for s in history
        ],
    }


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 58, hi: float | None = None) -> str:
    """Downsample to `width` buckets (mean per bucket) and render as blocks."""
    if not values:
        return ""
    hi = max(values) if hi is None else hi
    if hi <= 0:
        return SPARK[0] * width
    n = len(values)
    out = []
    for i in range(width):
        lo_i = i * n // width
        hi_i = max(lo_i + 1, (i + 1) * n // width)
        bucket = values[lo_i:hi_i]
        v = sum(bucket) / len(bucket)
        frac = max(0.0, min(1.0, v / hi))
        out.append(SPARK[int(frac * (len(SPARK) - 1))])
    return "".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--task", default="learnable", choices=["learnable", "flat", "hard", "noisy"]
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--group", type=int, default=8, help="rollouts per prompt (GRPO G)")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--noise",
        type=float,
        default=None,
        help="override the verifier error rate (0.0 = honest, 0.5 = coin flip)",
    )
    p.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = p.parse_args()

    result = run(args.task, args.steps, args.group, args.lr, args.seed, args.noise)

    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
        return

    hist = result["history"]
    n = len(hist)
    n_actions = build_task(args.task).n_actions

    print()
    print(f"  task        {result['task']}  ({result['description']})")
    print(f"  group size  {result['group_size']}    steps {n}    lr {args.lr}")
    print()
    print("   step    pass   std(r)    |A|       H")
    for s in [hist[0], hist[n // 4], hist[n // 2], hist[3 * n // 4], hist[-1]]:
        print(
            f"   {s['step']:>4}    {s['pass_rate']:.2f}    {s['std_reward']:.3f}   "
            f"{s['mean_abs_advantage']:.3f}    {s['entropy']:.2f}"
        )
    print()
    print(f"  pass rate   {sparkline([s['pass_rate'] for s in hist], hi=1.0)}")
    print(f"  |advantage| {sparkline([s['mean_abs_advantage'] for s in hist])}")
    print(f"  entropy     {sparkline([s['entropy'] for s in hist], hi=math.log(n_actions))}")
    print()
    print(
        f"  verifier pass  {result['start_pass_rate']:.2f} -> {result['end_pass_rate']:.2f}"
        f"      true pass  {result['start_true_pass_rate']:.2f} -> {result['end_true_pass_rate']:.2f}"
    )
    print(
        f"  |A|            {result['start_advantage']:.3f} -> {result['end_advantage']:.3f}"
        f"      steps with any signal  {result['signal_fraction'] * 100:.0f}%"
    )
    print()

    learned = result["end_true_pass_rate"] - result["start_true_pass_rate"] > 0.10
    signal_frac = result["signal_fraction"]

    if args.task == "noisy":
        print("  NOISY VERIFIER — the model learned the task, and the advantage never")
        print("  went away. Look at the two pass columns: true accuracy is high, the")
        print("  verifier reports less, and the gap is pure noise that the policy keeps")
        print("  being pushed to chase. Compare |A| with --task learnable, where it")
        print("  decays as the task is solved. Here it is permanent, and it is not signal.")
    elif signal_frac < 0.05:
        print(f"  NO SIGNAL — only {signal_frac * 100:.0f}% of steps produced any spread at all.")
        print("  On the other steps std(r) = 0, so the advantage is exactly 0 and the")
        print("  gradient is exactly 0. The environment ran at full cost and taught nothing.")
    elif learned and result["end_advantage"] < 0.05:
        print("  LEARNED, THEN SATURATED — it climbed, and now the advantage is gone")
        print("  again. A task that has been solved and a task that was never solvable")
        print("  look identical from here: both have zero spread.")
    elif learned:
        print("  LEARNING — the policy moved toward the verifier and still has signal.")
    else:
        print("  STUCK — signal exists but the update did not move the policy.")
    print()

    if args.task == "flat":
        print("  Every reward is the same, so std(r) = 0, so every advantage is 0,")
        print("  so the gradient is exactly zero. This environment cost the same to run")
        print("  as a useful one and produced strictly nothing.")
        print()
    if args.task == "hard":
        print("  At 0.1% correctness, a group of 8 is almost never anything but all-wrong,")
        print("  so std(r) = 0 again. Too hard and too easy fail identically: the")
        print("  environment runs, costs full price, and teaches nothing.")
        print()


if __name__ == "__main__":
    main()
