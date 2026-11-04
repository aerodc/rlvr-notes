#!/usr/bin/env python3
"""
Difficulty calibration, without training anything.

Claim to test: an environment is only useful if its pass rate lands in a band,
roughly 20-80%. Where does that band come from?

It comes from the group. With binary rewards and a group of G rollouts, a group
carries information only if it is *not* all-identical. The probability of that is

    signal(p) = 1 - p^G - (1-p)^G

    p = pass rate, G = rollouts per prompt

Nothing here needs a model. You can measure an environment's difficulty and
decide whether it is worth training on before spending a single GPU-hour.

    python3 sweep.py
    python3 sweep.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import random


def signal_exact(p: float, group: int) -> float:
    """P(a group of `group` rollouts is not degenerate). Exact, closed form."""
    return 1.0 - p**group - (1.0 - p) ** group


def signal_monte_carlo(p: float, group: int, trials: int, rng: random.Random) -> float:
    """Same thing by simulation, so you can see the sampling error at your n."""
    informative = 0
    for _ in range(trials):
        successes = sum(1 for _ in range(group) if rng.random() < p)
        if 0 < successes < group:
            informative += 1
    return informative / trials


def rollouts_needed_for_p(p: float, half_width: float = 0.05) -> int:
    """Rollouts needed to estimate a pass rate p to +/- half_width (95% CI)."""
    if p <= 0.0 or p >= 1.0:
        return 0
    return math.ceil(p * (1 - p) / (half_width / 1.96) ** 2)


SPARK = "▁▂▃▄▅▆▇█"


def bar(value: float, width: int = 40) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "█" * filled + "·" * (width - filled)


def sweep(group_sizes: list[int], steps: int = 30) -> dict:
    ps = [round(i / (steps * 2), 4) for i in range(1, steps * 2)]
    rng = random.Random(0)

    out: dict = {"group_sizes": group_sizes, "curve": {}, "bands": {}, "rollouts_needed": {}}
    for g in group_sizes:
        out["curve"][g] = [{"p": p, "signal": round(signal_exact(p, g), 4)} for p in ps]

    peak = signal_exact(0.5, max(group_sizes))

    for g in group_sizes:
        half = [p for p in ps if signal_exact(p, g) >= 0.5]
        tenth = [p for p in ps if signal_exact(p, g) >= 0.1 * peak]
        out["bands"][g] = {
            "signal_at_p50": round(signal_exact(0.5, g), 4),
            "p where signal>=0.5": [min(half), max(half)] if half else None,
            "p where signal>=10% of peak": [min(tenth), max(tenth)] if tenth else None,
        }

    for p in (0.02, 0.1, 0.2, 0.5, 0.8, 0.9, 0.98):
        out["rollouts_needed"][p] = rollouts_needed_for_p(p)

    # Validate the closed form against simulation at a couple of points.
    out["monte_carlo_check"] = [
        {
            "p": p,
            "group": 8,
            "exact": round(signal_exact(p, 8), 4),
            "simulated": round(signal_monte_carlo(p, 8, 20_000, rng), 4),
        }
        for p in (0.05, 0.2, 0.5, 0.9)
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--groups", default="4,8,16", help="group sizes to compare")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    groups = [int(x) for x in args.groups.split(",")]
    data = sweep(groups)

    if args.json:
        print(json.dumps(data, indent=2))
        return

    print()
    print("  P(a group of G rollouts carries any information), by pass rate")
    print()
    header = "    p   " + "".join(f"   G={g:<3}" for g in groups)
    print(header)
    for row_p in (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99):
        cells = "".join(f"   {signal_exact(row_p, g):.3f} " for g in groups)
        print(f"  {row_p:>4.2f} {cells}")

    print()
    for g in groups:
        print(f"  G={g}  signal(p) as pass rate goes 0 -> 1")
        for row_p in (0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99):
            print(f"    p={row_p:>4.2f}  {bar(signal_exact(row_p, g))}  {signal_exact(row_p, g):.3f}")
        print()

    print("  The bands")
    for g in groups:
        b = data["bands"][g]
        half = b["p where signal>=0.5"]
        tenth = b["p where signal>=10% of peak"]
        print(f"    G={g:<3} signal at p=0.5: {b['signal_at_p50']:.3f}")
        print(f"          useful band (signal >= 0.5):          {half[0]:.2f} - {half[1]:.2f}")
        print(f"          non-trivial band (signal >= 10% peak): {tenth[0]:.2f} - {tenth[1]:.2f}")

    print()
    print("  Measuring difficulty: rollouts needed for +/-5% on the pass rate")
    for p, n in data["rollouts_needed"].items():
        print(f"    p={p:>4.2f} -> {n:>3} rollouts")

    print()
    print("  Closed form vs simulation (G=8, 20k trials)")
    for row in data["monte_carlo_check"]:
        print(f"    p={row['p']:<5} exact {row['exact']:.3f}   simulated {row['simulated']:.3f}")

    print()
    print("  Read this as: an environment at p=0.98 is not 'almost solved', it is")
    print("  silent. At G=8 fewer than 2% of groups carry any information at all, and")
    print("  the ones that do are teaching the model to fix rare slips, not the task.")
    print("  The mirror image at p=0.02 is indistinguishable in cost and equally silent.")
    print()


if __name__ == "__main__":
    main()
