# S3 — Where the 20–80% band comes from

```bash
make sweep
```

## The derivation

Binary reward, group of `G` rollouts, pass rate `p`. A group carries information
only if it is not degenerate — that is, not all-correct and not all-wrong:

```
signal(p) = 1 - p^G - (1-p)^G
```

No model, no training run, no GPU. `sweep.py` evaluates it exactly and checks it
against a simulation.

| p | G=4 | G=8 | G=16 |
|---|---|---|---|
| 0.01 | 0.039 | 0.077 | 0.149 |
| 0.05 | 0.185 | 0.337 | 0.560 |
| 0.20 | 0.589 | 0.832 | 0.972 |
| 0.50 | 0.875 | 0.992 | 1.000 |
| 0.80 | 0.589 | 0.832 | 0.972 |
| 0.95 | 0.185 | 0.337 | 0.560 |
| 0.99 | 0.039 | 0.077 | 0.149 |

At `G=8`, a pass rate of 0.98 means only **1.5% of groups** carry any signal at
all. The environment is not "nearly solved" — it is silent, and it costs exactly
as much to run as a useful one.

## Three things this buys you

1. **A rejection criterion that costs nothing.** Measure pass rate over ~100
   rollouts before training. If it is outside the band, do not train.
2. **A justification for calibration as product, not polish.** Difficulty is not
   a quality tweak; it decides whether the compute produces anything.
3. **An argument about `G`.** Larger `G` widens the useful band — at `G=16`,
   `p=0.05` still gives 56% informative groups. It also multiplies the cost of
   every step. That trade is the whole reason reset latency matters.

## Measuring p

Binomial, so the standard error at `n` rollouts is `sqrt(p(1-p)/n)`. For ±5% at
95% confidence:

| p | rollouts |
|---|---|
| 0.02 | 31 |
| 0.10 | 139 |
| 0.20 | 246 |
| 0.50 | 385 |

Roughly **400 rollouts** buys you ±5% across the whole range. That is minutes of
environment time and zero GPU time — which is the point.

## The exercise

Take your S2 environment and a small model. Measure pass rate at three
temperature settings. You are looking for a task you can make land at `p≈0.5` by
changing the task, not the model. If you cannot, the task is not calibrated —
it is a coin that only lands one way.
