# GRPO in 60 lines

The algorithm is smaller than the literature makes it look. Here is the whole thing, then the mapping to `s0_baseline/toy_grpo.py`.

## The objective

For each prompt `q`, sample a group of `G` completions `o_1 … o_G` from the current policy, score them, and compute

```
A_i = (r_i - mean(r)) / (std(r) + eps)          # advantage, within the group
```

Then take a policy-gradient step with a clipped ratio and a KL leash:

```
J = E[ 1/G Σ_i  min( ratio_i · A_i ,  clip(ratio_i, 1-ε, 1+ε) · A_i )  -  β·KL(π_θ ‖ π_ref) ]

ratio_i = π_θ(o_i | q) / π_old(o_i | q)
```

Three pieces, and only three:

| Piece | What it does | If you remove it |
|---|---|---|
| `A_i` | says *which* rollouts to reinforce | no learning signal |
| `clip` | limits how far one update can move the policy | training instability, occasional collapse |
| `β·KL` | keeps the model near its starting point | drifts into degenerate, high-reward gibberish |

The **group** is the trick. PPO needs a learned value network to estimate a baseline. GRPO uses the mean reward of the other rollouts from the same prompt. That is why it is cheap enough to run on environments where the reward is a binary test result.

## The gradient, concretely

For a one-token policy with logits `z` and `π = softmax(z)`, the score-function gradient of `log π(a)` with respect to `z` is

```
∂ log π(a) / ∂ z = onehot(a) - π
```

So a single REINFORCE-style update, using the group advantage as the weight, is

```
z += lr · A_i · (onehot(a_i) - π)
```

Read that line slowly, because it is the whole of S0:

- If `A_i > 0` (this rollout beat its siblings), every action *not* taken gets pushed down and the action taken gets pushed up.
- If `A_i = 0` — because all siblings scored the same — **the update is exactly zero**. Not small. Zero.

## What `toy_grpo.py` simplifies away

| Real GRPO | Toy |
|---|---|
| Multi-token sequence, per-token ratio | One token; ratio ≡ 1 |
| Clipped ratio | Omitted (with one step per batch it is inactive) |
| KL to a reference model | Omitted (the toy never drifts far) |
| Sampled from a real model | Sampled from a small softmax over actions |

The toy keeps the part that matters for an environment engineer: **sampling a group, and converting reward spread into a gradient.** If you want the remaining pieces, `s0_baseline/train_grpo.py` uses TRL, which implements the full objective.

## Why this matters on the environment side

The trainer computes `std(r)` from whatever your verifier returns. Two failure modes follow directly:

- **`std(r) = 0` by construction** — the task is too easy or too hard. Nothing to learn. Fix: calibration (S3).
- **`std(r) > 0` for the wrong reason** — the verifier is flaky, so identical rollouts score differently. The model is now optimising noise. Fix: determinism (S1).

There is no third failure mode at this layer. Every environment bug that matters is one of these two, wearing a different costume.
