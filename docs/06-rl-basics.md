# RL, from the bottom

A ladder with eight rungs. You do not need all of them to do this job, and rungs
1–4 are the ones that make the rest legible.

| Rung | Idea | You can stop here if… |
|---|---|---|
| 1 | The MDP: state, action, reward, return | you can name them in your own environment |
| 2 | Value, and why a baseline is free | you can prove subtracting a baseline doesn't bias the gradient |
| 3 | The policy gradient | you can say what gets reinforced and by how much |
| 4 | Advantage and the group | you can explain why `G` rollouts, not one |
| 5 | Actor-critic, PPO, GRPO as one family | you know what each one bought and what it cost |
| 6 | RLHF → RLVR: swapping the reward model for a verifier | you can say why a judge is not a verifier |
| 7 | What breaks: saturation, hacking, collapse | you can diagnose a flat reward curve |
| 8 | Where the environment sits | you can point at the two places your job touches the algorithm |

---

## Rung 1 — The MDP, in the vocabulary of an environment

Reinforcement learning is a loop with five parts:

```
S  states        what the world can be
A  actions       what the agent can do
P  transition    s, a  →  s'          (the environment)
R  reward        s, a  →  r           (the verifier)
γ  discount      0 < γ ≤ 1
```

Map that onto a task in a cloned app and it stops being abstract:

| MDP | Mail app environment |
|---|---|
| `S` | the whole mailbox state: messages, folders, labels |
| `A` | `archive(id)`, `label(id, tag)`, `reply(id, body)` |
| `P` | your `step()` function |
| `R` | 1 if the final state satisfies the task, else 0 |
| `γ` | irrelevant here — these are one-episode, terminal-reward tasks |

Two things follow immediately, and both are your responsibility:

- **`P` must be deterministic.** If the same `(s, a)` can produce different `s'`,
  the state the verifier sees is partly luck, and the reward inherits that luck.
- **`R` must depend on `s'`, not on what the agent said.** The reward is a function
  of the world, which is why `verifier(state)` beats `verifier(transcript)`.

## Rung 2 — Return, value, and the free lunch

The **return** is the discounted sum of what happens next:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …
```

The **value** of a state is the expected return from it under the current policy:

```
V^π(s) = E_π[ G_t | s ]
```

Now the fact that everything else rests on. For any function `b(s)` that depends
only on the state:

```
E_{a~π}[ ∇θ log πθ(a|s) · b(s) ]  =  b(s) · ∇θ Σ_a πθ(a|s)  =  b(s) · ∇θ 1  =  0
```

Which means: **you can subtract any state-dependent baseline from the return and
the gradient is unchanged in expectation.** You get a lower-variance estimate for
free. This is not a trick; it is the reason policy-gradient methods work at all.

## Rung 3 — The policy gradient

Increase the probability of actions that led to better-than-expected outcomes:

```
∇θ J(θ) ∝ E[ ∇θ log πθ(a|s) · A(s,a) ]
```

where the **advantage** is the return minus the baseline:

```
A(s,a) = Q^π(s,a) − V^π(s)      "better or worse than usual, here"
```

Read the sign: positive advantage → push that action up. Negative → push down.
Zero → **nothing happens**, and that is exactly the case the whole field
under-estimates.

## Rung 4 — Where the baseline comes from, and why `G > 1`

You do not know `V^π(s)`. You have to estimate it. GRPO estimates it with the
mean reward of `K` rollouts of the *same prompt*:

```
V̂(s) ≈ mean(r_1 … r_K)

A_i ≈ r_i − mean(r)
```

That is the entire idea. All `K` rollouts share the same state, so their mean is a
sample estimate of the value of that state. No critic network, no second model,
no extra parameters. This is why GRPO is cheap enough to run against environments
whose reward is a single test result.

The standard deviation division in the real formula is a further normalisation —
it changes the effective step size, not the direction.

### The experiment that makes it concrete

```bash
python3 s0_baseline/toy_grpo.py --task learnable --group 1    # learns nothing
python3 s0_baseline/toy_grpo.py --task learnable --group 8    # climbs cleanly
```

With `G = 1`, the mean is the only sample, so `r − mean = 0`, so every advantage is
zero, so the gradient is exactly zero. The policy sits at chance forever. You are
watching the baseline being absent.

Real REINFORCE with one sample avoids this by using the raw return as the weight
(high variance, no baseline). GRPO normalises away that option: **the group is not
a tuning knob, it is the estimator.**

That is also why `K` costs money. Each rollout is a full environment episode, and
`K` multiplies it.

## Rung 5 — One family, not four algorithms

| Method | Baseline | Update constraint | What it buys |
|---|---|---|---|
| REINFORCE | none | none | simplicity; terrible variance |
| Actor-critic | a learned `V(s)` | none | low variance; another network to train |
| PPO | learned `V(s)` | clip the ratio | stability |
| GRPO | **the group mean** | clip + KL | no critic; works with binary rewards |

Everything after Rung 4 is about variance, stability, and cost. None of it changes
the two questions you actually own: where the reward comes from, and whether the
group has spread.

## Rung 6 — RLHF → RLVR

RLHF puts a **learned reward model** where `V` sits. It is expensive, it drifts,
and it can be argued with. RLVR replaces it with a **program**:

```
RLHF:  reward = reward_model(prompt, response)      judged
RLVR:  reward = verifier(final_state)               checked
```

The moment the reward is a program, two things change:

- The signal is (potentially) noise-free and cheap — that is the whole sell.
- The failure mode moves from "the model flatters the judge" to "the model games
  the check".

You are building the check. That is the job.

## What to skip

- PPO's clip derivation, GAE, value-function bootstrapping.
- Q-learning, TD, off-policy correction.
- Convergence proofs, exploration theory.
- Implementing a trainer from scratch.

All of it is real RL. None of it changes a decision you will make about an
environment.

## Four things to read, in this order

1. **DeepSeek-R1** — §2 for GRPO as used at scale, §4 for the failure cases.
2. **DeepSeekMath** — where GRPO actually came from; the group-mean derivation.
3. **Epoch AI, *An FAQ on Reinforcement Learning Environments*** — the landscape.
4. **Sutton & Barto, ch. 13** — only if Rung 3 still feels like a formula you
   memorised rather than something you could rederive.
