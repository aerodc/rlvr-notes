# The mental model

Five minutes. This is the frame everything else hangs on.

## The loop

Training a model with RLVR is six steps. The environment owns two of them.

```
1. take a prompt from the dataset                     ─┐
2. roll out K completions from the current policy      │  ← the environment
3. score each completion  →  r_1 … r_K                ─┘
4. compute group-relative advantage  A_i = (r_i - mean) / std
5. take a policy gradient step: up-weight above-average rollouts
6. pull back toward the reference model (KL penalty)
```

Steps 2 and 3 are yours. Steps 4–6 belong to the trainer, and you mostly need to understand them well enough to know **what makes steps 2 and 3 useful or useless**.

## The one equation that matters

```
A_i = (r_i - mean(r)) / (std(r) + eps)
```

The advantage is computed **inside a group**. Its magnitude is driven by the spread of rewards in that group. Two consequences, and they are the whole job:

- **All rewards equal → no gradient.** Pass rate 100% or 0% means `std(r) = 0`. The sample is thrown away. This is not "a bit less useful"; it is exactly zero information.
- **Noisy reward → noisy gradient.** A verifier that returns different scores for the same trajectory inflates `std(r)` with something that is not signal. The model learns to satisfy the noise.

So an environment is valuable in proportion to how much *real* spread it produces, minus how much *fake* spread it produces.

## Where the environment sits

```
        ┌──────────────────────────────────────────┐
        │  what you build                          │
        │                                          │
prompt ─┤  task spec → sandbox → trajectory       ├─→ reward
        │                     ↘ evidence           │
        └──────────────────────────────────────────┘
                     ↑                    ↑
              reset / step         deterministic check
```

Three separable things, and it pays to keep them separable:

| Piece | Question it answers | Failure mode |
|---|---|---|
| **Task spec** | what counts as done? | ambiguous success criteria → the model is right and the verifier says no |
| **Sandbox** | where does the agent act? | unreproducible state → you cannot tell a good rollout from a lucky one |
| **Verifier** | did it actually happen? | checks something the model can fake → reward hacking |

## The three numbers you own

If you are the platform person, these are the numbers that decide whether the environment is worth anything. Everything else is instrumentation.

| Number | What it drives | Bad value looks like |
|---|---|---|
| **Pass rate** at rollout temperature | how much learning signal exists | 0.98 or 0.02 |
| **Nondeterminism rate** — how often the same trajectory scores differently | how much of the signal is noise | anything above ~0 |
| **p99 reset latency** | cost per training step, hence margin | seconds, when the trainer needs hundreds of resets per step |

A quick sanity check on the third one: cost per gradient step is roughly

```
prompts_per_batch × generations_per_prompt × (reset + rollout + score)
```

At 64 prompts × 8 generations × a 1.5s reset, you are spending 12 minutes of wall clock per step on resets alone. That number is why "reset latency" is a platform metric and not an infrastructure detail.

## The asymmetry to keep in mind

The trainer will happily consume a bad environment and produce a model that got slightly better at the wrong thing. Nothing will crash. The curve will even look reasonable.

That is why the failure modes in S4 matter more than the happy path in S0: **a broken verifier does not fail loudly, it fails in the lab's next eval.**
