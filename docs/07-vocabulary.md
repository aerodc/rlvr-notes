# The vocabulary, in the order it is forced on you

[`01-glossary.md`](01-glossary.md) is a dictionary. This is a chain: every word
below exists because the previous word created a problem.

Read it top to bottom once. After that it works as a reference.

---

## The chain

```
you have parameters θ and a number you want to improve     (a loss)
   → you cannot try every setting, so you need a direction  (the gradient)
   → the gradient only says which way, not how far          (the learning rate)
   → in RL the number is not differentiable w.r.t. θ        (the score-function trick)
   → so you minimise a surrogate, reward becomes a weight  (the surrogate loss)
   → that estimator is deafeningly noisy                    (the baseline)
   → what is left after the baseline is the useful part     (the advantage)
   → you still need a baseline, and a critic is expensive   (the group)
   → the optimiser will find shortcuts if you let it        (regularisation)
   → and it will stop exploring while doing so              (entropy)
   → and then the advantage goes to zero and it stops       (saturation)
```

---

## 1 · Parameters and loss

`θ` is everything the model can change — billions of weights. `L(θ)` is a single
number that says how wrong it currently is.

A loss has to satisfy three things:

1. be a **scalar** — an optimiser can only minimise a number,
2. be **differentiable** — otherwise there is no gradient (section 2),
3. be **aligned with what you actually want** — the one that is always violated,
   and always expensive.

Two standard shapes:

| Loss | Form | Where |
|---|---|---|
| mean squared error | `(ŷ − y)²` | regression |
| cross-entropy | `−log p(correct)` | classification, language models |

Language-model pretraining is just the average of the second:

```
L = − (1/N) Σ log p(token_i | prefix)
```

**Hold on to the shape `−log p(answer)`.** It comes back in section 4b with a
weight in front of it.

### Loss is not the metric

You **optimise** the loss; you **report** the metric. They drift apart:

- train loss down, validation loss up → overfitting,
- loss down, recall down → the loss never encoded what you care about.

A loss is a *proxy* for "good", not "good". Every Goodhart failure grows in that
gap — and reward hacking is the RL-shaped version of it.

## 2 · Gradient

`∇L(θ)` is the vector of partial derivatives: one number per parameter, saying how
much `L` changes if you nudge that parameter.

Two things it gives you, and no more:

- **direction** — the way uphill,
- **sensitivity** — how much each parameter matters right now.

```python
# s0_baseline/toy_grpo.py
grad[j] += adv * ((1.0 if j == a else 0.0) - probs[j])
```

That is a gradient being computed by hand. Everything else in the file is sampling
and bookkeeping.

**Why it is the only option:** with 10⁹ parameters there is no closed form. You
compute a direction and take a small step, repeatedly. Gradient descent is not a
good algorithm so much as the only feasible one.

## 3 · Learning rate

`θ ← θ − η·∇L`. The gradient says which way; `η` says how far.

- Too small → correct direction, glacial progress.
- Too large → you step over the valley and oscillate, or diverge.

Rung 3 of the ladder, and the first thing to check when training "doesn't work".

## 4 · The score-function trick

In supervised learning, `L` is an explicit function of `θ`, so backpropagation can
compute `∇L` exactly.

In RL the objective is

```
J(θ) = E_{a ~ π_θ}[ r(a) ]
```

and `r` is a **verifier** — a program with no derivative with respect to your
weights. You cannot differentiate the reward.

So you move the gradient somewhere differentiable:

```
∇θ J = Σ_a r(a)·∇θ πθ(a)
     = Σ_a r(a)·πθ(a)·∇θ log πθ(a)     because ∇π = π·∇log π
     = E[ r(a) · ∇θ log πθ(a) ]
```

The reward becomes a **weight**. The gradient comes entirely from `∇log π`.

This is why `log π` is everywhere in RL papers, and why the method is called
*policy gradient* rather than *reward gradient*. The reward was never
differentiated; it was only multiplied.

## 4b · The surrogate loss

You cannot write `L = J`, because `J` is not differentiable in `θ`. So you write a
**surrogate** whose gradient happens to point the right way:

```
L = − A · log πθ(a|s)
```

Check the sign: `∇(−A·log π) = −A·∇log π`, which is exactly the direction that
increases `J`. The surrogate is not the objective; it is a shape with the right
slope.

Now put it next to the pretraining loss from section 1:

```
SFT:   L = − 1 · log p(token)          weight is always 1
RL:    L = − A · log p(action)         weight is the advantage
```

**RL is supervised learning with a weight on each example, and the verifier sets
that weight.**

- `A > 0` → this trajectory is worth imitating
- `A < 0` → worth *anti*-imitating
- `A = 0` → weight zero; the example contributes nothing at all

That is the entire structural difference between SFT and RLVR — and it is why a
broken verifier is so destructive: **it is assigning the wrong weight to every
piece of data the model sees.**

### In RL the loss value means nothing

Only its direction does. Multiply every advantage by 10 and the loss scales by 10
while the gradient direction is unchanged — the only thing that moved is the step
size, and that is `η`'s job.

So in supervised learning, "loss 0.3" means the model improved. In RL, "loss 0.02"
means nothing whatsoever. This is why
[`s0_baseline/toy_grpo.py`](../s0_baseline/toy_grpo.py) deliberately prints no
loss: pass rate, `std(r)`, `|A|` and entropy all have physical meaning; the loss
does not.

## 5 · Variance, and why the naive estimator fails

`E[r · ∇log π]` is unbiased and almost useless on its own: `r` swings wildly
between samples, so the gradient estimate swings with it.

Two identical situations can produce opposite updates purely by luck. The estimator
is correct on average and unusable per step.

## 6 · Baseline

Subtract any function that depends only on the state:

```
E[ ∇log πθ(a|s) · b(s) ] = b(s) · ∇θ Σ_a πθ(a|s) = b(s) · ∇θ 1 = 0
```

**Unbiased, and lower variance.** This is the free lunch at the centre of the
field: you remove the part of the signal that is shared by every action, and what
remains is the part that discriminates.

The best baseline is `V(s)`, the value of the state — "how well do we usually do
from here". You do not know it, so you estimate it. How you estimate it is the
entire difference between algorithms.

## 7 · Advantage

```
A(s,a) = Q(s,a) − V(s)      "better or worse than usual, here"
```

The quantity that survives the baseline. Its role is purely as a weight:

- `A > 0` → push this action up
- `A < 0` → push it down
- `A = 0` → **nothing happens**

That last line is where most broken training runs live.

## 8 · The group

Actor-critic estimates `V(s)` with a second neural network. That is expensive, and
it has to be trained on the same noisy signal.

GRPO's answer: sample `K` rollouts of **the same prompt**, and use their mean.

```
V̂(s) ≈ mean(r_1 … r_K)
A_i  ≈ r_i − mean(r)
```

No critic. No extra parameters. This is why GRPO is cheap enough to run when the
reward is a single test result.

**The consequence you must internalise:** with `K = 1` the mean is the only sample,
so `r − mean = 0`, so every advantage is zero, so the gradient is exactly zero. Try
it:

```bash
python3 s0_baseline/toy_grpo.py --task learnable --group 1   # chance, forever
python3 s0_baseline/toy_grpo.py --task learnable --group 8   # climbs
```

`K` is not a tuning knob. It *is* the estimator.

## 9 · Regularisation

Once the optimiser can move, it moves toward whatever scores well — including
things that score well and are wrong.

In RLHF and RLVR the main regulariser is a **KL penalty** to the reference model:

```
objective = reward  −  β · KL(π_θ ‖ π_ref)
```

| `β` too high | the policy cannot move; reward barely rises |
| `β` too low or zero | the policy drifts into high-reward gibberish |

`β` is often very small — or zero — in RLVR, because a verifier is harder to flatter
than a reward model. Harder is not impossible; see [`s4_failure`](../s4_failure).

Related, and often also called regularisation:

- **Entropy bonus** — pay for exploration, discourage premature certainty.
- **Clip** — bound `π_θ/π_old` to `[1−ε, 1+ε]`, so one update cannot move the
  policy far. A trust region by another name.

## 10 · Entropy

`H(π) = −Σ p·log p`. How spread out the policy is.

- High → still exploring, lots of distinct rollouts.
- Falling steadily → converging, usually good.
- Near zero → the policy emits the same thing every time. The group has no spread,
  the advantage collapses to zero, and training has silently stopped.

Watch entropy the way you watch your fuel gauge, not the way you watch the loss.

## 11 · Saturation

The end state where pass rate approaches 1: every rollout succeeds, `std(r) = 0`,
`A = 0`, gradient zero. Nothing is broken and nothing is happening.

```
signal(p) = 1 − p^G − (1−p)^G
```

At `p = 0.98`, `G = 8`: **1.5%** of groups carry any information. See
[`s3_difficulty`](../s3_difficulty).

This is not a training bug. It is a property of the environment, and it is why
calibration is a product decision.

---

## Three sentences worth keeping

1. **The gradient is a direction, not a distance** — `η` decides the distance, and
   that is where most "it doesn't train" bugs live.
2. **The reward is never differentiated** — it is a weight, and the weight is
   `A`, and `A` is why the group exists.
3. **A regulariser is a deliberate loss of training score in exchange for not
   being gamed** — KL in the trainer, negative assertions in the verifier; same
   idea, different side of the wire.

## Pairs people mix up

| Not the same | Difference |
|---|---|
| loss / metric | you minimise one and report the other; when they diverge you have overfitting |
| gradient / loss | the loss is where you are; the gradient is which way to walk |
| reward / advantage | the reward is absolute; the advantage is relative to the group |
| baseline / critic | a baseline is any `b(s)`; a critic is a *learned* `V(s)` |
| regularisation / clip | both constrain the update; KL is a cost, clip is a hard bound |
| entropy / temperature | entropy is a property of the policy; temperature is a knob on sampling |
