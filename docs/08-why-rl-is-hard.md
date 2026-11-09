# Why this field is still new, and where the difficulty actually is

*Written for two uses: the answer to "so what do you make of this field?", and the
reasoning underneath [`04-memo.md`](04-memo.md).*

---

## "New" means three different things

RL as a word covers three layers with wildly different ages. Conflating them is why
people either dismiss the field as solved or treat it as magic.

| Layer | Age | State |
|---|---|---|
| **Mathematics** — MDPs, policy gradients, Bellman | 1957–1992 | settled; it is in textbooks |
| **Algorithms** — PPO 2017, GRPO 2024 | 2–9 years | broadly converged |
| **Production practice** — running RL at frontier scale, and *selling environments to people who do* | **~2 years** | no consensus, no standard library, no shared benchmark |

The mathematics is old. The algorithms are recent but stable. **The third layer is
what is new** — and it is new not because it is intellectually hard, but because
almost nobody has done it, so nobody has written down how it should be done.

---

## Five symptoms of an immature discipline

These are not opinions about difficulty. They are evidence that the field has not
yet accumulated the things mature fields accumulate.

### 1 · No shared abstraction

Supervised learning converged on `dataset → model → loss → optimiser`. Every
framework agrees, and a model written in one can be trained in another.

RL has rollout engine, weight synchronisation, environment, verifier, advantage
estimator, batch layout — and TRL, verl, OpenRLHF and NeMo-RL slice them
differently. **The environment interface only started standardising in 2025–26.**
In a mature subfield, that would be unthinkable.

### 2 · The policy being scored is not the policy being trained

Rollouts come from vLLM or SGLang. Training happens in FSDP or Megatron. The
numerical paths differ, so the thing that generated the trajectory and the thing
receiving the gradient are not quite the same policy. This is *training–inference
mismatch*, and it is still an active research topic — [ICML 2026 has work on
removing the inference nondeterminism introduced by tensor-parallel degree](https://proceedings.mlr.press/v306/zhang26ag.html),
and [naive low-precision rollouts with BF16 training break RL outright](https://browse-export.arxiv.org/pdf/2607.15810).

This class of bug **does not exist in supervised learning.** It is also a leading
hidden cause of "the curve just doesn't move".

### 3 · Small runs do not predict large runs

In supervised learning you debug on a small dataset and extrapolate. In RL,
something that trains cleanly at 1B frequently does not at 70B. You cannot use a
cheap experiment to decide whether the expensive one is worth running — which
raises the cost of iteration by an order of magnitude and makes judgement worth
more than search.

### 4 · Even the evaluation is contested

This is the sharpest one. [A NeurIPS 2025 paper argues that RLVR does not expand
reasoning capacity beyond the base model](https://papers.neurips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html) —
that it sharpens the distribution rather than reaching new capabilities, and that
at large `k` the base model could already get there. [A companion line of work
frames this as "the illusion of reasoning", via support and
entropy](https://icml.cc/virtual/2025/52438).

Whether or not they are right, the consequence for an environment vendor is
immediate: **if RLVR elicits rather than teaches, then a good environment is one
that covers the region the base model can already reach but does not yet
reliably choose.** That is a different design target from "teach it something new",
and the field has not settled which one it is.

### 5 · The cost structure is inverted

Supervised learning spends its FLOPs on the backward pass. RL spends them on
**generation**, and it is latency-bound: every gradient step waits for
`prompts × K` episodes to finish. This is why reset latency is not an
infrastructure detail — it is the denominator of training throughput.

---

## Where the difficulty is — two lists

### Field level (nobody solves these soon)

1. **Reward design.** Goodhart is not a bug to fix, it is a constraint to manage.
   You are always trading "verifiable" against "expressive".
2. **Long-horizon credit assignment.** On a 30-step task, which step was right? A
   binary terminal reward does not say.
3. **Reproducibility.** Randomness from sampling, from the engine, and from the
   environment compound. A good result is hard to repeat and therefore hard to
   trust.
4. **Throughput.** See above.

### Job level (these are yours)

1. **Fidelity and the transfer gap.** Whether the environment teaches the thing the
   lab wanted, rather than the thing the clone happens to reward.
2. **Difficulty is a moving target.** As models improve, your environments decay.
   This is a treadmill business, not a one-off delivery.
3. **The expressiveness ceiling of a verifier.** You can only assert what you can
   write down. Everything you cannot write down is a hole, and holes get found.
4. **You are on the critical path.** One slow environment slows every lab's every
   step.

---

## The analogy

**RL today is computer vision before ImageNet.**

The methods existed — SIFT, HOG, SVMs — and the mathematics was clear. What was
missing was a shared benchmark, a shared abstraction, and a shared way of deciding
whether a result was real. Progress came from individual craft in individual labs
rather than from the field accumulating.

That is exactly where RLVR sits: **best practice travels by word of mouth, not by
documentation and libraries.**

---

## What it means if you are joining this

The scarcity is not in algorithms. Anyone can read the GRPO paper; open
implementations are everywhere. The scarcity is in **judgement**:

- is this pass-rate distribution usable, or is it silent?
- is this curve saturated, or broken, or just slow?
- will this verifier survive an agent that is trying to game it?
- is this measurement real, or am I measuring my own harness?

None of that is in a textbook, because the textbook has not been written. It comes
from having done the loop yourself: built an environment, measured it, watched it
get hacked, fixed it.

**Three things to remember if you forget everything else:**

1. The field is new at the layer you work on, and old at the layer people quote at
   you. Know which one a claim is about.
2. Every hard problem here is a reward-shaping problem wearing a different costume.
3. The scarce resource is not compute or code. It is people who can tell a good
   environment from a plausible one.
