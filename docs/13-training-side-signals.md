# Signals from the training side

Three public artefacts, one question: **what does the trainer actually need from an
environment?** None of them are about environments directly, which is what makes
them useful — they describe the thing that consumes an environment, so the
requirements fall out of the constraints rather than the marketing.

| Artefact | Date | What it is |
|---|---|---|
| [RL infrastructure at Microsoft AI](https://microsoft.ai/news/rl-infrastructure-at-microsoft-ai/) ("Rocket") | Nov 2026 | an in-house **asynchronous** RL framework, published in unusual detail |
| [Kimi K3 / AgentENV](https://github.com/kvcache-ai/AgentENV) | Jul 2026 | a **co-located** RL system with microVM sandboxes (see [`10-agentenv-lessons.md`](10-agentenv-lessons.md)) |
| [Lambert, *Pushing Frontiers in Open LM Post-Training*](https://natolambert.com/slides/2026/cornell-post-training/talk) | Nov 2026 | six open problems in the field, from Trillium Labs |

---

## 1 · Rocket: the bottleneck is inverted

Rocket trains MAI Base 2 — 3T parameters, thousands of GPUs, **runs that last weeks
to months**. Three choices matter more than the rest.

**They deliberately over-allocate GPUs to the learner.**

> "We overallocate GPUs to our learner so that RL jobs are bottlenecked by inference
> capacity, not training capacity."

Some learner GPUs sit idle on purpose. That is an inversion of the usual instinct,
and it is correct once generation dominates wall-clock: a fast trainer that waits on
a rollout queue is a slow trainer. They auto-scale in-flight rollouts to match
inference capacity and report a completed group spending **12 seconds** in the
controller queue — treated as a service level, not an accident.

**They measure staleness instead of hoping about it.**

"Learner and sampler drift apart as rollouts age." They plot mean `|log-prob
difference|` against rollout age and fit a line, **R² = 0.76**. Staleness stops
being a vibe and becomes a quantity with a budget, enforced by staleness limits in
the scheduler.

**They chase bitwise agreement between sampler and learner.**

The sampler keeps tensor parallelism for speed, so they **replicate the attention
output-projection weights** to remove the cross-GPU reduction; the learner runs
**without TP** so its summation order matches; FA4 is aligned across learner forward,
sampler prefill and sampler decode; sliding-window K-block boundaries are made to
coincide. Their own summary: *"making sampler and learner match exactly meant
chasing down every kernel, reduction order, and rounding difference."*

A small-scale A/B (their Figure 6) starts two RL runs from the same checkpoint, with
and without bitwise matching, and gets different curves. **Numerical mismatch is not
a precision detail; it changes what the run learns.**

---

## 2 · AgentENV: environment state is training state

Covered in [`10-agentenv-lessons.md`](10-agentenv-lessons.md); the one line that
matters here:

> "resumable microVM sandboxes to preserve long-lived model **and environment** state"

with `pause` / `resume` / `snapshot` / `fork` as first-class lifecycle primitives,
and incremental checkpointing because full copies cannot be taken often enough.

---

## 3 · The two architectures, side by side

They solve the same problem from opposite ends, and that is the point.

| | Kimi K3 / AgentENV | Microsoft Rocket |
|---|---|---|
| Coupling | **co-located**, partial rollouts | **asynchronous**, explicit staleness budget |
| Bottleneck they optimise for | environment lifecycle | inference capacity |
| Central failure mode | unreconstructible long-horizon state | learner/sampler drift |
| Primary primitive | `pause` / `snapshot` / `fork` | bitwise alignment + fast weight transfer |
| What they both refuse | to treat "the policy that generated this" and "the policy being updated" as the same thing |

**Read together, they say one thing: the gap between generation and training is a
first-class engineering problem.** The environment sits inside that gap.

---

## 4 · Lambert's six open problems

From a 51-slide talk by the author of the RLHF Book, now Executive Director of
Trillium Labs:

1. **Harness specialization and design** — how to design environments for context
   management that enable new paradigms of using LLMs
2. Character training
3. **Open-source RL environments** — scalable, open data methods
4. Reasoning efficiency in RL
5. Scaling laws for RL
6. **Multi-agent scaling laws**

Two of the six are environments, and the first one is not "more environments" — it
is **a new kind of environment**, built around context management rather than around
a task.

His framing sentence is the one to keep:

> "Frontier, complete post-training is rapidly becoming less accessible due to
> resource requirements, similar to how pretraining looked at the beginning of 2023."

Post-training is becoming capital-intensive in the way pretraining was. That is why
Rocket runs for months, and why a lab that cannot afford months buys inputs instead.

---

## 5 · What a qualified environment has to provide

Put the three sources in one column and the rows are forced, not chosen.

| # | Requirement | Forced by | What breaks without it |
|---|---|---|---|
| 1 | **Byte-identical `reset`** | AgentENV; Rocket's reproducibility needs | pass rates are not comparable across rollouts |
| 2 | **Snapshot of live state** | AgentENV | long-horizon trajectories cannot be trained at all |
| 3 | **`fork`** | AgentENV; Lambert #6 (multi-agent) | no side-effect-free judging, no self-play |
| 4 | **`pause`/`resume` in tens of ms** | AgentENV; Rocket's staleness budget | rollout ages, so the advantage measures a policy that no longer exists |
| 5 | **p99 tail control** | Rocket (batch step time) | one slow environment sets the cost of the whole step |
| 6 | **Multi-agent capability** | Lambert #1, #6; Moonshot's agentic-RL posting | MARL and self-play are simply unavailable |
| 7 | **Deterministic scoring** | all three | fake variance; the model learns the noise |
| 8 | **A harness/context interface** | Lambert #1 | the environment cannot support the next usage paradigm |

**Notice what is not on the list: realism, task variety, domain coverage, visual
fidelity.** Those are the things vendors advertise. The things trainers are blocked
on are all *systems* properties.

That is the useful asymmetry. **A qualified environment is qualified almost entirely
by its systems behaviour; content is the easy half.** It is also why a four-person
platform team can matter in a company that employs seventy people to build content.

---

## 6 · The demand multiplier

Lambert's history has a detail worth pulling out. The 2026 recipe is **MOPD**:
train *N domain-specialist teachers*, then merge them into one student with
multi-teacher on-policy distillation. MiMo-V2-Flash introduced it; DeepSeek V4,
Nemotron 3 Ultra and recent GPT models reportedly use it.

If that is the shape, then environment demand scales with **the number of
specialists**, not the number of models:

```
one model  →  N domain teachers  →  N distinct environment distributions  →  1 student
```

Eight specialists is eight environment portfolios. That is a demand multiplier, and
it pushes in the opposite direction to "environments decay as models improve".

---

## 7 · What to do with this

**If you build environments:** the eight rows above are your acceptance criteria, and
they are testable before a customer sees the environment. The two cheapest to get
right now are `reset` determinism and p99 — both are already in
[`s2_env_service`](../s2_env_service).

**If you are deciding where to spend your own time:** every row on that list is a
systems skill, not a content skill. Content can be bought or generated; the systems
properties cannot be retrofitted.

**If you are writing anything for a lab:** do not lead with realism. Lead with the
shape of the rollout, the tail latency, and whether the state can be frozen. Those
are the three things their scheduler cares about.
