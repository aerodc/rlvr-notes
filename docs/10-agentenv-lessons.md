# What a frontier lab's environment platform actually looks like

Source: **Kimi K3: Open Frontier Intelligence** (Kimi Team, July 2026) —
[arXiv 2607.24653](https://ar5iv.labs.arxiv.org/html/2607.24653v1) — and the
infrastructure it open-sourced alongside the weights, all MIT:

| Repo | Layer | Why it exists |
|---|---|---|
| [AgentENV](https://github.com/kvcache-ai/AgentENV) | execution environment | sandboxes for long-horizon agentic RL |
| [MoonEP](https://github.com/MoonshotAI/MoonEP) | GPU communication | balanced expert parallelism at 896 experts |
| [FlashKDA](https://github.com/MoonshotAI/FlashKDA) | kernel | chunkwise Kimi Delta Attention on Hopper |

All performance numbers below are **project-reported and not independently
reproduced**. Treat them as shape, not as SLO.

---

## The one sentence that changes how you think about the job

From the paper's introduction:

> For million-token agentic RL, our **co-located system** combines **partial
> rollouts, external KV-cache retention, adaptive throttling and resumable
> microVM sandboxes** to preserve long-lived model **and environment** state.

Read the last four words again. **Environment state is training state.**

Once a trajectory is hundreds or thousands of tool calls, the sandbox's processes,
memory, filesystem and mutated tool state are not reconstructible. They are
state you must preserve, exactly like a KV cache. Every design decision below
follows from that one sentence.

---

## Five decisions worth copying

### 1 · Firecracker microVMs, not containers — and it was an incident, not a preference

The report is blunt: early container-based sandboxes hit **kernel panics and
deadlocks from unexpected agent operations.**

The agent will do things you did not enumerate. A container shares the host
kernel; a microVM does not. This is not a performance trade — it is the
difference between a sandbox and a suggestion.

**For an environment vendor:** the moment an environment runs agent-authored code,
isolation stops being a design choice. If someone proposes "just use Docker", the
answer is a link to this paragraph.

### 2 · `fork` — branch from a byte-identical world

The stated use is **"side-effect-free judging"**, plus parallel exploration.

This is the primitive that makes a verifier trustworthy. You cannot re-score a
trajectory you can no longer reproduce, and you cannot run a red-team agent against
a state without perturbing it. `replay()` in [`s1_verifier`](../s1_verifier) is the
toy version of this; in production it needs a frozen world, not a stored
transcript.

### 3 · Partial rollouts — generation and environment are two production lines

A long trajectory does not fit in one generation call. So: run a segment, pause the
sandbox, move the KV cache out, wait for the batch to fill, resume.

That is a throughput architecture. It means rollout and training run at different
speeds, with a buffer between them — and the buffer is where staleness lives.

### 4 · Adaptive throttling — backpressure, because stale data is wrong data

If sandboxes produce faster than the trainer consumes, the policy that generated a
trajectory is no longer the policy receiving the gradient. The advantage computed
from that trajectory is then measuring the wrong policy.

This is the off-policy staleness problem, expressed as a rate limiter. Worth
remembering the next time a training curve wobbles for no visible reason.

### 5 · Incremental checkpointing — dirty pages, not whole VMs

Reported: **checkpoint as low as ~133 ms, resume as low as ~49 ms.**

The storage stack explains the numbers: OCI images, OverlayBD, a bounded local hot
cache, object storage or a distributed filesystem for persistent snapshots, and
`ublk` for I/O. None of that is decorative — full copies cannot be taken at the
frequency RL needs.

---

## The metrics this changes

The v0 contract in [`s2_env_service/SPEC.md`](../s2_env_service/SPEC.md) measures
`reset`. After reading AgentENV, that looks like measuring the wrong thing.

| Metric | What it actually drives | When it is bad |
|---|---|---|
| p50/p99 `reset` | cost of starting a rollout | seconds |
| **p50/p99 `pause` + `resume` per tool call** | **cost of a long trajectory — the real denominator** | tens of ms × hundreds of calls |
| p50/p99 `snapshot` size and write throughput | how often you can afford a checkpoint | GBs per snapshot |
| `fork` latency | how cheaply you can re-score or red-team | seconds |

The arithmetic that matters: a 300-tool-call trajectory paying 40 ms of pause/resume
per call is **12 seconds of pure overhead per rollout**, before the model generates
a token. That is a throughput number, a cost number and a product number at once.

---

## What this means for an environment vendor

**AgentENV is MIT-licensed, Kubernetes-deployable, and exposes an E2B-compatible
API.** That is a lab telling you what its standard runtime is going to be.

The conclusion follows directly:

> **Do not compete with the runtime. Ship things that run inside it.**

Deliver task specs, verifiers, images and difficulty metadata — the *contents* of
an environment — and let the customer's AgentENV (or equivalent) provide the
lifecycle. Competing on runtime means a four-person platform team against a
frontier lab's infrastructure team plus an MIT-licensed community project.

This is the same conclusion [`s2_env_service/SPEC.md`](../s2_env_service/SPEC.md)
reached from first principles: an environment should be a **portable artifact**,
not a hosted service.

---

## How to read it, in order

1. **The paper's introduction and infrastructure section** (~30 min). Look for the
   architecture choices, not the benchmark table.
2. **[AgentENV's README](https://github.com/kvcache-ai/AgentENV)** (~30 min). The
   lifecycle primitives and the deployment docs are the useful part.
3. **[This Chinese digest](https://github.com/qfc-network/ai-infra/blob/main/moonshot/kimi-k3-infra/zh.md)**
   (~30 min). The trade-off table is better than anything in the paper.
4. **AgentENV's Kubernetes documentation** (~20 min). This is the answer to "how
   will a customer actually run what you sell them?"

## Caveats

- Numbers are project-reported, on their hardware, with their workloads.
- K3's own release is a bet on a specific architecture (KDA + extreme MoE); some
  of this infrastructure is shaped by that bet and will not transfer.
- AgentENV is young. Reading it teaches you the *shape* of the problem; it is not
  yet a standard you should assume will exist in three years.
