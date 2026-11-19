# What labs actually buy

An evidence set, not an opinion. Two kinds of evidence:

1. **What they hire for** — job postings are the least polished, most honest public
   document a lab produces about its internal bottlenecks.
2. **What they believe** — the public writing that shapes where money goes.

The purpose is to answer one question from the buyer's side: *what do these labs
need that they cannot build quickly enough themselves?*

---

## Part 1 · What the postings say

| Lab / role | The line that matters |
|---|---|
| Moonshot, **Research Scientist / Engineer – Agentic RL** | "Invent the next leaps in agentic, multi-agent RL — **algorithms, environments, infrastructure**" |
| Moonshot, **RL Infra 研究工程师** | "对 RL 训练中的实际问题（**训推不一致、Rollout 长尾**）排查和解决有经验"; 加分项含 **"环境 scaling"** 方向论文 |
| OpenAI, **Agent Post-Training, Frontier Evals and Environments Research** | "Create ambitious reinforcement learning environments"; "analyze the scalability, reliability, and **variance** of evaluation methods"; ~$300k–500k |
| Anthropic, **Research Engineer, Cybersecurity RL** | "Design and implement RL environments for cybersecurity applications" — hiring the **domain expert**, not the vendor |
| Moonshot, **安全研究与攻防能力专家 (RL Team)** | 把"漏洞挖掘、二进制逆向、sandbox 逃逸、Exploit 构造"转化为"可泛化、可自我演进的训练 Pipeline 与强化学习环境" |

### Six inferences

1. **Environments are named as a research pillar**, alongside algorithms and
   infrastructure. Not data supply, not a task set — a pillar.
2. **MARL and self-play appear in the same breath as `fork`-able sandboxes.** A
   multi-agent environment is a different product from a task with a grader: shared
   mutable world, per-agent credit assignment, opponent policy distributions.
3. **Rollout tail latency is an explicit hiring criterion.** The slowest trajectory
   in a batch sets the step time. p99, not mean.
4. **The sandbox is treated as an attack surface** — "simulator security" is a
   nice-to-have, and AgentENV's microVM decision came from container incidents.
5. **They hire the domain expert and the systems engineer as a pair.** Security
   needs kernel/exploit people; 3D needs Blender/CAD people. This is the part a
   generalist vendor cannot supply.
6. **OpenAI files evals and environments under one role.** To a frontier lab,
   measuring and generating are the same activity with different reward plumbing.

### The distinction that matters

There are **two different jobs** hiding in "environments":

| | What it is | Who does it |
|---|---|---|
| **Inventing paradigms** | MARL, self-play, long-horizon autonomy, new kinds of world | The role above: RL mastery + systems + shipped artefacts |
| **Producing volume** | 500 Gmail/Slack/Salesforce tasks with verifiers and calibrated difficulty | Plausibly bought |

Confusing these leads to the wrong anxiety. **They are not trying to replace a task
factory; they are trying to invent a new kind of task.** The threat to a vendor is
not "the lab builds Gmail clones" — it is "the lab invents multi-agent environments
and the vendor cannot build them."

---

## Part 2 · What the thesis says

Will DePue, [*A Stargate for Data*](https://willdepue.net/writings/a-stargate-for-data/)
(July 2026). The strongest public statement of the bull case for the data business.

### Four claims that hold up

1. **The binding constraint moved from compute to data.** The compute project is
   funded; the data project is not.
2. **Compute is a commodity, data is not.** Everyone buys the same chips. Models
   have felt similar because they were trained on substantially the same internet;
   as corpora diverge, models will diverge.
3. **The O-ring argument.** "A dataset that buys a 1% bump can justify a previously
   unjustifiable collection cost when it is the difference between a system that
   does 99% of a job and one that does all of it." This is the pricing argument for
   any environment vendor, and it is stronger than any argument from effort.
4. **Non-stationarity does not destroy value.** Demand keeps shifting; the long tail
   keeps getting longer; "the value of the marginal dataset goes up, not down."

### The weakest link

The essay equates *data* with *collected data*, and treats synthesis as a footnote
("fundamental progress is slow and highly unpredictable").

That conflates two different things: an algorithmic breakthrough, and **engineering
a generator**. A parameterised task template that samples new instances is not a
research result — it is production engineering, and it is already how difficulty
treadmills get built. Generators have near-zero marginal cost and do not depend on
anyone's private corpus.

**A vendor who sells collected data is selling a corpus. A vendor who sells
generators plus verifiers is selling a factory.** The essay argues for the first and
accidentally describes the second.

*(Also worth noting: $7B today, ">10x by 2030", headline ">$100B". The numbers are
rhetoric, not a model. Treat the direction as the claim, not the magnitude.)*

---

## Part 3 · The reframe

If you read the essay against itself, what is scarce is not data. It is **expert
time** — people who can say what "correct" means in a domain.

Mercor's scale is expert hours. "Niche expert workflows, live environments, and
evaluations requiring increasingly obscure talent" is a description of expert time.
The missing data is "tacit knowledge that lives in people's heads."

**A clone is a way to amortise one expert's judgment across ten thousand tasks.**

That reframes the product: it is not a dataset, it is **an executable form of expert
judgment**. That framing prices better, defends better, and survives automation
better — because a generator plus a verifier is the thing that keeps producing after
the corpus is exhausted.

---

## Part 4 · What it means for an environment vendor

1. **Price on the O-ring, not on effort.** "This closes the last 1% of the task
   distribution" is a different number from "this took 400 hours."
2. **Exclusivity becomes a platform requirement.** If data is a moat, a lab will not
   accept an environment resold to a competitor. Provenance, licence scope and
   provable isolation move from nice-to-have to contract terms — the `spec_hash` in
   [`s2_env_service/SPEC.md`](../s2_env_service/SPEC.md) is the first inch of this.
3. **Sell capability, not corpus.** The comp for "data vendor" is a services multiple;
   the comp for "infrastructure" is not. Same business, two stories, two multiples.
4. **The treadmill is hedged, not terminal.** Environments decay, but the marginal
   value of each *new* environment rises, because what remains is harder and more
   domain-specific. That is a hedge — provided you can keep producing.
5. **Do not compete with the runtime.** See
   [`docs/10-agentenv-lessons.md`](10-agentenv-lessons.md): ship contents.

---

## Part 5 · What it means for you personally

| Invest in | Avoid |
|---|---|
| Determinism, fork, snapshot, tail latency | Being the person who hand-writes tasks |
| Difficulty calibration as a measurable system | Competing on "realism" of clones |
| Multi-agent and self-play environments | The commodity layer of general software operation |
| Verifiers that survive a red-team agent | Any claim you cannot put a number behind |

The sentence to carry: **the scarce input is not data, it is people who can tell a
good environment from a plausible one — and then build the thing that tells them.**

---

## Caveats

- Five postings and one essay is a sample, not a survey. Postings over-represent what
  a lab is *missing*, not what it does well.
- DePue is a founder in this market and writing from a position. The thesis is
  directionally right; the numbers are advocacy.
- Job postings go stale. The Moonshot agentic-RL role was posted in April 2026 and
  is still open and marked urgent seven months later — which tells you something
  about scarcity, and something about the bar.
