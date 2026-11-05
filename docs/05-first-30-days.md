# First 30 working days — Platform

*Dec 1 2026 – Jan 9 2027 · a commitment, not a proposal*

---

## What this period is for

I arrive without a baseline: I don't yet know what our pass rates are, what an
episode costs, or where the delivery pipeline waits. Everything below is aimed at
producing that baseline and one small thing that is actually used. Judgements and
architecture come after, and will be worth more for having waited.

Because of the holiday period, this is **30 working days spread across six calendar
weeks** — three working weeks in December, a light week, then the write-up.

---

## Three deliverables

### 1 · By 12 December — the pipeline map

One page: a task from the customer's brief to the moment it lands in a lab's
training run. Every hand-off, every queue, and how long each one waits.

Produced by shadowing, not by asking: I'll sit with one delivery pod through a full
cycle. The artefacts are a diagram and a list of the three slowest steps, with the
wait time attached.

**Done when** someone in delivery reads it and recognises their own week in it.

### 2 · By 19 December — the baseline

On three environments that have already shipped: pass-rate distribution, determinism
rate, p99 reset latency, and cost per episode. Read-only; nothing changes during the
measurement.

**Done when** the numbers exist and are reproducible from a script in the repo,
not from a one-off notebook.

### 3 · By 9 January — one thing in use, and a proposal

**The thing:** the smallest tool that removes one of the three slowest steps from
deliverable 1. Success is not that it shipped; it is that someone uses it on a task
I didn't touch.

**The proposal:** one page for Q1 — what I'd work on next, and what I'd stop doing.

---

## What I need from the team

| Ask | Why | From |
|---|---|---|
| Read access: clone repos, env packaging, delivery tracking | to measure anything | Will |
| One lab feedback channel, even a redacted one | the only signal that says whether an env worked | Delivery lead |
| 3 × 1 hour: delivery lead, clone engineer, research | the map is wrong if it's built from documents | Will to broker |
| A sandbox quota to run baseline rollouts | ~400 rollouts per env to get ±5% on pass rate | Platform |
| Permission to instrument read-only | nothing in this plan changes a live environment | Will |

---

## What I will not do this month

- **No architecture proposals.** Before the baseline they would be opinions.
- **No changes to any environment a lab is currently training on.** A mid-run change
  makes the pass rate uninterpretable and the regression unattributable.
- **No tool without a named user who asked for it.** This is the rule that keeps a
  four-person platform team from becoming a ticket queue, and the one I would most
  like to be held to.

---

## How to judge whether it worked

At day 30 there should be answers — numbers, not adjectives — to five questions:

1. What is the pass-rate distribution across what we have shipped?
2. What does one episode cost us, end to end?
3. What is our determinism rate, and who measures it?
4. Who owns fidelity for each clone, and what is the number this month?
5. What is our first diagnostic when a lab says "we trained on it and nothing moved"?

If I can answer those, the month was worth its salary. If I can't, the plan was
wrong and I'd rather say so on 9 January than in March.
