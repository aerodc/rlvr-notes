# Roadmap

Five stages. Each one ends with a **number** you can put in a sentence, not just code that runs.

Total: roughly 20 evenings. If you are short on time, do S0, S1, S3 — those three carry most of the understanding. S2 is the one that looks most like the day job; do it if you have the evenings.

---

## S0 — See the mechanism (1 evening)

**Goal:** understand why reward *spread* is the thing, by watching a policy learn and then fail to learn.

**Do:**
```bash
make toy        # a task where 30% of actions are correct
make toy-flat   # a task where 100% of actions are correct
```
Read `s0_baseline/toy_grpo.py` — 120 lines, no dependencies. Find the three lines that compute the advantage.

**Produce:** the one-sentence answer to "why does a 100%-pass-rate environment teach a model nothing?"

**Done when:** you can explain it without looking at the code.

---

## S1 — Build a verifier you would ship (2–3 evenings)

**Goal:** write a verifier that a lab could audit, and prove it is deterministic.

**Do:**
- Read `s1_verifier/SPEC.md` for the contract: `Score(reward, subscores, evidence, verifier_version)`.
- Implement checks for the three task types in `tasks.py`.
- Run `make test`. The determinism test replays one trajectory 100 times and asserts a single distinct score.

**Produce:** `measures/RESULTS.md` row — *"verifier: N checks, evidence on every check, 0/100 nondeterministic on replay."*

**Done when:** you can hand the verifier to someone else and they can tell, from the output alone, why a rollout scored 0.

**The design point:** `evidence` is not logging. It is the output. A verifier that says "0" and nothing else costs a lab an hour per failed task.

---

## S2 — Make the environment a service (4–5 evenings)

**Goal:** the shift from "a function I call" to "a thing that runs somewhere and several trainers call at once."

**Do:**
- Read `s2_env_service/SPEC.md`. Implement the four endpoints in `server.mjs` if the skeleton is missing pieces.
- Run `make env` then `make env-test`.
- Then break it on purpose: run 20 resets concurrently and measure p50/p95/p99.

**Produce:** *"p99 reset latency under 20 concurrent clients: X ms."*

**Done when:** you can explain why `reset` must not be implemented by replaying a seed, and why the env should not be a Python import.

**The design point:** the wire contract is the product. Once `reset/step/score` is an HTTP contract, an environment can be written by one team and consumed by a lab that will never read your code.

---

## S3 — Calibrate difficulty (3–4 evenings)

**Goal:** derive the learnable band instead of taking it on faith.

**Do:**
```bash
make sweep
```
The script simulates rollouts at pass rates from 0.01 to 0.99 and measures the *expected within-group spread* at K=8. No model is involved — this is arithmetic on a binomial.

**Produce:** the chart, and *"expected advantage signal peaks at p=0.5 and is ~0 below 5% and above 95%."*

**Done when:** you can tell someone how to measure an environment's difficulty *before* training on it, and how many rollouts you need to detect it.

**The design point:** difficulty is measurable for free. You do not need a training run to know an environment is useless.

---

## S4 — Break your own verifier (3–4 evenings)

**Goal:** internalise the failure modes before someone else finds them.

**Do:**
- `s4_failure/leaky_verifier.py` implements a verifier that checks only the final string.
- A scripted "hacker" policy returns the expected answer without doing the task. It scores 1.0.
- Then apply the fix: check *state transitions*, not the final string, and require evidence from the world.

**Produce:** *"string-matching verifier: hacker scores 1.0, true success 0.0. state-based verifier: hacker scores 0.0."*

**Done when:** you can name the three ways a code environment gets hacked (answer leakage, checker shortcut, state that is not actually enforced).

---

## What you carry into the job

Not "I know RL". Three artefacts instead:

1. A measured **p99 reset latency** and a **nondeterminism rate** on something you built.
2. A **difficulty calibration** method, with a chart.
3. A one-page **memo**: what makes an environment valuable to a lab, and where the environments you have seen fall short.

That is a much stronger opening position than a reading list.
