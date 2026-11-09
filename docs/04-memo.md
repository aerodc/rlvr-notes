# What makes an RL environment worth buying

*draft v0.1 · written before day one · numbers marked ⬜ come from measurement, not assumption*

## The claim

An environment's value is not its realism, and not its difficulty. It is the amount
of **real** reward spread it produces per dollar, minus the amount of **fake** spread
it produces.

Spread is what GRPO converts into gradient: `A_i = (r_i − mean) / std`. Real spread
means the model can be right or wrong. Fake spread means the same behaviour scores
differently twice. Every environment vendor optimises the first. Very few measure
the second, and it is the cheap one.

The arithmetic is unforgiving. For a group of `G` rollouts at pass rate `p`, the
probability that a group carries **any** signal is

```
1 − p^G − (1−p)^G
```

At `p = 0.98` and `G = 8`, that is **1.5%**. A saturated environment is not "almost
solved" — it is silent, at the same cost as a useful one. The mirror image at
`p = 0.02` is indistinguishable from the inside.

## The five numbers

| # | Measure | How | Target | Ours |
|---|---|---|---|---|
| 1 | Pass-rate distribution across shipped tasks | ~400 rollouts per task (≈±5%) | median 0.3–0.7; <10% of tasks outside 0.05–0.95 | ⬜ |
| 2 | Nondeterminism rate | replay one trajectory ×100 | exactly 0 | ⬜ |
| 3 | p99 reset latency | load test at the trainer's concurrency | ⬜ ms | ⬜ |
| 4 | Red-team success rate | an agent whose only objective is reward without the task | 0 | ⬜ |
| 5 | Verifier coverage | fraction of stated requirements backed by a gating check | 100% of what the instruction names | ⬜ |

Numbers 1 and 2 are the ones nobody measures, and both cost minutes of CPU.

## Where clone-based environments specifically fail

Cloning an application gets you the pixels. The failure is semantic, and it comes in
four shapes:

- **Fidelity is not appearance.** A clone has different bugs, different edge cases,
  different concurrency semantics. The agent learns *the clone*.
- **The transfer gap.** Reward rises in the clone; the lab's own eval does not move.
  That is not a training bug, it is an environment bug, and it is indistinguishable
  from a reward-hacking bug until someone measures both.
- **Clone-specific exploits.** The agent finds a shortcut that exists only because
  the clone is a clone.
- **Answer leakage.** The target value is sitting in the seed data. It always is.

One countermeasure covers all four: assert on world state rather than on the agent's
transcript, assert the *negatives* ("everything else untouched"), and run a red-team
agent nightly with a failing build if it ever scores 1.0 without doing the task.

## What I would not do in month one

Migrate anything. Changing an environment while a lab is mid-run makes the pass rate
uninterpretable and the regression unattributable. Measure first; the measurements
are arguments, and an argument is worth more than an opinion.

## The five questions I would ask first

1. What is the pass-rate distribution of the tasks we have already shipped?
   *(If nobody knows, that is the answer, and it is the first thing to fix.)*
2. Who owns the fidelity number for each clone, and what is it this month?
3. Is there a red-team agent in CI? What does it score today?
4. Over the last month of lab training runs, what fraction of groups had zero spread?
5. When a lab says "we trained on your environment and our eval didn't move", what is
   our first diagnostic — and how long does it take to run?

## What would change my mind

- **If pass rates are already well calibrated across the portfolio**, calibration is
  not the lever — cost is, and the memo should be about latency and throughput.
- **If fidelity is already tracked per clone and is high**, the transfer gap is
  smaller than I assume and the bottleneck is authoring throughput.
- **If red-teaming already exists and is clean**, the residual risk is verifier
  *coverage* — requirements that are latent in the task but never asserted, which no
  amount of red-teaming will find because nothing points at them.

---

*One page. If it needs two, the argument is not finished.*

---

*Why these positions and not others — the three layers of "newness", the five signs
of an immature discipline, and where the difficulty actually sits:
[`08-why-rl-is-hard.md`](08-why-rl-is-hard.md).*
