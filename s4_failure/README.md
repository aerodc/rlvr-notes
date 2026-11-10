# S4 — Break your own verifier

```bash
python3 leaky_verifier.py
```

Three agents, three verifiers, one question: can the verifier tell an agent that
did the task from an agent that described it?

| agent | true success | v_string | v_loose | v_state |
|---|---|---|---|---|
| honest | 1.0 | 1.0 | 1.0 | 1.0 |
| echo (does nothing, restates the instruction) | 0.0 | **1.0** | 0.0 | 0.0 |
| loose (does half the task) | 0.0 | **1.0** | **1.0** | 0.0 |

Two of the three verifiers are hacked. Only the one that reads the world state
separates them.

## The three hacks

**1. Answer leakage.** The expected value is present in the instruction, the
fixture files, the test names, or the tool output. The agent echoes it. In a
cloned app this is everywhere: the seed data contains the target row.

**2. Checker shortcut.** The verifier reads the agent's reply rather than the
world. This is the most expensive mistake in the file above, because it is also
the easiest one to write, and because it produces beautifully rising curves.

**3. Loose assertion.** The task says "set the priority and assign it to alice";
the check asserts only the priority. The agent does the cheap half, the reward
says done, and the model learns to do the cheap half.

## The fix that actually works

Not a stricter regex. Three changes:

- **Assert on world state, never on the transcript.** The transcript is an
  artefact the agent controls; the world is the thing the task is about.
- **Assert the negatives too.** "Issue 2 untouched" is a requirement. Most
  verifiers only assert what should have changed, and that is precisely the gap
  an agent will find — see the `constraint` category in `s1_verifier`.
- **Run a red-team agent nightly.** Its only objective is to score 1.0 without
  doing the task. If it succeeds, the build fails. This is the only mechanism
  that keeps working after you stop looking.

## A real one, from a published training run

The matrix above is a toy. Here is the same class of failure from an actual
3B-parameter run — the [SmolLM3 training playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
([Chinese translation](https://huggingface.co/spaces/fat-pig/smol-training-playbook-zh)),
describing what happened when they applied GRPO as-is:

> 出乎我们意料，直接照搬 GRPO 引发了一种 reward hacking：尽管从没被要求输出长思维链（CoT），
> 模型学会了动用底层的 reasoning 能力去抬高奖励。reward 在涨，补全的长度也在涨。
> …用 GRPO 做 RLVR，把我们 reasoning 模型的 `/no_think` 模式变成了一个与 `/think`
> 模式非常像的模式。

Their fix was a **length penalty**, tuned to a 2.5–3k token band. In the language of
[`07-vocabulary`](../docs/07-vocabulary.md#9--regularisation), they added a
regulariser — because the reward had no way to say "be brief".

### This one has a different shape

Your matrix shows a verifier being **wrong**. This case shows a verifier being
**right** and still producing the wrong policy: correctness went up, length went up
with it, and length was never asked for. The model did not find a hole in the
checker. It found a real channel in the reward that nobody had constrained.

That gives a cleaner taxonomy than "reward hacking":

| Failure | Mechanism | Repair |
|---|---|---|
| **The checker is wrong** | string match, loose assertion, answer leakage | assert on state; assert the negatives |
| **The reward is under-specified** | a correlate of success gets optimised too | add a constraint: a penalty, a length band, a cost |
| **The reward is right and the world is wrong** | clone fidelity, transfer gap | fix the clone, not the verifier |

The middle row is the one that surprises people, because **no bug exists.** There is
nothing to fix in `verify()`. The defect is in what the reward failed to *mention*,
and the only repair is to mention it.

Which is why length penalties, KL terms and negative assertions belong to one
conversation: they are the same move — **constrain the solution space, because no
objective can say everything you mean.**

## The thing to internalise

A broken verifier does not fail loudly. Training runs, the reward curve climbs,
nobody files a bug. The failure appears three weeks later as "we trained on your
environment and our eval did not move" — and by then the cost is not your
compute, it is your customer's trust.

This is why `s1_verifier` asserts evidence on every check and `s4` runs the
hacker: both are cheap, and both fail before the customer does.
