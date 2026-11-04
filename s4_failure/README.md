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

## The thing to internalise

A broken verifier does not fail loudly. Training runs, the reward curve climbs,
nobody files a bug. The failure appears three weeks later as "we trained on your
environment and our eval did not move" — and by then the cost is not your
compute, it is your customer's trust.

This is why `s1_verifier` asserts evidence on every check and `s4` runs the
hacker: both are cheap, and both fail before the customer does.
