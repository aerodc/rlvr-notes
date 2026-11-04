# S2 — An environment is a service

```bash
make env        # terminal 1: node server.mjs
make env-test   # terminal 2: runs the invariants, spawns its own server if needed
make env-load   # reset latency under 20 concurrent clients
```

## Why a service and not a library

A library means the training team imports your code. That couples their Python
version to your TypeScript, their release cycle to yours, and makes every
environment bug a shared debugging session over a language boundary.

An HTTP contract means an environment can be written by one team, run by another,
and consumed by a lab that will never read either codebase. It also means the
environment can be benchmarked, load-tested, and replaced independently.

See [`SPEC.md`](SPEC.md) for the four endpoints and the invariants.

## What the invariants are worth

| Invariant | Tested by | What breaks without it |
|---|---|---|
| `reset` is byte-identical per task | `reset_is_identical` | pass rates are not comparable across rollouts |
| `score` is idempotent | `score_is_idempotent` | you cannot debug a score you cannot re-read |
| invalid actions error but do not crash | `invalid_action_is_reported` / `_harmless` | one bad tool call kills a rollout at step 9 |
| sessions are isolated | `sessions_are_isolated` | one rollout contaminates another, and you would never see it |
| every check emits evidence | `evidence_is_complete` | a 0 costs an hour to explain |

## The number to take away

`make env-load` prints p50/p95/p99 for `reset`. On a laptop, unoptimised, it
should land in the low single-digit milliseconds — because the world is a JSON
object in memory. The interesting part is not the number, it is what happens when
the world is a container with a database and a browser.

That is the exercise worth doing: replace `makeWorld()` with something that has
to spin up a Postgres and a headless Chrome, and watch p99 go from 3ms to 2s.
At 64 prompts × 8 rollouts per gradient step, that difference is the margin.

## Ways to break it on purpose

- Make `reset` replay a seed instead of copying a snapshot. Run 1,000 resets and
  diff the worlds. Find the drift.
- Make `score` depend on `Date.now()` for a "recent activity" check. Watch
  idempotence die.
- Drop the session isolation by storing the world on the task object instead of
  the session. Watch `sessions_are_isolated` fail — and then imagine shipping it.
