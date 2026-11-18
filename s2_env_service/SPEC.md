# Environment wire contract

**v0 is implemented in `server.mjs`. v1 is specified here and not implemented** —
see [Lifecycle primitives](#lifecycle-primitives--v1).

An environment is a **service**, not a library. That one decision is what lets
one team build environments and another team train on them without either
reading the other's code, and it is what makes the interface testable.

v0 is the smallest contract that supports a short RLVR rollout. It is not the
smallest contract that supports a real training run — that one needs lifecycle
primitives, because once a trajectory runs to hundreds of tool calls, **the
sandbox's state is training state.** See
[`docs/10-agentenv-lessons.md`](../docs/10-agentenv-lessons.md).

---

## Endpoints

### `GET /health`

```json
{ "ok": true, "sessions": 3 }
```

### `GET /tasks`

```json
[
  { "task_id": "triage-001", "instruction": "Set issue #1 to priority high..." }
]
```

### `POST /reset`

Request:

```json
{ "task_id": "triage-001" }
```

Response:

```json
{
  "session_id": "9f2c...",
  "task_id": "triage-001",
  "observation": { "issues": [...], "comments": [...] },
  "steps_remaining": 12
}
```

`reset` must return a world that is **identical every time** for the same
`task_id`. Not "equivalent" — identical. The session id is allowed to differ;
the world is not.

### `POST /step`

Request:

```json
{ "session_id": "9f2c...", "action": { "tool": "update_issue", "id": 1, "fields": { "priority": "high" } } }
```

Response:

```json
{ "observation": {...}, "done": false, "steps_remaining": 11 }
```

An action that cannot be applied returns HTTP 200 with `"error"` populated and
the world unchanged. **Invalid actions are part of the task, not an exception
path** — an agent will try them, and the run must continue.

### `POST /score`

Request: `{ "session_id": "9f2c..." }`

Response:

```json
{
  "task_id": "triage-001",
  "reward": 1.0,
  "subscores": { "core": 1.0, "constraint": 1.0 },
  "evidence": [
    { "requirement": "issue_1_priority_is_high", "passed": true,
      "expected": "high", "actual": "high", "gating": true }
  ],
  "verifier_version": "0.1.0",
  "spec_hash": "44157adf8258"
}
```

Scoring is **idempotent**: calling `/score` twice on the same session returns
the same reward. A score that changes on re-read is a score you cannot train on.

---

## Lifecycle primitives — v1

Not implemented here. Specified because a training run cannot live without them,
and because getting them wrong is invisible until you are already at scale.

| Endpoint | Request | Response |
|---|---|---|
| `POST /pause` | `{session_id}` | `{snapshot_id, memory_bytes, disk_bytes}` |
| `POST /resume` | `{snapshot_id}` | `{session_id, observation}` |
| `POST /snapshot` | `{session_id}` | `{snapshot_id, size_bytes, incremental: true}` |
| `POST /fork` | `{snapshot_id, n}` | `[{session_id}, ...]` |

Why each one exists, in the order you will need them:

- **`pause` / `resume`** — the sandbox must release CPU and memory while the policy
  generator catches up. A long trajectory is generated in segments, and the world
  in between is not reconstructible.
- **`snapshot`** — a checkpoint for a trajectory that cannot be rebuilt. It must be
  **incremental** (dirty pages and filesystem deltas), not a full copy: full copies
  cannot be taken at the frequency RL needs.
- **`fork`** — branch from a byte-identical world. Two uses that matter: scoring a
  frozen state without perturbing it (*side-effect-free judging*), and running a
  red-team agent against a real state without contaminating the original.

## Metrics that matter

If you can instrument four numbers, instrument these. `reset` alone is the v0
metric, and it is the wrong one for long-horizon environments.

| Metric | What it drives | Bad looks like |
|---|---|---|
| p50 / p99 `reset` | cost of starting a rollout | seconds |
| **p50 / p99 `pause` + `resume`, per tool call** | **cost of a long trajectory — the real denominator** | tens of ms × hundreds of calls |
| p50 / p99 `snapshot` size, write throughput | how often you can afford a checkpoint | GBs per snapshot |
| `fork` latency | how cheaply you can re-score or red-team | seconds |

The arithmetic: 300 tool calls × 40 ms of pause/resume is **12 seconds of overhead
per rollout**, before the model emits a token.

---

## Invariants

| Invariant | Why |
|---|---|
| `reset` is a snapshot, not a replay | Replaying a seed drifts. A drifted reset is a silently different task. |
| Same task → same initial world, byte for byte | Otherwise pass rate is not comparable across rollouts. |
| `score` is a pure function of world state | No clock, no randomness, no network. |
| Every check emits evidence | A 0 you cannot explain costs an hour per task. |
| Sessions are isolated | One rollout must not be able to see or affect another. |
| `score` is idempotent | Re-scoring is part of every debugging session. |

These extend to the v1 lifecycle primitives:

| Invariant | Why |
|---|---|
| `pause` → `resume` round-trips to a byte-identical world | Otherwise the resumed trajectory is a different task |
| `fork` yields N worlds byte-identical to the parent at fork time | Judging and exploration both assume no divergence |
| `snapshot` is incremental | Full copies cap how often you can checkpoint |
| A resumed session keeps the same session semantics | `steps_remaining`, isolation, idempotent `score` |

## What is deliberately missing

- **The v1 lifecycle primitives.** Specified above, not implemented. The toy world
  is a JSON object in memory; pausing it is meaningless until the world is a
  process with a filesystem.
- **No authentication.** This is a lab, not a deployment.
- **No persistence.** Restarting the server drops all sessions. Fine for a lab,
  unacceptable for a trainer mid-run — which is exactly the kind of requirement
  you only discover when you try to use it.
- **No sandboxing.** The actions are validated data, not code. The moment an
  environment runs agent-authored code, isolation stops being a design choice —
  a container is not a boundary, and the incident that taught this is in
  [`docs/10-agentenv-lessons.md`](../docs/10-agentenv-lessons.md).
