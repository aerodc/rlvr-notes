# Environment wire contract — v0

An environment is a **service**, not a library. That one decision is what lets
one team build environments and another team train on them without either
reading the other's code, and it is what makes the interface testable.

This is the smallest contract that supports RLVR rollouts.

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

## Invariants

| Invariant | Why |
|---|---|
| `reset` is a snapshot, not a replay | Replaying a seed drifts. A drifted reset is a silently different task. |
| Same task → same initial world, byte for byte | Otherwise pass rate is not comparable across rollouts. |
| `score` is a pure function of world state | No clock, no randomness, no network. |
| Every check emits evidence | A 0 you cannot explain costs an hour per task. |
| Sessions are isolated | One rollout must not be able to see or affect another. |
| `score` is idempotent | Re-scoring is part of every debugging session. |

## What is deliberately missing

- **No authentication.** This is a lab, not a deployment.
- **No persistence.** Restarting the server drops all sessions. Fine for a lab,
  unacceptable for a trainer mid-run — which is exactly the kind of requirement
  you only discover when you try to use it.
- **No sandboxing.** The actions are validated data, not code. The moment an
  environment runs agent-authored code, isolation stops being a design choice.
