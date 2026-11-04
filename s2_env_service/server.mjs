#!/usr/bin/env node
/**
 * A minimal RL environment, served over HTTP. Node standard library only.
 *
 * The point of this file is the shape, not the domain:
 *
 *   reset  -> a world, byte-identical every time for a given task
 *   step   -> apply one action, return the new observation
 *   score  -> a pure function of world state, with evidence
 *
 * Sessions are isolated and held in memory. Restarting the process drops them,
 * which is fine for a lab and unacceptable for a trainer mid-run.
 *
 *   node server.mjs [--port 8787]
 */

import { createServer } from "node:http";
import { randomUUID } from "node:crypto";

const MAX_STEPS = 12;
const VERIFIER_VERSION = "0.1.0";

// --------------------------------------------------------------------------
// Worlds. A factory, not a shared object — reset must hand out a fresh copy.
// --------------------------------------------------------------------------

const TASKS = {
  "triage-001": {
    instruction: "Set issue #1 to priority high and assign it to alice. Do not change any other issue.",
    makeWorld: () => ({
      issues: [
        { id: 1, title: "Fix login redirect", priority: "medium", assignee: null, status: "open" },
        { id: 2, title: "Tune search latency", priority: "medium", assignee: "bob", status: "open" },
        { id: 3, title: "Deprecate legacy API", priority: "low", assignee: null, status: "open" },
      ],
      comments: [],
    }),
  },
  "comment-002": {
    instruction: "Add a comment to issue #1 explaining what you found.",
    makeWorld: () => ({
      issues: [
        { id: 1, title: "Fix login redirect", priority: "medium", assignee: null, status: "open" },
      ],
      comments: [],
    }),
  },
  "audit-003": {
    instruction: "Mark every high-priority issue as needing review.",
    makeWorld: () => ({
      issues: [
        { id: 1, title: "Fix login redirect", priority: "high", assignee: null, status: "open" },
        { id: 2, title: "Tune search latency", priority: "medium", assignee: "bob", status: "open" },
        { id: 3, title: "Rotate credentials", priority: "high", assignee: null, status: "open" },
      ],
      comments: [],
    }),
  },
};

// --------------------------------------------------------------------------
// Actions. Validated data, never executed code.
// --------------------------------------------------------------------------

function applyAction(world, action) {
  if (!action || typeof action !== "object") {
    return { error: "action must be an object" };
  }
  const issue = (id) => world.issues.find((i) => i.id === id);

  switch (action.tool) {
    case "update_issue": {
      const target = issue(action.id);
      if (!target) return { error: `no issue with id ${action.id}` };
      const allowed = ["priority", "assignee", "status", "title"];
      const fields = action.fields ?? {};
      const unknown = Object.keys(fields).filter((k) => !allowed.includes(k));
      if (unknown.length) return { error: `unknown fields: ${unknown.join(", ")}` };
      Object.assign(target, fields);
      return { ok: true };
    }
    case "add_comment": {
      if (!issue(action.issue_id)) return { error: `no issue with id ${action.issue_id}` };
      if (typeof action.body !== "string" || !action.body.trim()) {
        return { error: "body must be a non-empty string" };
      }
      world.comments.push({ issue_id: action.issue_id, body: action.body });
      return { ok: true };
    }
    case "finish":
      return { ok: true };
    default:
      return { error: `unknown tool: ${action.tool}` };
  }
}

// --------------------------------------------------------------------------
// The verifier. Pure function of world state.
// --------------------------------------------------------------------------

function check(requirement, passed, expected, actual, extra = {}) {
  return { requirement, passed, expected, actual, ...extra };
}

const VERIFIERS = {
  "triage-001"(w) {
    const i1 = w.issues.find((i) => i.id === 1) ?? {};
    const i2 = w.issues.find((i) => i.id === 2) ?? {};
    const i3 = w.issues.find((i) => i.id === 3) ?? {};
    return [
      check("issue_1_priority_is_high", i1.priority === "high", "high", i1.priority, { gating: true, category: "core" }),
      check("issue_1_assigned_to_alice", i1.assignee === "alice", "alice", i1.assignee, { gating: true, category: "core" }),
      check("issue_2_untouched", i2.priority === "medium", "medium", i2.priority, { category: "constraint" }),
      check("issue_3_untouched", i3.priority === "low", "low", i3.priority, { category: "constraint" }),
    ];
  },
  "comment-002"(w) {
    const c = w.comments[0];
    return [
      check("a_comment_exists", w.comments.length === 1, 1, w.comments.length, { gating: true, category: "core" }),
      check("comment_is_on_issue_1", c?.issue_id === 1, 1, c?.issue_id ?? null, { gating: true, category: "core" }),
      check("comment_is_not_empty", typeof c?.body === "string" && c.body.trim().length >= 5, ">=5 chars", c?.body ?? null, { category: "quality" }),
    ];
  },
  "audit-003"(w) {
    const highs = w.issues.filter((i) => i.priority === "high");
    const others = w.issues.filter((i) => i.priority !== "high");
    return [
      check(
        "all_high_priority_flagged",
        highs.length > 0 && highs.every((i) => i.needs_review === true),
        true,
        highs.map((i) => ({ id: i.id, needs_review: i.needs_review ?? null })),
        { gating: true, category: "core" },
      ),
      check(
        "low_priority_not_flagged",
        others.every((i) => i.needs_review === undefined),
        true,
        others.map((i) => ({ id: i.id, needs_review: i.needs_review ?? null })),
        { category: "constraint" },
      ),
    ];
  },
};

function specHash(taskId) {
  // Stable within a process; good enough to attribute a score to a spec version.
  const blob = JSON.stringify(TASKS[taskId].makeWorld());
  let h = 0n;
  for (const byte of Buffer.from(blob)) h = (h * 31n + BigInt(byte)) % 0xffffffffffffn;
  return h.toString(16).padStart(12, "0").slice(0, 12);
}

function scoreWorld(taskId, world) {
  const evidence = VERIFIERS[taskId](world);
  const total = evidence.length;
  const earned = evidence.filter((e) => e.passed).length;
  const gatingFailed = evidence.some((e) => e.gating && !e.passed);

  const subscores = {};
  for (const e of evidence) {
    const cat = e.category ?? "core";
    subscores[cat] ??= { earned: 0, total: 0 };
    subscores[cat].total += 1;
    if (e.passed) subscores[cat].earned += 1;
  }
  const flat = Object.fromEntries(
    Object.entries(subscores).map(([k, v]) => [k, Number((v.earned / v.total).toFixed(4))]),
  );

  return {
    task_id: taskId,
    reward: gatingFailed ? 0 : Number((earned / total).toFixed(4)),
    subscores: flat,
    evidence,
    verifier_version: VERIFIER_VERSION,
    spec_hash: specHash(taskId),
  };
}

// --------------------------------------------------------------------------
// Session store
// --------------------------------------------------------------------------

const sessions = new Map();

function createSession(taskId) {
  const task = TASKS[taskId];
  if (!task) return null;
  const sessionId = randomUUID();
  const session = {
    sessionId,
    taskId,
    world: task.makeWorld(), // fresh copy every reset
    steps: 0,
    done: false,
  };
  sessions.set(sessionId, session);
  return session;
}

// --------------------------------------------------------------------------
// HTTP plumbing
// --------------------------------------------------------------------------

function send(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
      if (raw.length > 1_000_000) reject(new Error("body too large"));
    });
    req.on("end", () => {
      if (!raw) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch {
        reject(new Error("invalid JSON body"));
      }
    });
    req.on("error", reject);
  });
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");

    if (req.method === "GET" && url.pathname === "/health") {
      return send(res, 200, { ok: true, sessions: sessions.size });
    }

    if (req.method === "GET" && url.pathname === "/tasks") {
      return send(
        res,
        200,
        Object.entries(TASKS).map(([task_id, t]) => ({ task_id, instruction: t.instruction })),
      );
    }

    if (req.method === "POST" && url.pathname === "/reset") {
      const body = await readBody(req);
      const session = createSession(body.task_id);
      if (!session) return send(res, 400, { error: `unknown task_id: ${body.task_id}` });
      return send(res, 200, {
        session_id: session.sessionId,
        task_id: session.taskId,
        observation: session.world,
        steps_remaining: MAX_STEPS,
      });
    }

    if (req.method === "POST" && url.pathname === "/step") {
      const body = await readBody(req);
      const session = sessions.get(body.session_id);
      if (!session) return send(res, 404, { error: "unknown session_id" });

      const result = applyAction(session.world, body.action);
      const errored = Boolean(result.error);
      if (!errored) session.steps += 1;

      const done = !errored && (body.action?.tool === "finish" || session.steps >= MAX_STEPS);
      session.done = done;

      return send(res, 200, {
        observation: session.world,
        done,
        steps_remaining: MAX_STEPS - session.steps,
        ...(errored ? { error: result.error } : {}),
      });
    }

    if (req.method === "POST" && url.pathname === "/score") {
      const body = await readBody(req);
      const session = sessions.get(body.session_id);
      if (!session) return send(res, 404, { error: "unknown session_id" });
      return send(res, 200, scoreWorld(session.taskId, session.world));
    }

    if (req.method === "DELETE" && url.pathname === "/session") {
      const body = await readBody(req);
      const existed = sessions.delete(body.session_id);
      return send(res, 200, { ok: existed });
    }

    return send(res, 404, { error: "not found" });
  } catch (err) {
    return send(res, 400, { error: String(err.message ?? err) });
  }
});

const portArg = process.argv.indexOf("--port");
const port = portArg > -1 ? Number(process.argv[portArg + 1]) : 8787;

server.listen(port, () => {
  console.log(`env service listening on http://127.0.0.1:${port}`);
  console.log(`tasks: ${Object.keys(TASKS).join(", ")}`);
});
