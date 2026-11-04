#!/usr/bin/env python3
"""
Client for the environment service — and its test suite.

By default it starts the Node server itself, runs the checks, and shuts it down,
so `make env-test` is one command. Point it at an existing server with --url.

    python3 client.py --selftest          # correctness + determinism + idempotence
    python3 client.py --load --clients 20  # reset latency under concurrency

The selftest asserts the invariants in SPEC.md that are easy to claim and
expensive to skip:

    1. reset is byte-identical for the same task
    2. doing nothing scores 0
    3. doing the task scores 1
    4. score is idempotent
    5. an invalid action errors but does not break the run
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SERVER = pathlib.Path(__file__).parent / "server.mjs"
DEFAULT_URL = "http://127.0.0.1:8787"


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class EnvClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str):
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def health(self) -> dict:
        return self._get("/health")

    def tasks(self) -> list[dict]:
        return self._get("/tasks")

    def reset(self, task_id: str) -> dict:
        return self._post("/reset", {"task_id": task_id})

    def step(self, session_id: str, action: dict) -> dict:
        return self._post("/step", {"session_id": session_id, "action": action})

    def score(self, session_id: str) -> dict:
        return self._post("/score", {"session_id": session_id})

    def close(self, session_id: str) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/session",
            data=json.dumps({"session_id": session_id}).encode(),
            headers={"content-type": "application/json"},
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())


# --------------------------------------------------------------------------- #
# Scripted agent: a fixed action sequence, so the test is about the env
# --------------------------------------------------------------------------- #


def solve_triage(client: EnvClient) -> dict:
    ep = client.reset("triage-001")
    sid = ep["session_id"]
    pre = ep["observation"]
    after = client.step(sid, {"tool": "update_issue", "id": 1, "fields": {"priority": "high"}})
    assert not after.get("error"), after
    after = client.step(sid, {"tool": "update_issue", "id": 1, "fields": {"assignee": "alice"}})
    assert not after.get("error"), after
    return {"session_id": sid, "start": pre, "end": after["observation"]}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[k]


def selftest(client: EnvClient) -> dict:
    results: dict[str, object] = {}
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results[name] = "pass" if condition else "FAIL"
        if not condition:
            failures.append(f"{name}: {detail}")

    # 1. reset is byte-identical for the same task
    a = client.reset("triage-001")
    b = client.reset("triage-001")
    same = json.dumps(a["observation"], sort_keys=True) == json.dumps(b["observation"], sort_keys=True)
    results["reset_is_identical"] = same
    results["reset_nondeterminism"] = 0.0 if same else 1.0
    if not same:
        failures.append("reset_is_identical: two resets produced different worlds")

    # 2. doing nothing scores zero
    idle = client.reset("triage-001")
    idle_score = client.score(idle["session_id"])
    results["idle_scores_zero"] = idle_score["reward"] == 0.0
    if idle_score["reward"] != 0.0:
        failures.append(f"idle_scores_zero: got {idle_score['reward']}")

    # 3. doing the task scores one
    solved = solve_triage(client)
    s1 = client.score(solved["session_id"])
    results["solved_scores_one"] = s1["reward"] == 1.0
    if s1["reward"] != 1.0:
        failures.append(f"solved_scores_one: got {s1['reward']}; evidence={s1['evidence']}")

    # 4. score is idempotent
    scores = [client.score(solved["session_id"])["reward"] for _ in range(20)]
    idem = len(set(scores)) == 1
    results["score_is_idempotent"] = idem
    if not idem:
        failures.append(f"score_is_idempotent: {sorted(set(scores))}")

    # 5. invalid action errors, does not crash, does not change the world
    ep = client.reset("triage-001")
    before = json.dumps(ep["observation"], sort_keys=True)
    bad = client.step(ep["session_id"], {"tool": "drop_database"})
    after_world = json.dumps(bad["observation"], sort_keys=True)
    results["invalid_action_is_reported"] = bool(bad.get("error"))
    results["invalid_action_is_harmless"] = before == after_world
    if not bad.get("error"):
        failures.append("invalid_action_is_reported: no error field")
    if before != after_world:
        failures.append("invalid_action_is_harmless: world changed after an invalid action")

    # 6. evidence on every check
    complete = all(e.get("requirement") and "expected" in e for e in s1["evidence"])
    results["evidence_is_complete"] = complete
    if not complete:
        failures.append("evidence_is_complete: an evidence entry was missing fields")

    # 7. sessions are isolated
    x = client.reset("triage-001")
    y = client.reset("triage-001")
    client.step(x["session_id"], {"tool": "update_issue", "id": 1, "fields": {"priority": "high"}})
    y_view = client.score(y["session_id"])
    isolated = y_view["reward"] == 0.0
    results["sessions_are_isolated"] = isolated
    if not isolated:
        failures.append("sessions_are_isolated: a step in one session affected another")

    results["failures"] = failures
    return results


def load_test(client: EnvClient, clients: int, repeats: int) -> dict:
    """Measure reset latency, including under concurrency. This is the COGS number."""

    def one(_):
        t0 = time.perf_counter()
        ep = client.reset("triage-001")
        dt = (time.perf_counter() - t0) * 1000
        client.close(ep["session_id"])
        return dt

    # Warm up so the first request does not pay for connection setup.
    one(0)

    serial = [one(i) for i in range(repeats)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        concurrent = list(pool.map(one, range(repeats)))
    wall = time.perf_counter() - t0

    return {
        "clients": clients,
        "requests": repeats,
        "serial_p50_ms": round(statistics.median(serial), 2),
        "serial_p99_ms": round(percentile(serial, 99), 2),
        "concurrent_p50_ms": round(statistics.median(concurrent), 2),
        "concurrent_p95_ms": round(percentile(concurrent, 95), 2),
        "concurrent_p99_ms": round(percentile(concurrent, 99), 2),
        "throughput_rps": round(repeats / wall, 1) if wall > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def wait_for(url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=0.5).read()
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.1)
    raise SystemExit(f"server did not start at {url}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default=None, help="use an existing server instead of spawning one")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--load", action="store_true")
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args()

    if not args.selftest and not args.load:
        args.selftest = True

    proc = None
    url = args.url
    if url is None:
        url = f"http://127.0.0.1:{args.port}"
        proc = subprocess.Popen(
            ["node", str(SERVER), "--port", str(args.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    try:
        wait_for(url)
        client = EnvClient(url)

        if args.selftest:
            results = selftest(client)
            print(json.dumps(results, indent=2))
            if results["failures"]:
                print(f"\n{len(results['failures'])} FAILURE(S)", file=sys.stderr)
                sys.exit(1)
            print("\nall invariants hold")

        if args.load:
            print(json.dumps(load_test(client, args.clients, args.requests), indent=2))
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()
