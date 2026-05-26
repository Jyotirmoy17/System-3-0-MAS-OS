"""Integration tests for the System-3/0 demo API (mas/web/server.py)."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _make_run_log(goal_id: str = "adversarial_002",
                  condition: str = "FULL_FRAMEWORK") -> dict[str, Any]:
    now = "2026-05-02T10:00:00Z"
    return {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            goal_id,
        "goal_type":          "adversarial_reliability",
        "pipeline_depth_k":   4,
        "condition":          condition,
        "routing_skipped":    False,
        "seed":               42,
        "model_slm":          "phi3:mini",
        "model_llm":          "llama3.1:8b",
        "started_at":         now,
        "completed_at":       now,
        "latency_s":          3.5,
        "success":            1,
        "cascade_failure":    0,
        "tokens":             {"slm": 400, "llm": 1200, "total": 1600},
        "s0":                 {"catches": 1, "repairs": 1, "would_have_failed": 1},
        "escalations":        {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step": [
            {
                "step_id": 1, "agent_id": "WebAgent", "tier": 1,
                "raw_output": "...", "schema_valid": True,
                "would_have_been_invalid": False, "downstream_consumed": True,
                "tokens_used": 420, "latency_ms": 1200,
                "adversarial_corruption_applied": None,
            },
        ],
        "final_output":       '{"answer": "42"}',
        "ground_truth":       "42",
        "rubric_scores":      {"c1": True},
        "ground_truth_match": True,
        "voc_trace":          [0.8, 0.4],
        "crash_reason":       None,
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
        "backend":            "ollama",
        "api_cost_usd":       0.0,
    }


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path):
    """
    Reset server-level singletons before each test.

    Patches _runs, _rate_windows, and the DB path so tests don't share
    state and all write to a temp directory.
    """
    import mas.web.server as srv

    # Fresh in-memory state
    srv._runs.clear()
    srv._rate_windows.clear()
    srv._sse_queues.clear()

    # Redirect SQLite to a temp directory
    original_data_dir = srv._DATA_DIR
    srv._DATA_DIR = tmp_path
    srv._init_db()

    yield

    # Restore
    srv._DATA_DIR = original_data_dir
    srv._runs.clear()
    srv._rate_windows.clear()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return a TestClient wrapping the FastAPI app."""
    from mas.web.server import app
    return TestClient(app, raise_server_exceptions=False)
# Used for tests that check status / stream endpoints.

def _mock_run_worker(run_id: str, goal_id: str, condition: str) -> None:
    """Synchronous drop-in for _run_worker: sets status=done immediately."""
    import mas.web.server as srv

    run_log = _make_run_log(goal_id, condition)
    run_log["run_id"] = run_id

    # Publish SSE events
    for step in run_log.get("per_step", []):
        srv._publish_event(run_id, "step", step)
    srv._publish_event(run_id, "done", {
        "success":         run_log["success"],
        "cascade_failure": run_log["cascade_failure"],
    })

    with srv._runs_lock:
        if run_id in srv._runs:
            srv._runs[run_id]["status"]  = "done"
            srv._runs[run_id]["run_log"] = run_log


def _post_run_patched(client: TestClient, goal: str = "adversarial_002",
                      condition: str = "FULL_FRAMEWORK") -> "Response":
    """POST /api/run with _run_worker patched to a no-op synchronous stub."""
    import mas.web.server as srv
    with patch.object(srv, "_run_worker", lambda *a: None):
        return client.post("/api/run", json={"goal": goal, "condition": condition})


def test_health_endpoint(client: TestClient) -> None:
    """GET /api/health returns 200 with the expected shape."""
    with patch("mas.web.server._requests_lib.get") as mock_get:
        mock_get.return_value.status_code = 200
        resp = client.get("/api/health")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["ok"] is True
    assert "ollama" in body
    assert "version" in body
    assert isinstance(body["ollama"], bool)
    assert isinstance(body["version"], str)


def test_run_endpoint_queues_run(client: TestClient) -> None:
    """POST /api/run with a valid goal returns 202 with run_id and status=queued."""
    resp = _post_run_patched(client, "adversarial_002", "FULL_FRAMEWORK")

    assert resp.status_code == 202, f"Expected 202: {resp.text}"
    body = resp.json()
    assert "run_id" in body
    assert body["status"] == "queued"
    uuid.UUID(body["run_id"])  # raises if invalid


def test_run_endpoint_validates_condition(client: TestClient) -> None:
    """POST /api/run with an invalid condition returns 422 (Pydantic) or 400."""
    resp = client.post("/api/run", json={
        "goal": "adversarial_002",
        "condition": "INVALID_CONDITION",
    })
    assert resp.status_code in (400, 422), (
        f"Expected 400 or 422 for invalid condition, got {resp.status_code}: {resp.text}"
    )


def test_run_endpoint_validates_goal(client: TestClient) -> None:
    """POST /api/run with a goal not in the demo set returns 400."""
    resp = client.post("/api/run", json={
        "goal": "arbitrary_user_goal_NOTALLOWED",
        "condition": "FULL_FRAMEWORK",
    })
    assert resp.status_code == 400, (
        f"Expected 400 for non-demo goal, got {resp.status_code}: {resp.text}"
    )
    assert "demo set" in resp.json().get("detail", "").lower() or \
           "allowed" in resp.json().get("detail", "").lower()


def test_run_status_endpoint(client: TestClient) -> None:
    """GET /api/run/{run_id} returns RunLog once done; 404 for unknown ID."""
    import mas.web.server as srv

    resp_404 = client.get("/api/run/nonexistent-id")
    assert resp_404.status_code == 404

    run_id  = str(uuid.uuid4())
    run_log = _make_run_log()
    run_log["run_id"] = run_id
    with srv._runs_lock:
        srv._runs[run_id] = {
            "status":    "done",
            "goal_id":   "adversarial_002",
            "condition": "FULL_FRAMEWORK",
            "run_log":   run_log,
        }

    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200, f"Expected 200: {resp.text}"
    body = resp.json()
    assert body["run_id"]    == run_id
    assert body["status"]    == "done"
    assert body["run_log"]   is not None
    assert body["condition"] == "FULL_FRAMEWORK"


def test_stream_sse(client: TestClient) -> None:
    """
    GET /api/run/{run_id}/stream returns SSE-formatted text for a completed run.

    Since the run is pre-seeded as 'done', the endpoint replays per-step
    events and then sends a final 'done' event — all in a single response.
    """
    import mas.web.server as srv

    run_id  = str(uuid.uuid4())
    run_log = _make_run_log()
    run_log["run_id"] = run_id

    with srv._runs_lock:
        srv._runs[run_id] = {
            "status":    "done",
            "goal_id":   "adversarial_002",
            "condition": "FULL_FRAMEWORK",
            "run_log":   run_log,
        }

    resp = client.get(f"/api/run/{run_id}/stream")
    assert resp.status_code == 200, f"Expected 200: {resp.text}"
    assert "text/event-stream" in resp.headers.get("content-type", "")

    text = resp.text
    # Must contain SSE event markers
    assert "event:" in text, f"No SSE events in stream: {text[:200]}"
    assert "data:"  in text, f"No SSE data in stream: {text[:200]}"
    # Must end with a done event
    assert "event: done" in text or "event: error" in text, (
        f"Stream did not terminate with done/error: {text[-300:]}"
    )


def test_waitlist_stores_email(client: TestClient, tmp_path: Path) -> None:
    """POST /api/waitlist stores the email in SQLite."""
    import mas.web.server as srv

    resp = client.post("/api/waitlist", json={
        "email":    "test@example.com",
        "use_case": "thesis demo",
    })
    assert resp.status_code == 200, f"Expected 200: {resp.text}"
    assert resp.json()["ok"] is True

    # Verify row exists in DB
    count = srv._count_waitlist("test@example.com")
    assert count == 1, f"Expected 1 row in waitlist, got {count}"


def test_waitlist_duplicate_idempotent(client: TestClient) -> None:
    """
    Submitting the same email twice does not create duplicate rows.

    INSERT OR IGNORE semantics: second call is a no-op.
    """
    import mas.web.server as srv

    payload = {"email": "dup@example.com", "use_case": "test"}

    resp1 = client.post("/api/waitlist", json=payload)
    resp2 = client.post("/api/waitlist", json=payload)

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    count = srv._count_waitlist("dup@example.com")
    assert count == 1, f"Expected 1 row (idempotent), got {count}"


def test_rate_limit_429(client: TestClient) -> None:
    """
    The 11th run request from the same IP within one hour is rejected with 429.

    Patches _run_worker to a no-op so only rate limiting is exercised.
    """
    import mas.web.server as srv

    responses = [
        _post_run_patched(client, "adversarial_002", "FULL_FRAMEWORK")
        for _ in range(11)
    ]

    status_codes = [r.status_code for r in responses]
    assert all(s == 202 for s in status_codes[:10]), (
        f"Expected first 10 to be 202, got: {status_codes[:10]}"
    )
    assert status_codes[10] == 429, (
        f"Expected 11th request to be 429, got: {status_codes[10]}"
    )
    assert "rate limit" in responses[10].json().get("detail", "").lower()


def test_health_endpoint_ollama_down(client: TestClient) -> None:
    """GET /api/health reports ollama=False when the Ollama server is unreachable."""
    with patch("mas.web.server._requests_lib.get", side_effect=Exception("connection refused")):
        resp = client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"]     is True
    assert body["ollama"] is False


def test_run_status_queued_state(client: TestClient) -> None:
    """GET /api/run/{run_id} returns status=queued and null run_log mid-execution."""
    import mas.web.server as srv

    run_id = str(uuid.uuid4())
    with srv._runs_lock:
        srv._runs[run_id] = {
            "status":    "queued",
            "goal_id":   "adversarial_002",
            "condition": "FULL_FRAMEWORK",
            "run_log":   None,
        }

    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"]  == "queued"
    assert body["run_log"] is None


def test_demo_goals_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    DEMO_GOAL_IDS env var overrides the built-in allow-list.
    The function _load_demo_goals() returns exactly those IDs.
    """
    monkeypatch.setenv("DEMO_GOAL_IDS", "my_goal_1, my_goal_2,, my_goal_3")
    import importlib
    import mas.web.server as srv

    result = srv._load_demo_goals()
    assert result == frozenset({"my_goal_1", "my_goal_2", "my_goal_3"})


def test_rate_window_expiry(client: TestClient) -> None:
    """
    Timestamps older than 1 hour are evicted from the rate window.

    Seeds the rate window with 10 stale timestamps (>1h old), then sends
    a fresh request.  The request should succeed (202) because the stale
    entries are evicted.
    """
    import collections
    import mas.web.server as srv

    stale_ts = time.time() - 7200  # 2 hours ago
    test_ip  = "testclient"
    with srv._rate_lock:
        srv._rate_windows[test_ip] = collections.deque([stale_ts] * 10)

    resp = _post_run_patched(client, "adversarial_002", "FULL_FRAMEWORK")
    assert resp.status_code == 202, (
        f"Expected 202 after stale-entry eviction, got {resp.status_code}"
    )


def test_sandbox_create_and_cleanup(tmp_path: Path) -> None:
    """_make_sandbox creates a directory; _cleanup_sandbox removes it."""
    import mas.web.server as srv

    old_runs_dir = srv._RUNS_DIR
    srv._RUNS_DIR = tmp_path / "runs"
    srv._RUNS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        run_id  = str(uuid.uuid4())
        sandbox = srv._make_sandbox(run_id)

        assert sandbox.is_dir(), "Sandbox directory was not created"
        assert sandbox.name == run_id

        srv._cleanup_sandbox(run_id)
        assert not sandbox.exists(), "Sandbox directory was not removed"
    finally:
        srv._RUNS_DIR = old_runs_dir


def test_waitlist_invalid_email(client: TestClient) -> None:
    """POST /api/waitlist with an invalid email returns 422."""
    resp = client.post("/api/waitlist", json={"email": "not-an-email"})
    assert resp.status_code == 422, (
        f"Expected 422 for invalid email, got {resp.status_code}: {resp.text}"
    )
