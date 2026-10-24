"""FastAPI application exposing the System-3/0 framework over HTTP."""
from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests as _requests_lib  # for Ollama health check
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from mas.web.schemas import (
    HealthResponse,
    RunQueued,
    RunRequest,
    RunStatusResponse,
    WaitlistRequest,
    WaitlistResponse,
)

log = logging.getLogger(__name__)


_CORS_ORIGIN  = os.environ.get("CORS_ORIGIN", "*")
_OLLAMA_URL   = os.environ.get("OLLAMA_URL",  "http://localhost:11434")
_DATA_DIR     = Path(os.environ.get("DATA_DIR", "./data"))
_RUNS_DIR     = Path(os.environ.get("RUNS_DIR", "/tmp/mas-runs"))
_RATE_LIMIT   = int(os.environ.get("RATE_LIMIT", "10"))
_API_VERSION  = os.environ.get("API_VERSION", "0.1.0")
_RUN_TTL_S    = 3600   # sandbox cleanup after 1 hour


_BUILTIN_DEMO_GOALS: frozenset[str] = frozenset({
    "adversarial_002",
    "adversarial_006",
    "code_001",
    "code_005",
    "financial_analysis_001",
    "financial_analysis_004",
    "research_synth_001",
    "research_synth_002",
    "sequential_001",
    "sequential_004",
})


def _load_demo_goals() -> frozenset[str]:
    """
    Return the allowed demo goal IDs.

    Reads DEMO_GOAL_IDS env var (comma-separated) if set; otherwise falls
    back to the built-in 10-goal allow-list.
    """
    env = os.environ.get("DEMO_GOAL_IDS", "")
    if env.strip():
        return frozenset(g.strip() for g in env.split(",") if g.strip())
    return _BUILTIN_DEMO_GOALS


DEMO_GOALS = _load_demo_goals()


# run_id → {"status": str, "goal_id": str, "condition": str,
#            "run_log": dict|None, "events": list[str]}
_runs: dict[str, dict[str, Any]] = {}
_runs_lock = threading.Lock()

# SSE event queues: run_id → list of asyncio.Queue objects (one per listener)
_sse_queues: dict[str, list[asyncio.Queue]] = {}
_sse_lock = threading.Lock()

# Rate limiting: IP → deque of epoch timestamps (seconds)
_rate_windows: dict[str, collections.deque] = {}
_rate_lock = threading.Lock()

# Background thread pool (single worker: Ollama is the bottleneck)
_executor = ThreadPoolExecutor(max_workers=2)


def _init_db() -> None:
    """Initialise the SQLite waitlist database (idempotent)."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = _DATA_DIR / "waitlist.db"
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS waitlist (
                email      TEXT PRIMARY KEY,
                use_case   TEXT,
                created_at REAL NOT NULL
            )
        """)
        con.commit()
    finally:
        con.close()


def _db_path() -> str:
    return str(_DATA_DIR / "waitlist.db")


def _upsert_waitlist(email: str, use_case: str | None) -> None:
    """Insert or ignore a waitlist entry.

Uses INSERT OR IGNORE so duplicate emails are silently dropped."""
    con = sqlite3.connect(_db_path())
    try:
        con.execute(
            "INSERT OR IGNORE INTO waitlist (email, use_case, created_at) "
            "VALUES (?, ?, ?)",
            (email, use_case, time.time()),
        )
        con.commit()
    finally:
        con.close()


def _count_waitlist(email: str) -> int:
    """Return the number of rows with *email* in the waitlist table."""
    con = sqlite3.connect(_db_path())
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM waitlist WHERE email = ?", (email,)
        ).fetchone()
        return row[0] if row else 0
    finally:
        con.close()


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if the request is allowed; False if the IP is over-limit.

Maintains a sliding 1-hour window per IP address."""
    now = time.time()
    window_start = now - 3600.0

    with _rate_lock:
        dq = _rate_windows.setdefault(client_ip, collections.deque())
        # Evict timestamps older than 1 hour
        while dq and dq[0] < window_start:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT:
            return False
        dq.append(now)
        return True


def _make_sandbox(run_id: str) -> Path:
    """Create a private working directory for a run."""
    sandbox = _RUNS_DIR / run_id
    sandbox.mkdir(parents=True, exist_ok=True)
    return sandbox


def _cleanup_sandbox(run_id: str) -> None:
    """Remove the sandbox directory for *run_id* (best-effort)."""
    sandbox = _RUNS_DIR / run_id
    try:
        if sandbox.exists():
            shutil.rmtree(sandbox, ignore_errors=True)
    except Exception as exc:
        log.warning("Sandbox cleanup failed for %s: %s", run_id, exc)


def _publish_event(run_id: str, event_type: str, data: Any) -> None:
    """Push a server-sent event to all active SSE listeners for *run_id*.

Called from the background executor thread; uses thread-safe queue puts."""
    payload = json.dumps(data)
    with _sse_lock:
        queues = list(_sse_queues.get(run_id, []))
    for q in queues:
        try:
            q.put_nowait({"event": event_type, "data": payload})
        except asyncio.QueueFull:
            log.warning("SSE queue full for run %s — dropping event", run_id)


async def _sse_generator(run_id: str, loop: AbstractEventLoop) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted strings for one run.

Registers a queue with _sse_queues, then yields events as they arrive.
Sends a "done" sentinel to close the stream."""
    q: asyncio.Queue = asyncio.Queue(maxsize=256)

    with _sse_lock:
        _sse_queues.setdefault(run_id, []).append(q)

    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a keepalive comment
                yield ": keepalive\n\n"
                continue

            event_type = item["event"]
            data       = item["data"]
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type in ("done", "error"):
                break
    finally:
        with _sse_lock:
            listeners = _sse_queues.get(run_id, [])
            if q in listeners:
                listeners.remove(q)
            if not listeners:
                _sse_queues.pop(run_id, None)


def _run_worker(run_id: str, goal_id: str, condition: str) -> None:
    """Execute one run in a background thread.

Loads the goal via goal_loader, calls run_one(), and publishes per-step
SSE events.  Updates _runs with the final RunLog."""
    sandbox = _make_sandbox(run_id)

    # Mark running
    with _runs_lock:
        if run_id in _runs:
            _runs[run_id]["status"] = "running"

    run_log: dict[str, Any] | None = None
    try:
        # Import inside thread to avoid top-level Ollama import at startup
        from mas.eval.harness import run_one   # noqa: PLC0415

        overrides: dict[str, Any] = {
            "slm":        "phi3:mini",
            "llm":        "llama3.1:8b",
            "ollama_url": _OLLAMA_URL,
            "tau":        0.72,
        }

        run_log = run_one(
            goal_id         = goal_id,
            condition       = condition,
            seed            = 42,
            model_overrides = overrides,
            _skip_warmup    = True,
        )

        # Publish per-step SSE events
        for step in run_log.get("per_step", []):
            _publish_event(run_id, "step", {
                "step_id":               step.get("step_id"),
                "agent_id":              step.get("agent_id"),
                "schema_valid":          step.get("schema_valid"),
                "would_have_been_invalid": step.get("would_have_been_invalid"),
                "tokens_used":           step.get("tokens_used"),
            })

        _publish_event(run_id, "done", {
            "success":         run_log.get("success"),
            "cascade_failure": run_log.get("cascade_failure"),
            "tokens":          run_log.get("tokens"),
            "latency_s":       run_log.get("latency_s"),
        })

        with _runs_lock:
            _runs[run_id]["status"]  = "done"
            _runs[run_id]["run_log"] = run_log

    except Exception as exc:
        log.exception("Run worker failed for run_id=%s goal=%s", run_id, goal_id)
        err_msg = f"{type(exc).__name__}: {exc}"
        _publish_event(run_id, "error", {"message": err_msg})
        with _runs_lock:
            _runs[run_id]["status"]    = "error"
            _runs[run_id]["run_log"]   = run_log
            _runs[run_id]["error_msg"] = err_msg
    finally:
        # Schedule sandbox cleanup after TTL
        def _deferred_cleanup() -> None:
            time.sleep(_RUN_TTL_S)
            _cleanup_sandbox(run_id)
            with _runs_lock:
                _runs.pop(run_id, None)

        t = threading.Thread(target=_deferred_cleanup, daemon=True)
        t.start()


@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise database and sandbox directory on startup."""
    _init_db()
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("System-3/0 API started (version %s)", _API_VERSION)
    yield


app = FastAPI(
    title       = "System-3/0 Demo API",
    description = (
        "HTTP API exposing the System-3/0 multi-agent LLM orchestration "
        "framework.  See the thesis (https://doi.org/10.5281/zenodo.22057717), Chapter 6, for the experimental protocol."
    ),
    version     = _API_VERSION,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = _lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = [_CORS_ORIGIN] if _CORS_ORIGIN != "*" else ["*"],
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["*"],
    allow_credentials = _CORS_ORIGIN != "*",
)


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """
    Liveness check.

    Returns 200 with Ollama reachability status.  Does not call Ollama
    itself; performs a lightweight HTTP HEAD against the Ollama API tags
    endpoint.
    """
    ollama_ok = False
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _requests_lib.get(f"{_OLLAMA_URL}/api/tags", timeout=2),
        )
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    return HealthResponse(ok=True, ollama=ollama_ok, version=_API_VERSION)


@app.post("/api/run", response_model=RunQueued, status_code=202, tags=["runs"])
async def post_run(req: RunRequest, request: Request) -> RunQueued:
    """
    Queue a new run.

    Body: {goal: "<demo_goal_id>", condition: "<condition>"}
    Returns: {run_id: "<uuid>", status: "queued"}

    Raises 400 if the goal is not in the curated demo set or the condition
    is invalid.  Raises 429 if the IP has exceeded the rate limit.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit check
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {_RATE_LIMIT} runs per hour per IP.",
        )

    # Goal allow-list check
    if req.goal not in DEMO_GOALS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Goal {req.goal!r} is not in the curated demo set.  "
                f"Allowed goals: {sorted(DEMO_GOALS)}."
            ),
        )

    run_id = str(uuid.uuid4())

    with _runs_lock:
        _runs[run_id] = {
            "status":    "queued",
            "goal_id":   req.goal,
            "condition": req.condition,
            "run_log":   None,
        }

    # Dispatch to background thread
    asyncio.get_event_loop().run_in_executor(
        _executor,
        _run_worker,
        run_id,
        req.goal,
        req.condition,
    )

    return RunQueued(run_id=run_id, status="queued")


@app.get("/api/run/{run_id}", response_model=RunStatusResponse, tags=["runs"])
async def get_run(run_id: str) -> RunStatusResponse:
    """
    Poll the status or final RunLog for a run.

    Returns the full RunLog (schema in mas/eval/harness.py) once the run completes.
    Returns 404 if the run_id is unknown or has been cleaned up.
    """
    with _runs_lock:
        entry = _runs.get(run_id)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    return RunStatusResponse(
        run_id    = run_id,
        goal_id   = entry.get("goal_id"),
        condition = entry.get("condition"),
        status    = entry["status"],
        run_log   = entry.get("run_log"),
    )


@app.get("/api/run/{run_id}/stream", tags=["runs"])
async def stream_run(run_id: str) -> StreamingResponse:
    """
    Server-sent event stream for per-step updates.

    Each event has the form::

        event: step
        data: {"step_id": 1, "agent_id": "WebAgent", ...}

    The stream ends with an event: done or event: error.
    Returns 404 if the run_id is unknown.
    """
    with _runs_lock:
        entry = _runs.get(run_id)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    # If already done, send a single synthetic done event
    if entry["status"] in ("done", "error"):
        run_log = entry.get("run_log") or {}

        async def _instant_stream() -> AsyncGenerator[str, None]:
            # Replay per-step events from the already-completed RunLog
            for step in run_log.get("per_step", []):
                data = json.dumps({
                    "step_id":               step.get("step_id"),
                    "agent_id":              step.get("agent_id"),
                    "schema_valid":          step.get("schema_valid"),
                    "would_have_been_invalid": step.get("would_have_been_invalid"),
                    "tokens_used":           step.get("tokens_used"),
                })
                yield f"event: step\ndata: {data}\n\n"
            final_event = "done" if entry["status"] == "done" else "error"
            final_data = json.dumps({
                "success":         run_log.get("success"),
                "cascade_failure": run_log.get("cascade_failure"),
            })
            yield f"event: {final_event}\ndata: {final_data}\n\n"

        return StreamingResponse(
            _instant_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    loop = asyncio.get_event_loop()
    return StreamingResponse(
        _sse_generator(run_id, loop),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/waitlist", response_model=WaitlistResponse, tags=["meta"])
async def post_waitlist(req: WaitlistRequest) -> WaitlistResponse:
    """
    Subscribe to the launch waitlist.

    Idempotent: submitting the same e-mail twice does not create a duplicate.
    Returns {ok: true} on success.
    """
    await asyncio.get_event_loop().run_in_executor(
        None,
        _upsert_waitlist,
        str(req.email),
        req.use_case,
    )
    return WaitlistResponse(ok=True)
