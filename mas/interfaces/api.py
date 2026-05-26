"""FastAPI HTTP + WebSocket server.  Serves the web dashboard UI and."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger(__name__)

app = FastAPI(title="MAS-OS", version="2.0.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_DIR = Path(__file__).parent.parent.parent / "ui"
if (_UI_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR / "static")), name="static")

_kernel: Optional[Any] = None


def init(kernel: Any) -> None:
    """Attach a MASKernel instance.  Must be called before serving."""
    global _kernel
    _kernel = kernel
    log.info("API: kernel attached")


@app.get("/")
async def index() -> Any:
    f = _UI_DIR / "dashboard.html"
    return FileResponse(str(f)) if f.exists() else HTMLResponse(
        "<h1>MAS-OS running</h1><p>dashboard.html not found</p>"
    )


@app.get("/api/health")
async def health() -> dict:
    if _kernel is None:
        return {"status": "no_kernel"}
    return {"status": "ok", **_kernel.telemetry()}


@app.get("/api/registry")
async def registry() -> dict:
    if _kernel is None:
        return {"error": "kernel not initialised"}
    return _kernel.registry.summary()


@app.post("/api/run")
async def run_goal(body: dict) -> dict:
    if _kernel is None:
        return {"error": "Kernel not initialised"}
    goal = str(body.get("goal", "")).strip()
    if not goal:
        return {"error": "Field 'goal' is required"}
    return _kernel.run(goal)


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    log.info("WebSocket connected: %s", session_id)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong", "ts": time.time()})
                continue

            if msg.get("type") != "goal":
                continue

            goal = str(msg.get("goal", "")).strip()
            if not goal:
                continue

            if _kernel is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Kernel not initialised",
                })
                continue

            await websocket.send_json({
                "type": "goal_received",
                "goal": goal,
                "ts":   time.time(),
            })

            try:
                async for chunk in _kernel.run_stream(goal, session_id):
                    await websocket.send_json(chunk)
                    await asyncio.sleep(0)   # yield to event loop
                await websocket.send_json(_kernel.telemetry() | {"type": "telemetry"})
            except Exception:
                log.exception("Streaming error for session %s", session_id)
                await websocket.send_json({"type": "error", "message": "Streaming failed"})

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", session_id)
    except Exception:
        log.exception("WebSocket error for session %s", session_id)
