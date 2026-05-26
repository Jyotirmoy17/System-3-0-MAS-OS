"""MAS-OS background daemon. Unix socket server. Newline-delimited JSON protocol."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

SOCKET_PATH = "/tmp/mas_os.sock"
PID_FILE    = "/tmp/mas_os.pid"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE   = _REPO_ROOT / "workspace"
LOG_FILE    = WORKSPACE / "logs" / "daemon.log"

log = logging.getLogger("mas.daemon")


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    try:
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from mas.kernel import create_kernel  # noqa: PLC0415
        _kernel = create_kernel(workspace=WORKSPACE)
        log.info("Kernel initialised")
    except Exception:
        log.exception("Kernel initialisation failed")
    return _kernel


class _History:
    MAX = 30

    def __init__(self) -> None:
        self._path = WORKSPACE / "memory" / "history.json"
        self._data: List[Dict[str, Any]] = self._load()

    def push(self, goal: str, answer: str, tokens: int, ok: bool) -> None:
        self._data.insert(0, {
            "goal":   goal[:100],
            "answer": answer[:250],
            "tokens": tokens,
            "ok":     ok,
            "ts":     time.time(),
        })
        self._data = self._data[:self.MAX]
        self._save()

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._data[:n]

    def _load(self) -> List[Dict[str, Any]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, default=str),
            encoding="utf-8",
        )


_history = _History()


async def _send(writer: asyncio.StreamWriter, data: Dict[str, Any]) -> None:
    try:
        writer.write((json.dumps(data) + "\n").encode("utf-8"))
        await writer.drain()
    except Exception:
        pass


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    log.debug("Client connected")
    try:
        while True:
            try:
                raw = await asyncio.wait_for(reader.readline(), timeout=300)
            except asyncio.TimeoutError:
                break
            if not raw:
                break

            try:
                msg = json.loads(raw.decode("utf-8").strip())
            except json.JSONDecodeError:
                continue

            kind = msg.get("type")

            if kind == "ping":
                await _send(writer, {"type": "pong", "ts": time.time()})

            elif kind == "history":
                await _send(writer, {
                    "type":    "history",
                    "entries": _history.recent(10),
                })

            elif kind == "goal":
                goal = str(msg.get("goal", "")).strip()
                sid  = str(msg.get("session_id", ""))
                if not goal:
                    continue

                log.info("Goal: %.80s", goal)
                kernel = _get_kernel()

                if kernel is None:
                    await _send(writer, {
                        "type":    "error",
                        "message": "Kernel not ready — is Ollama running?",
                    })
                    continue

                await _send(writer, {"type": "goal_received", "goal": goal})

                try:
                    async for chunk in kernel.run_stream(goal, sid):
                        await _send(writer, chunk)
                    result = kernel.run(goal, sid)
                    _history.push(
                        goal,
                        result.get("answer", ""),
                        result.get("total_tokens", 0),
                        bool(result.get("answer")),
                    )
                    await _send(writer, {"type": "result", **result})
                except Exception:
                    log.exception("Stream/run error")
                    await _send(writer, {
                        "type":    "error",
                        "message": "Execution error — see daemon log",
                    })

    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass
    log.debug("Client disconnected")


async def _serve() -> None:
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    server = await asyncio.start_unix_server(_handle, SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    log.info("Daemon listening on %s", SOCKET_PATH)
    async with server:
        await server.serve_forever()


def _setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


def cmd_start() -> None:
    _setup_logging()
    Path(PID_FILE).write_text(str(os.getpid()))
    log.info("MAS-OS daemon started (PID %d)", os.getpid())
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        log.info("Daemon stopped by KeyboardInterrupt")
    finally:
        for p in (PID_FILE, SOCKET_PATH):
            try:
                os.remove(p)
            except OSError:
                pass


def cmd_stop() -> None:
    pid_path = Path(PID_FILE)
    if not pid_path.exists():
        print("Daemon is not running (no PID file)")
        return
    pid = int(pid_path.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped daemon (PID {pid})")
    except ProcessLookupError:
        print(f"PID {pid} not found — daemon may have already exited")
    pid_path.unlink(missing_ok=True)


def cmd_status() -> None:
    pid_path = Path(PID_FILE)
    if not pid_path.exists():
        print("Daemon: NOT running")
        return
    pid = int(pid_path.read_text())
    try:
        os.kill(pid, 0)
        print(f"Daemon: running (PID {pid})")
    except ProcessLookupError:
        print(f"Daemon: stale PID {pid} — not actually running")


_UNIT = """\
[Unit]
Description=MAS-OS Agentic Operating System Daemon
After=network.target

[Service]
Type=simple
ExecStart={python} -m mas.interfaces.daemon start
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""


def cmd_install() -> None:
    svc_dir = Path.home() / ".config" / "systemd" / "user"
    svc_dir.mkdir(parents=True, exist_ok=True)
    svc_file = svc_dir / "mas-os.service"
    svc_file.write_text(_UNIT.format(python=sys.executable))
    os.system("systemctl --user daemon-reload")
    os.system("systemctl --user enable mas-os.service")
    print(f"Installed: {svc_file}")
    print("Start now:  systemctl --user start mas-os.service")
    print("On boot:    already enabled")


if __name__ == "__main__":
    cmds = {
        "start":   cmd_start,
        "stop":    cmd_stop,
        "status":  cmd_status,
        "install": cmd_install,
    }
    arg = sys.argv[1] if len(sys.argv) > 1 else "start"
    fn  = cmds.get(arg)
    if fn:
        fn()
    else:
        print(f"Unknown command: {arg}. Use: {list(cmds)}")
        sys.exit(1)
