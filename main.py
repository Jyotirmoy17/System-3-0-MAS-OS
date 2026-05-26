#!/usr/bin/env python3
"""MAS-OS entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
_DEFAULT_WORKSPACE = _REPO / "workspace"

_CLI_EPILOG = """
Commands:
  serve              Start the web dashboard server
  run "goal"         Run a single goal and print the result
  popup              Launch the Spotlight-style popup
  daemon start|...   Control the background daemon
  check              Verify Ollama, models, workspace
  registry           List all registered agents

Global options:
  --slm MODEL        SLM model name (default: phi3:mini)
  --llm MODEL        LLM model name (default: llama3.1:8b)
  --url URL          Ollama base URL
  --budget N         Token budget (default: 50000)
  --tau F            Escalation threshold (default: 0.72)
  --workspace DIR    Workspace root directory
  --port N           Dashboard port (default: 8000)
  --host HOST        Dashboard bind host
  --log LEVEL        Logging level (default: INFO)
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "mas-os",
        description = "MAS-OS — Agentic Operating System",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = _CLI_EPILOG,
    )

    # Global options
    p.add_argument("--slm",       default="phi3:mini",                         metavar="MODEL")
    p.add_argument("--llm",       default="llama3.1:8b",                       metavar="MODEL")
    p.add_argument("--url",       default="http://localhost:11434",             metavar="URL")
    p.add_argument("--budget",    default=50_000,  type=int,                   metavar="N")
    p.add_argument("--tau",       default=0.72,    type=float,                 metavar="F")
    p.add_argument("--workspace", default=str(_DEFAULT_WORKSPACE),  metavar="DIR")
    p.add_argument("--port",      default=8000,    type=int,                   metavar="N")
    p.add_argument("--host",      default="0.0.0.0",                           metavar="HOST")
    p.add_argument("--log",       default="INFO",                               metavar="LEVEL")

    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # serve
    sub.add_parser("serve",    help="Start web dashboard on --host:--port")

    # run
    run_p = sub.add_parser("run", help="Execute a single goal and print the result")
    run_p.add_argument("goal", help="Natural language goal to execute")

    # check
    sub.add_parser("check",    help="Verify Ollama, model availability, workspace")

    # registry
    sub.add_parser("registry", help="List all registered agents and capabilities")

    # popup
    sub.add_parser("popup",    help="Launch Spotlight-style popup (Ubuntu + pywebview)")

    # daemon
    d = sub.add_parser("daemon", help="Control the background daemon")
    d.add_argument(
        "daemon_cmd",
        nargs   = "?",
        default = "status",
        choices = ["start", "stop", "status", "install"],
        metavar = "start|stop|status|install",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command is None:
        _banner()
        parser.print_help()
        return

    from mas.interfaces.cli import (  # noqa: PLC0415
        setup_logging,
        cmd_check,
        cmd_registry,
        cmd_run,
        cmd_serve,
        cmd_popup,
        cmd_daemon,
    )
    setup_logging(args.log)

    dispatch = {
        "check":    cmd_check,
        "registry": cmd_registry,
        "run":      cmd_run,
        "serve":    cmd_serve,
        "popup":    cmd_popup,
        "daemon":   cmd_daemon,
    }
    dispatch[args.command](args)


def _banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║  MAS-OS — Agentic Operating System                          ║
║  System-3/0 Blueprint · Jyotirmoy Singh · BITS Pilani       ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
