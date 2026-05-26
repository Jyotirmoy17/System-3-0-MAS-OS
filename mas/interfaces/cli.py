"""All CLI command implementations.  main.py calls into this."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
        level   = getattr(logging, level.upper(), logging.INFO),
    )


def _banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║  MAS-OS — Agentic Operating System                          ║
║  System-3/0 Blueprint · Jyotirmoy Singh · BITS Pilani       ║
╚══════════════════════════════════════════════════════════════╝
""")


def cmd_check(args: argparse.Namespace) -> None:
    from mas.core.ollama_client import OllamaClient  # noqa: PLC0415
    from mas.agents.defaults    import build_registry # noqa: PLC0415

    _banner()
    print("── Health Check ─────────────────────────────────────\n")

    client = OllamaClient(args.url)

    if not client.is_available():
        print(f"  ✗  Ollama not reachable at {args.url}")
        print("     Start it with: ollama serve")
        sys.exit(1)

    models = client.list_models()
    print(f"  ✓  Ollama running  |  models: {models}")

    ok = True
    for label, model in [("SLM (S1)", args.slm), ("LLM (S2)", args.llm)]:
        has = client.has_model(model)
        print(f"  {'✓' if has else '✗'}  {label}: {model}")
        if not has:
            print(f"     Pull it with: ollama pull {model}")
            ok = False

    R = build_registry(args.slm, args.llm)
    s = R.summary()
    print(f"\n  ✓  Registry: {s['total']} agents  "
          f"({s['s1_count']} S1 · {s['s2_count']} S2)")
    for aid, desc in s["agents"].items():
        print(f"       {aid:<26}  {desc[:55]}")

    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    print(f"\n  ✓  Workspace: {ws}")

    if ok:
        print("\n  All checks passed.  Run: python main.py serve\n")
    else:
        sys.exit(1)


def cmd_registry(args: argparse.Namespace) -> None:
    from mas.agents.defaults import build_registry  # noqa: PLC0415

    R = build_registry(args.slm, args.llm)
    s = R.summary()
    print(f"\nAgent Registry — {s['total']} agents\n")
    print(f"{'ID':<26} {'Tier':<5} {'Strategy':<18} Description")
    print("─" * 82)
    for aid, spec in R._specs.items():
        print(f"{aid:<26} {spec.tier.value:<5} {spec.strategy:<18} {spec.description[:35]}")
    print(f"\nCapabilities: {', '.join(s['capabilities'])}")


def cmd_run(args: argparse.Namespace) -> None:
    from mas.kernel import create_kernel  # noqa: PLC0415

    _banner()
    print(f"Goal: {args.goal}\n{'─' * 60}\n")

    kernel = create_kernel(
        slm_model  = args.slm,
        llm_model  = args.llm,
        ollama_url = args.url,
        budget     = args.budget,
        tau        = args.tau,
        workspace  = Path(args.workspace),
    )
    result = kernel.run(args.goal)

    print(f"Answer:\n{result['answer']}\n{'─' * 60}")
    print(f"Confidence:  {result['confidence']}")
    print(f"Complexity:  {result['complexity']}")
    print(f"Tokens used: {result['total_tokens']:,}")
    print(f"Escalations: {result['escalations']}")
    print(f"Schema Vs:   {result['system0']['Vs']}")

    print("\nAgent trace:")
    for ev in result.get("agent_events", []):
        t = ev.get("type", "")
        if t == "plan":
            print(f"  [S3] {ev.get('complexity')} → {ev.get('selected_agent')}")
        elif t == "agent_start":
            print(f"  [  ] {ev.get('agent_name', ev.get('agent_id'))}")
        elif t == "tool_call":
            icon = "✓" if ev.get("success") else "✗"
            print(f"       {icon} {ev.get('tool')} — {str(ev.get('preview',''))[:60]}")
        elif t == "s0_validate":
            pred = ev.get("predicate", "none")
            esc  = f" → escalating ({pred})" if pred != "none" else ""
            print(f"  [S0] Vs={ev.get('Vs')}{esc}")
        elif t in ("error", "halt"):
            print(f"  [!!] {t}: {ev.get('msg') or ev.get('reason')}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn                                         # noqa: PLC0415
    from mas.kernel         import create_kernel           # noqa: PLC0415
    from mas.interfaces.api import app, init as api_init  # noqa: PLC0415

    _banner()
    print(f"  SLM:        {args.slm}")
    print(f"  LLM:        {args.llm}")
    print(f"  Budget:     {args.budget:,} tokens")
    print(f"  Threshold:  τ = {args.tau}")
    print(f"  Workspace:  {args.workspace}")
    print(f"  Dashboard:  http://localhost:{args.port}\n")

    kernel = create_kernel(
        slm_model  = args.slm,
        llm_model  = args.llm,
        ollama_url = args.url,
        budget     = args.budget,
        tau        = args.tau,
        workspace  = Path(args.workspace),
    )
    api_init(kernel)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log.lower())


def cmd_popup(_args: argparse.Namespace) -> None:
    from mas.interfaces.popup import main as popup_main  # noqa: PLC0415
    popup_main()


def cmd_daemon(args: argparse.Namespace) -> None:
    from mas.interfaces.daemon import (  # noqa: PLC0415
        cmd_install, cmd_start, cmd_status, cmd_stop,
    )
    dispatch = {
        "start":   cmd_start,
        "stop":    cmd_stop,
        "status":  cmd_status,
        "install": cmd_install,
    }
    sub = getattr(args, "daemon_cmd", "status")
    fn  = dispatch.get(sub)
    if fn:
        fn()
    else:
        print(f"Unknown daemon command: {sub}")
        sys.exit(1)
