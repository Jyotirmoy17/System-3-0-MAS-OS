"""Interactive terminal demo for the System-3/0 framework."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


# Event kind constants
K_STEP_START   = "step_start"    # step begins
K_STEP_DONE    = "step_done"     # step completes, schema valid
K_STEP_INVALID = "step_invalid"  # step output fails schema (not yet caught)
K_S0_CATCH     = "s0_catch"      # System-0 catches invalid output
K_S0_REPAIR    = "s0_repair"     # System-0 repairs output
K_CASCADE      = "cascade"       # corrupted output forwarded downstream
K_ESCALATE     = "escalate"      # System-3 predicate fires
K_SUCCESS      = "success"       # final: run succeeds
K_FAILURE      = "failure"       # final: cascade failure


@dataclass
class DemoEvent:
    """One timestamped event in a scripted or live-replayed demo run."""
    t: float            # relative time in seconds from demo start
    condition: str      # "FULL_FRAMEWORK" | "BASELINE_LANGGRAPH" | "ANY"
    kind: str           # one of the K_* constants above
    agent_id: str = ""  # agent name for step events
    step_id: int  = 0   # 1-based step index
    message: str  = ""  # human-readable detail
    tokens: int   = 0   # tokens used in this step (or total for summary events)
    latency_s: float = 0.0  # step or run latency


# Times are chosen so the cascade demo finishes in ~30 seconds at 1x speed.
# The --speed flag divides all sleep durations (2x → ~15 s, etc.).

_CASCADE_SCRIPT: list[DemoEvent] = [
    # ── FULL_FRAMEWORK ────────────────────────────────────────────────
    DemoEvent(t=0.0,  condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=1, agent_id="WebAgent (S1)",
              message="Fetching and parsing source documents"),
    DemoEvent(t=2.5,  condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=1, agent_id="WebAgent (S1)",
              tokens=420, latency_s=2.5),
    DemoEvent(t=2.7,  condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=2, agent_id="ClassifierAgent (S1)",
              message="Classifying corruption patterns"),
    # Step 2 produces invalid output — System-0 catches and repairs it
    DemoEvent(t=5.1,  condition="FULL_FRAMEWORK", kind=K_STEP_INVALID,
              step_id=2, agent_id="ClassifierAgent (S1)",
              message='type mismatch: "confidence" expected float, got "high"'),
    DemoEvent(t=5.3,  condition="FULL_FRAMEWORK", kind=K_S0_CATCH,
              step_id=2, agent_id="ClassifierAgent (S1)",
              message='type mismatch: confidence expected float, got "high"'),
    DemoEvent(t=5.7,  condition="FULL_FRAMEWORK", kind=K_S0_REPAIR,
              step_id=2, agent_id="ClassifierAgent (S1)",
              message="constrained decoding → confidence: 0.87",
              tokens=380, latency_s=2.6),
    DemoEvent(t=5.9,  condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=2, agent_id="ClassifierAgent (S1)",
              tokens=380, latency_s=2.6),
    DemoEvent(t=6.1,  condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=3, agent_id="ReActAgent (S2)",
              message="Reasoning over classified output"),
    DemoEvent(t=13.5, condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=3, agent_id="ReActAgent (S2)",
              tokens=1240, latency_s=7.4),
    DemoEvent(t=13.7, condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=4, agent_id="VerifierAgent (S2)",
              message="Verifying answer against ground truth"),
    DemoEvent(t=20.2, condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=4, agent_id="VerifierAgent (S2)",
              tokens=890, latency_s=6.5),
    DemoEvent(t=20.5, condition="FULL_FRAMEWORK", kind=K_SUCCESS,
              tokens=2930, latency_s=20.5),

    # ── BASELINE_LANGGRAPH ────────────────────────────────────────────
    DemoEvent(t=0.0,  condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
              step_id=1, agent_id="ChainNode:WebAgent",
              message="Fetching and parsing source documents"),
    DemoEvent(t=2.5,  condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
              step_id=1, agent_id="ChainNode:WebAgent",
              tokens=420, latency_s=2.5),
    DemoEvent(t=2.7,  condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
              step_id=2, agent_id="ChainNode:ClassifierAgent",
              message="Classifying corruption patterns"),
    # Step 2 invalid — no System-0, so it cascades forward
    DemoEvent(t=5.1,  condition="BASELINE_LANGGRAPH", kind=K_STEP_INVALID,
              step_id=2, agent_id="ChainNode:ClassifierAgent",
              message='type mismatch: "confidence" expected float, got "high"'),
    DemoEvent(t=5.2,  condition="BASELINE_LANGGRAPH", kind=K_CASCADE,
              step_id=2, agent_id="ChainNode:ClassifierAgent",
              message="corrupted output forwarded — no enforcement"),
    DemoEvent(t=5.3,  condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
              step_id=2, agent_id="ChainNode:ClassifierAgent",
              tokens=380, latency_s=2.6),
    DemoEvent(t=5.5,  condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
              step_id=3, agent_id="ChainNode:ReActAgent",
              message="Reasoning over corrupted classification"),
    DemoEvent(t=12.5, condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
              step_id=3, agent_id="ChainNode:ReActAgent",
              tokens=1580, latency_s=7.0),  # more tokens: reasoning over bad input
    DemoEvent(t=12.7, condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
              step_id=4, agent_id="ChainNode:VerifierAgent",
              message="Verifying answer (based on corrupted context)"),
    DemoEvent(t=18.5, condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
              step_id=4, agent_id="ChainNode:VerifierAgent",
              tokens=740, latency_s=5.8),
    DemoEvent(t=18.8, condition="BASELINE_LANGGRAPH", kind=K_FAILURE,
              tokens=3120, latency_s=18.8),
]

_BUDGET_SCRIPT: list[DemoEvent] = [
    # ── FULL_FRAMEWORK (LinUCB routing) ──────────────────────────────
    DemoEvent(t=0.0,  condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=1, agent_id="WebAgent (S1)",
              message="LinUCB selects cheapest capable agent"),
    DemoEvent(t=2.1,  condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=1, agent_id="WebAgent (S1)", tokens=380),
    DemoEvent(t=2.3,  condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=2, agent_id="CalculatorAgent (S1)",
              message="LinUCB routes math sub-task to specialist"),
    DemoEvent(t=3.4,  condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=2, agent_id="CalculatorAgent (S1)", tokens=240),
    DemoEvent(t=3.6,  condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=3, agent_id="ReActAgent (S2)",
              message="S2 synthesis — S1 subtasks already done"),
    DemoEvent(t=9.8,  condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=3, agent_id="ReActAgent (S2)", tokens=820),
    DemoEvent(t=10.1, condition="FULL_FRAMEWORK", kind=K_SUCCESS,
              tokens=1440, latency_s=10.1),

    # ── BASELINE_LANGGRAPH (single-agent, no routing) ─────────────────
    DemoEvent(t=0.0,  condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
              step_id=1, agent_id="baseline_llm (S2)",
              message="Single agent — all sub-tasks sent to LLM directly"),
    DemoEvent(t=7.2,  condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
              step_id=1, agent_id="baseline_llm (S2)", tokens=1180),
    DemoEvent(t=7.4,  condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
              step_id=2, agent_id="baseline_llm (S2)",
              message="Math sub-task also handled by same LLM"),
    DemoEvent(t=14.1, condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
              step_id=2, agent_id="baseline_llm (S2)", tokens=1420),
    DemoEvent(t=14.3, condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
              step_id=3, agent_id="baseline_llm (S2)",
              message="Synthesis — repeating context already computed"),
    DemoEvent(t=21.5, condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
              step_id=3, agent_id="baseline_llm (S2)", tokens=1680),
    DemoEvent(t=21.8, condition="BASELINE_LANGGRAPH", kind=K_SUCCESS,
              tokens=4280, latency_s=21.8),
]

_ESCALATE_SCRIPT: list[DemoEvent] = [
    # Single-condition display showing predicate escalation
    DemoEvent(t=0.0,  condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=1, agent_id="WebAgent (S1)",
              message="Step 1: document retrieval"),
    DemoEvent(t=2.3,  condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=1, agent_id="WebAgent (S1)", tokens=410),
    DemoEvent(t=2.5,  condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=2, agent_id="ClassifierAgent (S1)",
              message="Step 2: classification"),
    DemoEvent(t=4.8,  condition="FULL_FRAMEWORK", kind=K_STEP_INVALID,
              step_id=2, agent_id="ClassifierAgent (S1)",
              message='required field "label" missing'),
    DemoEvent(t=4.9,  condition="FULL_FRAMEWORK", kind=K_S0_CATCH,
              step_id=2, agent_id="ClassifierAgent (S1)",
              message='missing required field "label"'),
    DemoEvent(t=5.2,  condition="FULL_FRAMEWORK", kind=K_S0_REPAIR,
              step_id=2, message="repair attempt 1", tokens=380),
    # Repair fails; second failure → P2 threshold crossed
    DemoEvent(t=5.4,  condition="FULL_FRAMEWORK", kind=K_STEP_INVALID,
              step_id=2, message="repair still invalid — schema_failure_rate=0.67"),
    DemoEvent(t=5.6,  condition="FULL_FRAMEWORK", kind=K_ESCALATE,
              step_id=2,
              message="P2: schema_failure_rate > 0.5 → VerifierRetryAgent"),
    DemoEvent(t=5.8,  condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=3, agent_id="VerifierRetryAgent (S2)",
              message="Retry with full schema constraint injection"),
    DemoEvent(t=13.2, condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=3, agent_id="VerifierRetryAgent (S2)", tokens=1150),
    DemoEvent(t=13.4, condition="FULL_FRAMEWORK", kind=K_STEP_START,
              step_id=4, agent_id="ReActAgent (S2)",
              message="Synthesis on verified output"),
    DemoEvent(t=19.8, condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
              step_id=4, agent_id="ReActAgent (S2)", tokens=980),
    DemoEvent(t=20.1, condition="FULL_FRAMEWORK", kind=K_SUCCESS,
              tokens=2520, latency_s=20.1),
]


def _build_panel_text(events: list[DemoEvent], condition: str) -> Text:
    """Convert a list of DemoEvents for one condition into a rich.Text block."""
    t = Text()
    step_open = False

    for ev in events:
        if ev.kind == K_STEP_START:
            if step_open:
                t.append("\n")
            t.append(f"\n  Step {ev.step_id}", style="bold white")
            t.append(f"  {ev.agent_id}", style="bold cyan")
            if ev.message:
                t.append(f"\n    {ev.message}", style="dim white")
            step_open = True

        elif ev.kind == K_STEP_DONE:
            tok_str = f"  [{ev.tokens:,} tok]" if ev.tokens else ""
            t.append(f"\n    ✓ valid{tok_str}", style="green")

        elif ev.kind == K_STEP_INVALID:
            t.append(f"\n    ✗ schema invalid", style="yellow")
            if ev.message:
                t.append(f" — {ev.message}", style="dim yellow")

        elif ev.kind == K_S0_CATCH:
            t.append(f"\n    ⚡ System-0 CAUGHT", style="bold green")
            if ev.message:
                t.append(f" — {ev.message}", style="green")

        elif ev.kind == K_S0_REPAIR:
            msg = ev.message or "constrained decoding"
            t.append(f"\n    ↩  repaired → {msg}", style="green")

        elif ev.kind == K_CASCADE:
            t.append(f"\n    ✗ CORRUPTED OUTPUT PASSED FORWARD", style="bold red")
            if ev.message:
                t.append(f"\n      {ev.message}", style="dim red")

        elif ev.kind == K_ESCALATE:
            t.append(f"\n\n  ⬆  System-3 PREDICATE FIRED", style="bold magenta")
            if ev.message:
                t.append(f"\n    {ev.message}", style="magenta")

        elif ev.kind == K_SUCCESS:
            t.append(f"\n\n  ✓  SUCCESS", style="bold white on dark_green")
            if ev.tokens:
                t.append(f"\n  Tokens: {ev.tokens:,}", style="dim green")
            if ev.latency_s:
                t.append(f"  ·  Latency: {ev.latency_s:.1f}s", style="dim green")

        elif ev.kind == K_FAILURE:
            t.append(f"\n\n  ✗  CASCADE FAILURE", style="bold white on red")
            if ev.tokens:
                t.append(f"\n  Tokens: {ev.tokens:,}", style="dim red")
            if ev.latency_s:
                t.append(f"  ·  Latency: {ev.latency_s:.1f}s", style="dim red")

    return t


_CONDITION_STYLES: dict[str, str] = {
    "FULL_FRAMEWORK":     "bold green",
    "BASELINE_LANGGRAPH": "bold red",
}

_CONDITION_BORDERS: dict[str, str] = {
    "FULL_FRAMEWORK":     "green",
    "BASELINE_LANGGRAPH": "red",
}


def _make_panel(events: list[DemoEvent], condition: str) -> Panel:
    text = _build_panel_text(events, condition)
    border = _CONDITION_BORDERS.get(condition, "white")
    return Panel(
        text,
        title=f"[{_CONDITION_STYLES.get(condition, 'white')}]{condition}[/]",
        border_style=border,
        padding=(0, 1),
    )


def _play_two_column(
    all_events: list[DemoEvent],
    left_cond:  str,
    right_cond: str,
    speed:      float,
    console:    Console,
    no_live:    bool,
) -> None:
    """Play a scripted two-column demo."""
    left_events:  list[DemoEvent] = []
    right_events: list[DemoEvent] = []

    sorted_events = sorted(all_events, key=lambda e: e.t)

    def _refresh_layout(layout: Layout) -> None:
        layout["left"].update(_make_panel(left_events,  left_cond))
        layout["right"].update(_make_panel(right_events, right_cond))

    def _dispatch(ev: DemoEvent) -> None:
        if ev.condition == left_cond:
            left_events.append(ev)
        else:
            right_events.append(ev)

    if no_live:
        # Tests / non-interactive: emit each event as a plain print
        for ev in sorted_events:
            _dispatch(ev)
        console.print(_make_panel(left_events,  left_cond))
        console.print(_make_panel(right_events, right_cond))
        return

    layout = Layout()
    layout.split_row(Layout(name="left"), Layout(name="right"))
    layout["left"].update(_make_panel(left_events,  left_cond))
    layout["right"].update(_make_panel(right_events, right_cond))

    prev_t = 0.0
    with Live(layout, console=console, refresh_per_second=20, screen=False) as live:
        for ev in sorted_events:
            dt = (ev.t - prev_t) / speed
            if dt > 0:
                time.sleep(dt)
            prev_t = ev.t
            _dispatch(ev)
            _refresh_layout(layout)
            live.refresh()


def _play_single_column(
    all_events: list[DemoEvent],
    condition:  str,
    speed:      float,
    console:    Console,
    no_live:    bool,
) -> None:
    """Play a scripted single-column demo (escalate mode)."""
    events: list[DemoEvent] = []
    sorted_events = sorted(all_events, key=lambda e: e.t)

    if no_live:
        for ev in sorted_events:
            events.append(ev)
        console.print(_make_panel(events, condition))
        return

    panel_ref: list[Panel] = [_make_panel(events, condition)]

    prev_t = 0.0
    with Live(panel_ref[0], console=console, refresh_per_second=20, screen=False) as live:
        for ev in sorted_events:
            dt = (ev.t - prev_t) / speed
            if dt > 0:
                time.sleep(dt)
            prev_t = ev.t
            events.append(ev)
            p = _make_panel(events, condition)
            panel_ref[0] = p
            live.update(p)


def _runlog_to_events(run_log: dict[str, Any]) -> list[DemoEvent]:
    """Convert a real RunLog dict into a list of DemoEvents for replay."""
    condition = run_log.get("condition", "FULL_FRAMEWORK")
    events: list[DemoEvent] = []
    t = 0.0

    for step in run_log.get("per_step", []):
        step_id  = step.get("step_id", 0)
        agent_id = step.get("agent_id", "unknown")
        lat_s    = step.get("latency_ms", 1000) / 1000.0
        tokens   = step.get("tokens_used", 0)
        valid    = step.get("schema_valid", True)
        whi      = step.get("would_have_been_invalid", False)

        events.append(DemoEvent(
            t=t, condition=condition, kind=K_STEP_START,
            step_id=step_id, agent_id=agent_id,
        ))

        if not valid and not whi:
            # System-0 caught and repaired
            events.append(DemoEvent(
                t=t + lat_s * 0.6, condition=condition, kind=K_S0_CATCH,
                step_id=step_id, agent_id=agent_id,
            ))
            events.append(DemoEvent(
                t=t + lat_s * 0.8, condition=condition, kind=K_S0_REPAIR,
                step_id=step_id, tokens=tokens,
            ))
        elif whi:
            # Would have been invalid — baseline cascade
            events.append(DemoEvent(
                t=t + lat_s * 0.7, condition=condition, kind=K_CASCADE,
                step_id=step_id,
            ))

        events.append(DemoEvent(
            t=t + lat_s, condition=condition, kind=K_STEP_DONE,
            step_id=step_id, agent_id=agent_id,
            tokens=tokens, latency_s=lat_s,
        ))
        t += lat_s

    total_tokens = run_log.get("tokens", {}).get("total", 0)
    total_lat    = run_log.get("latency_s", t)
    if run_log.get("success") == 1:
        events.append(DemoEvent(
            t=t + 0.2, condition=condition, kind=K_SUCCESS,
            tokens=total_tokens, latency_s=total_lat,
        ))
    else:
        events.append(DemoEvent(
            t=t + 0.2, condition=condition, kind=K_FAILURE,
            tokens=total_tokens, latency_s=total_lat,
        ))

    return events


def _live_run_two(
    goal_id:    str,
    left_cond:  str,
    right_cond: str,
    speed:      float,
    console:    Console,
    no_live:    bool,
) -> None:
    """Run two conditions via run_one() and replay the results."""
    from mas.eval.harness import run_one  # noqa: PLC0415

    console.print(Rule(f"Running {left_cond} vs {right_cond} on {goal_id} …"))

    with console.status("Executing FULL_FRAMEWORK …", spinner="dots"):
        left_log = run_one(goal_id, left_cond,  42, {})
    with console.status(f"Executing {right_cond} …", spinner="dots"):
        right_log = run_one(goal_id, right_cond, 42, {})

    # Normalise timestamps so both replays finish at the same wall-clock time
    left_events  = _runlog_to_events(left_log)
    right_events = _runlog_to_events(right_log)

    # Scale to a max 25-second replay
    def _scale(evs: list[DemoEvent], max_t: float = 25.0) -> list[DemoEvent]:
        actual_max = max((e.t for e in evs), default=1.0)
        ratio = max_t / actual_max if actual_max > 0 else 1.0
        return [
            DemoEvent(**{**ev.__dict__, "t": ev.t * ratio})
            for ev in evs
        ]

    left_events  = _scale(left_events)
    right_events = _scale(right_events)

    _play_two_column(
        left_events + right_events,
        left_cond, right_cond,
        speed, console, no_live,
    )


_DEFAULT_CASCADE_GOAL = "adversarial_002"


def demo_cascade(
    speed:     float    = 1.0,
    live_mode: bool     = False,
    goal_id:   str | None = None,
    _events:   list[DemoEvent] | None = None,
    _console:  Console | None = None,
    _no_live:  bool     = False,
) -> None:
    """Demo: System-0 cascade failure prevention.

Runs FULL_FRAMEWORK vs BASELINE_LANGGRAPH on an adversarial instance.
FULL_FRAMEWORK catches a type-flip corruption; BASELINE cascades to
a ground-truth mismatch."""
    console = _console or Console()
    console.print(Rule(
        "[bold green]System-3/0[/]  vs  [bold red]BASELINE_LANGGRAPH[/]"
        "  —  cascade failure demonstration",
        style="bold white",
    ))
    console.print()

    if live_mode:
        gid = goal_id or _DEFAULT_CASCADE_GOAL
        _live_run_two(gid, "FULL_FRAMEWORK", "BASELINE_LANGGRAPH",
                      speed, console, _no_live)
        return

    events = _events if _events is not None else _CASCADE_SCRIPT
    _play_two_column(
        events, "FULL_FRAMEWORK", "BASELINE_LANGGRAPH",
        speed, console, _no_live,
    )

    # Post-demo summary
    console.print()
    console.print(Rule("Result"))
    full_tok = next(
        (e.tokens for e in events
         if e.condition == "FULL_FRAMEWORK" and e.kind == K_SUCCESS),
        None,
    )
    base_tok = next(
        (e.tokens for e in events
         if e.condition == "BASELINE_LANGGRAPH" and e.kind in (K_SUCCESS, K_FAILURE)),
        None,
    )
    base_outcome = next(
        (e.kind for e in events
         if e.condition == "BASELINE_LANGGRAPH" and e.kind in (K_SUCCESS, K_FAILURE)),
        K_FAILURE,
    )
    if full_tok:
        console.print(f"  FULL_FRAMEWORK  — [green]SUCCESS[/]  ({full_tok:,} tokens)")
    if base_tok:
        colour = "red" if base_outcome == K_FAILURE else "green"
        label  = "CASCADE FAILURE" if base_outcome == K_FAILURE else "SUCCESS"
        console.print(f"  BASELINE        — [{colour}]{label}[/]  ({base_tok:,} tokens)")
    console.print()


def demo_budget(
    speed:    float    = 1.0,
    _events:  list[DemoEvent] | None = None,
    _console: Console | None = None,
    _no_live: bool     = False,
) -> None:
    """Demo: token-cost comparison on a multi-step goal.

Compares FULL_FRAMEWORK (LinUCB routing) against BASELINE_LANGGRAPH
(single-agent baseline)."""
    console = _console or Console()
    console.print(Rule(
        "[bold green]FULL_FRAMEWORK (LinUCB routing)[/]"
        "  vs  [bold red]BASELINE_LANGGRAPH[/]"
        "  —  token-cost comparison",
        style="bold white",
    ))
    console.print()

    events = _events if _events is not None else _BUDGET_SCRIPT
    _play_two_column(
        events, "FULL_FRAMEWORK", "BASELINE_LANGGRAPH",
        speed, console, _no_live,
    )

    # Post-demo summary
    console.print()
    console.print(Rule("Result"))
    full_tok = next(
        (e.tokens for e in events
         if e.condition == "FULL_FRAMEWORK" and e.kind == K_SUCCESS),
        None,
    )
    base_tok = next(
        (e.tokens for e in events
         if e.condition == "BASELINE_LANGGRAPH" and e.kind == K_SUCCESS),
        None,
    )
    if full_tok and base_tok and base_tok > 0:
        saved   = base_tok - full_tok
        pct     = 100.0 * saved / base_tok
        console.print(
            f"  FULL_FRAMEWORK (routed)  — [green]{full_tok:,} tokens[/]"
        )
        console.print(
            f"  BASELINE_LANGGRAPH       — [red]{base_tok:,} tokens[/]"
        )
        console.print(
            f"  Routing saved [bold green]{saved:,} tokens ({pct:.0f}%)[/]"
        )
    console.print()


def demo_escalate(
    speed:    float    = 1.0,
    _events:  list[DemoEvent] | None = None,
    _console: Console | None = None,
    _no_live: bool     = False,
) -> None:
    """Demo: System-3 P2 predicate firing and VerifierRetryAgent escalation.

Shows a single FULL_FRAMEWORK run where repeated schema failures trigger
the P2 predicate, escalating to the VerifierRetryAgent."""
    console = _console or Console()
    console.print(Rule(
        "[bold green]FULL_FRAMEWORK[/]"
        "  —  System-3 predicate escalation (P2)",
        style="bold white",
    ))
    console.print()

    events = _events if _events is not None else _ESCALATE_SCRIPT
    _play_single_column(events, "FULL_FRAMEWORK", speed, console, _no_live)

    console.print()
    console.print(Rule("Result"))
    outcome = next(
        (e.kind for e in events
         if e.condition == "FULL_FRAMEWORK" and e.kind in (K_SUCCESS, K_FAILURE)),
        None,
    )
    if outcome == K_SUCCESS:
        console.print("  [bold green]P2 escalation resolved — run SUCCEEDED[/]")
    elif outcome == K_FAILURE:
        console.print("  [bold red]Escalation could not recover — CASCADE FAILURE[/]")
    console.print()


def run_demo(
    mode:      str,
    speed:     float = 1.0,
    live_mode: bool  = False,
    goal_id:   str | None = None,
    _events:   list[DemoEvent] | None = None,
    _console:  Console | None = None,
    _no_live:  bool  = False,
) -> None:
    """Dispatch to the requested demo mode."""
    console = _console or Console()

    if mode == "cascade":
        demo_cascade(speed=speed, live_mode=live_mode, goal_id=goal_id,
                     _events=_events, _console=console, _no_live=_no_live)

    elif mode == "budget":
        demo_budget(speed=speed,
                    _events=_events, _console=console, _no_live=_no_live)

    elif mode == "escalate":
        demo_escalate(speed=speed,
                      _events=_events, _console=console, _no_live=_no_live)

    elif mode == "all":
        demo_cascade( speed=speed, live_mode=live_mode, goal_id=goal_id,
                      _events=_events, _console=console, _no_live=_no_live)
        demo_budget(  speed=speed,
                      _events=_events, _console=console, _no_live=_no_live)
        demo_escalate(speed=speed,
                      _events=_events, _console=console, _no_live=_no_live)

    else:
        raise ValueError(
            f"Unknown demo mode: {mode!r}. "
            "Valid modes: cascade, budget, escalate, all."
        )
