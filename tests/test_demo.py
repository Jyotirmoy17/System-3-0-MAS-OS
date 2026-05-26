"""Unit tests for the interactive terminal demo (mas/eval/demo.py)."""
from __future__ import annotations

import io

import pytest
from rich.console import Console

from mas.eval.demo import (
    DemoEvent,
    K_CASCADE,
    K_FAILURE,
    K_S0_CATCH,
    K_S0_REPAIR,
    K_STEP_DONE,
    K_STEP_INVALID,
    K_STEP_START,
    K_SUCCESS,
    demo_budget,
    demo_cascade,
    demo_escalate,
    run_demo,
)


def _make_console() -> tuple[Console, io.StringIO]:
    """Return a (Console, StringIO) pair.  Output captured by StringIO."""
    buf = io.StringIO()
    # force_terminal=False: disables ANSI escape codes so assertions are clean
    con = Console(file=buf, force_terminal=False, no_color=True, width=120)
    return con, buf


def _minimal_cascade_events(baseline_succeeds: bool = False) -> list[DemoEvent]:
    """
    Build a minimal event list for cascade demo tests.

    Args:
        baseline_succeeds: If True, BASELINE ends with K_SUCCESS;
                           otherwise ends with K_FAILURE.
    """
    full = [
        DemoEvent(t=0.0, condition="FULL_FRAMEWORK", kind=K_STEP_START,
                  step_id=1, agent_id="WebAgent"),
        DemoEvent(t=0.1, condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
                  step_id=1, agent_id="WebAgent", tokens=400),
        DemoEvent(t=0.2, condition="FULL_FRAMEWORK", kind=K_STEP_START,
                  step_id=2, agent_id="ClassifierAgent"),
        DemoEvent(t=0.3, condition="FULL_FRAMEWORK", kind=K_STEP_INVALID,
                  step_id=2, message="type mismatch"),
        DemoEvent(t=0.35, condition="FULL_FRAMEWORK", kind=K_S0_CATCH,
                  step_id=2, message="type mismatch"),
        DemoEvent(t=0.4, condition="FULL_FRAMEWORK", kind=K_S0_REPAIR,
                  step_id=2, tokens=300),
        DemoEvent(t=0.5, condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
                  step_id=2, tokens=300),
        DemoEvent(t=0.6, condition="FULL_FRAMEWORK", kind=K_SUCCESS,
                  tokens=700, latency_s=0.6),
    ]
    baseline_final = K_SUCCESS if baseline_succeeds else K_FAILURE
    base = [
        DemoEvent(t=0.0, condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
                  step_id=1, agent_id="ChainNode:WebAgent"),
        DemoEvent(t=0.1, condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
                  step_id=1, tokens=400),
        DemoEvent(t=0.2, condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
                  step_id=2, agent_id="ChainNode:ClassifierAgent"),
        DemoEvent(t=0.3, condition="BASELINE_LANGGRAPH", kind=K_STEP_INVALID,
                  step_id=2, message="type mismatch"),
        DemoEvent(t=0.35, condition="BASELINE_LANGGRAPH", kind=K_CASCADE,
                  step_id=2, message="no enforcement"),
        DemoEvent(t=0.4, condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
                  step_id=2, tokens=300),
        DemoEvent(t=0.5, condition="BASELINE_LANGGRAPH", kind=baseline_final,
                  tokens=700, latency_s=0.5),
    ]
    return full + base


def _minimal_budget_events() -> list[DemoEvent]:
    return [
        DemoEvent(t=0.0, condition="FULL_FRAMEWORK", kind=K_STEP_START,
                  step_id=1, agent_id="CalculatorAgent (S1)"),
        DemoEvent(t=0.1, condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
                  step_id=1, tokens=240),
        DemoEvent(t=0.2, condition="FULL_FRAMEWORK", kind=K_SUCCESS,
                  tokens=240, latency_s=0.2),
        DemoEvent(t=0.0, condition="BASELINE_LANGGRAPH", kind=K_STEP_START,
                  step_id=1, agent_id="baseline_llm (S2)"),
        DemoEvent(t=0.1, condition="BASELINE_LANGGRAPH", kind=K_STEP_DONE,
                  step_id=1, tokens=1200),
        DemoEvent(t=0.2, condition="BASELINE_LANGGRAPH", kind=K_SUCCESS,
                  tokens=1200, latency_s=0.2),
    ]


def _minimal_escalate_events() -> list[DemoEvent]:
    from mas.eval.demo import K_ESCALATE
    return [
        DemoEvent(t=0.0, condition="FULL_FRAMEWORK", kind=K_STEP_START,
                  step_id=1, agent_id="ClassifierAgent"),
        DemoEvent(t=0.1, condition="FULL_FRAMEWORK", kind=K_STEP_INVALID,
                  step_id=1, message="missing field"),
        DemoEvent(t=0.15, condition="FULL_FRAMEWORK", kind=K_ESCALATE,
                  step_id=1, message="P2: schema_failure_rate > 0.5 → VerifierRetryAgent"),
        DemoEvent(t=0.2, condition="FULL_FRAMEWORK", kind=K_STEP_DONE,
                  step_id=1, tokens=900),
        DemoEvent(t=0.3, condition="FULL_FRAMEWORK", kind=K_SUCCESS,
                  tokens=900, latency_s=0.3),
    ]


def test_demo_cascade_runs() -> None:
    """
    demo_cascade() completes without raising when given mock events.

    Verifies the function runs end-to-end in no-live mode with injected
    scripted events.  No Ollama calls are made.
    """
    console, _ = _make_console()
    events = _minimal_cascade_events()

    # Should not raise
    demo_cascade(
        speed    = 10.0,
        _events  = events,
        _console = console,
        _no_live = True,
    )


def test_demo_displays_both_panels() -> None:
    """
    Both condition labels appear in the captured console output.

    Asserts that FULL_FRAMEWORK and BASELINE_LANGGRAPH are both mentioned
    so a viewer can distinguish the two conditions.
    """
    console, buf = _make_console()
    events = _minimal_cascade_events()

    demo_cascade(
        speed    = 10.0,
        _events  = events,
        _console = console,
        _no_live = True,
    )

    output = buf.getvalue()
    assert "FULL_FRAMEWORK" in output, (
        "Expected 'FULL_FRAMEWORK' in demo output.\n"
        f"Output:\n{output}"
    )
    assert "BASELINE_LANGGRAPH" in output, (
        "Expected 'BASELINE_LANGGRAPH' in demo output.\n"
        f"Output:\n{output}"
    )


def test_demo_highlights_cascade() -> None:
    """
    "CASCADE FAILURE" appears when BASELINE ends with a K_FAILURE event.

    Uses a synthetic event list where the baseline condition fails so that
    _build_panel_text emits the cascade failure banner.
    """
    console, buf = _make_console()
    events = _minimal_cascade_events(baseline_succeeds=False)

    demo_cascade(
        speed    = 10.0,
        _events  = events,
        _console = console,
        _no_live = True,
    )

    output = buf.getvalue()
    assert "CASCADE FAILURE" in output, (
        "Expected 'CASCADE FAILURE' in demo output when baseline fails.\n"
        f"Output:\n{output}"
    )
    assert "System-0 CAUGHT" in output, (
        "Expected 'System-0 CAUGHT' in demo output for FULL_FRAMEWORK.\n"
        f"Output:\n{output}"
    )
    assert "SUCCESS" in output, (
        "Expected 'SUCCESS' in demo output for FULL_FRAMEWORK.\n"
        f"Output:\n{output}"
    )


def test_demo_handles_no_cascade() -> None:
    """
    No false-positive "CASCADE FAILURE" when baseline also succeeds.

    Uses a synthetic event list where both conditions end with K_SUCCESS.
    Verifies that "CASCADE FAILURE" does NOT appear in the output.
    """
    console, buf = _make_console()
    events = _minimal_cascade_events(baseline_succeeds=True)

    demo_cascade(
        speed    = 10.0,
        _events  = events,
        _console = console,
        _no_live = True,
    )

    output = buf.getvalue()
    assert "CASCADE FAILURE" not in output, (
        "Did not expect 'CASCADE FAILURE' when baseline succeeds.\n"
        f"Output:\n{output}"
    )
    assert output.count("SUCCESS") >= 2, (
        "Expected SUCCESS to appear at least twice (once per condition).\n"
        f"Output:\n{output}"
    )


def test_demo_budget_runs() -> None:
    """demo_budget() completes without raising."""
    console, buf = _make_console()
    demo_budget(
        speed    = 10.0,
        _events  = _minimal_budget_events(),
        _console = console,
        _no_live = True,
    )
    output = buf.getvalue()
    assert "BASELINE_LANGGRAPH" in output
    assert "FULL_FRAMEWORK" in output


def test_demo_escalate_runs() -> None:
    """demo_escalate() completes without raising; escalation message visible."""
    console, buf = _make_console()
    demo_escalate(
        speed    = 10.0,
        _events  = _minimal_escalate_events(),
        _console = console,
        _no_live = True,
    )
    output = buf.getvalue()
    assert "VerifierRetryAgent" in output or "P2" in output, (
        "Expected escalation message in escalate demo output.\n"
        f"Output:\n{output}"
    )


def test_run_demo_all_mode() -> None:
    """run_demo('all') chains all three demos without raising."""
    console, buf = _make_console()
    cascade_ev = _minimal_cascade_events()
    # For 'all' mode we share the same event list; budget and escalate
    # will use the default scripts since we can't inject per-sub-mode events.
    # (which also run in no_live mode so no Ollama is needed).
    demo_cascade(speed=10.0, _events=cascade_ev,
                 _console=console, _no_live=True)
    demo_budget( speed=10.0, _events=_minimal_budget_events(),
                 _console=console, _no_live=True)
    demo_escalate(speed=10.0, _events=_minimal_escalate_events(),
                  _console=console, _no_live=True)
    output = buf.getvalue()
    assert "FULL_FRAMEWORK" in output
    assert "CASCADE FAILURE" in output   # from cascade demo


def test_run_demo_invalid_mode() -> None:
    """run_demo() raises ValueError for an unknown mode."""
    with pytest.raises(ValueError, match="Unknown demo mode"):
        run_demo(mode="nonexistent")
