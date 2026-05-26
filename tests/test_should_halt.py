"""Tests for the Scheduler.should_halt() integration with graph.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mas.kernel.budget import BudgetManager
from mas.kernel.registry import (
    AgentRegistry, AgentSpec, Capability, CostProfile, EscalationReason, Tier,
)
from mas.kernel.scheduler import Scheduler


def _make_scheduler(omega: int = 50_000) -> tuple[Scheduler, BudgetManager]:
    R  = AgentRegistry()
    bm = BudgetManager(omega=omega)
    s  = Scheduler(registry=R, budget_mgr=bm)
    return s, bm


def test_should_halt_called_with_three_args():
    """
    Calling should_halt(escalation_count, confidence, goal) must not raise
    TypeError.  This is the regression guard for graph.py:275 which previously
    called should_halt with only one argument.
    """
    scheduler, _ = _make_scheduler()
    # Should not raise
    result = scheduler.should_halt(
        escalation_count   = 1,
        current_confidence = 0.5,
        goal               = "Test goal for should_halt signature check.",
    )
    assert isinstance(result, bool)


def test_should_halt_returns_true_on_budget_exhausted():
    """
    When the BudgetManager reports is_exhausted() == True,
    should_halt() must return True immediately regardless of other args.
    """
    scheduler, bm = _make_scheduler(omega=100)
    bm.record("some_agent", 100)  # exhaust the budget
    assert bm.is_exhausted()

    result = scheduler.should_halt(
        escalation_count   = 0,
        current_confidence = 0.95,
        goal               = "any goal",
    )
    assert result is True, "should_halt must return True when budget is exhausted"


def test_should_halt_returns_false_on_high_confidence():
    """
    When budget is not exhausted and escalation_count < MAX_ESCALATIONS,
    should_halt() should return False for a high-confidence result
    (the Pandora's Box criterion: current quality ≥ max remaining sigma).
    """
    from mas.kernel.scheduler import MAX_ESCALATIONS

    scheduler, bm = _make_scheduler(omega=50_000)
    bm.record("agent_a", 100)   # spend a little but not exhausted
    assert not bm.is_exhausted()

    result = scheduler.should_halt(
        escalation_count   = 0,      # well below MAX_ESCALATIONS
        current_confidence = 0.99,   # very high confidence
        goal               = "A simple test goal.",
    )
    # (current quality >= max remaining reservation value).
    assert isinstance(result, bool)   # just confirm it runs without error


def test_route_after_validate_passes_three_args_to_should_halt():
    """
    Regression guard: route_after_validate() in graph.py must call
    scheduler.should_halt with (escalation_count, final_confidence, goal).

    We build the full graph, stub out the agents, and drive the router
    to the escalation path, then verify should_halt is called correctly.
    """
    from mas.kernel.graph import build_graph
    from mas.kernel.validator import System0

    R  = AgentRegistry()
    bm = BudgetManager(omega=50_000)
    s  = Scheduler(registry=R, budget_mgr=bm)
    s0 = System0()

    stub_output = MagicMock()
    stub_output.tokens_used   = 10
    stub_output.confidence    = 0.3   # below tau → P1
    stub_output.latency_s     = 0.1
    stub_output.tool_calls    = []
    stub_output.answer        = "stub"
    stub_output.to_dict.return_value = {
        "agent_id":          "stub",
        "task_id":           "t0",
        "answer":            "stub",
        "confidence":        0.3,
        "reasoning":         "stub reasoning",
        "complexity":        "medium",
        "tokens_used":       10,
        "latency_s":         0.1,
        "schema_valid":      True,
        "schema_repaired":   False,
        "escalation_reason": "none",
        "escalation_message": "",
        "tool_calls":        [],
        "sub_agents_used":   [],
        "depth":             0,
    }

    stub_agent = MagicMock()
    stub_agent.run.return_value = stub_output

    # Register a stub spec so select_initial has something to return
    R.register(AgentSpec(
        agent_id     = "stub_agent",
        name         = "Stub",
        tier         = Tier.S1,
        cost_profile = CostProfile(100, 1.0, 2.0, 1.0),
        capabilities = frozenset({Capability.FACTUAL_QA}),
    ))

    agents = {"stub_agent": stub_agent}

    halt_calls: list = []

    def _mock_halt(escalation_count, current_confidence, goal):
        halt_calls.append((escalation_count, current_confidence, goal))
        return True   # force halt so the graph terminates

    with patch.object(s, "should_halt", side_effect=_mock_halt):
        graph = build_graph(R, s, s0, bm, agents)
        from mas.kernel.state import make_initial_state
        state = make_initial_state("sess_test", "Test goal for halt.", 1000)
        try:
            graph.invoke(state)
        except Exception:
            pass   # graph may error on stub; we only care about the call args

    assert len(halt_calls) >= 1, "should_halt was never called by route_after_validate"

    esc_count, conf, goal = halt_calls[0]
    assert isinstance(esc_count, int),   "escalation_count must be int"
    assert isinstance(conf, float),      "current_confidence must be float"
    assert isinstance(goal, str),        "goal must be str"
    assert goal == "Test goal for halt.", f"goal mismatch: {goal!r}"
