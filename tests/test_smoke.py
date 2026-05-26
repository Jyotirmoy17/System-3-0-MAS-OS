"""Smoke tests: verify that the package graph imports cleanly and that."""
from __future__ import annotations


def test_imports_work() -> None:
    """All top-level packages import without errors."""
    import mas.kernel                          # noqa: F401
    import mas.kernel.registry                # noqa: F401
    import mas.kernel.validator               # noqa: F401
    import mas.kernel.scheduler               # noqa: F401
    import mas.kernel.budget                  # noqa: F401
    import mas.kernel.schemas                 # noqa: F401
    import mas.kernel.state                   # noqa: F401
    import mas.agents                         # noqa: F401
    import mas.agents.base                    # noqa: F401
    import mas.tools                          # noqa: F401
    import mas.tools.base                     # noqa: F401


def test_agent_registry_crud() -> None:
    """AgentRegistry register/find/unregister round-trips correctly."""
    from mas.kernel.registry import (
        AgentRegistry, AgentSpec, Capability, CostProfile, Tier,
    )

    R = AgentRegistry()
    spec = AgentSpec(
        agent_id     = "test_agent",
        name         = "Test Agent",
        tier         = Tier.S1,
        model_name   = "phi3:mini",
        cost_profile = CostProfile(200, 2.0, 5.0, 1.0),
        capabilities = frozenset({Capability.FACTUAL_QA}),
        tools        = [],
        strategy     = "direct",
    )
    R.register(spec)

    assert "test_agent" in R
    assert R.get_spec("test_agent") is spec
    results = R.find_capable({Capability.FACTUAL_QA})
    assert any(s.agent_id == "test_agent" for s in results)

    R.unregister("test_agent")
    assert "test_agent" not in R


def test_system0_valid_passthrough() -> None:
    """System0 passes a well-formed agent_response dict unchanged."""
    from mas.kernel.validator import System0

    s0   = System0(confidence_threshold=0.72)
    data = {"answer": "Paris", "confidence": 0.95, "reasoning": "Capital of France."}
    result = s0.validate_dict(data, schema_name="agent_response",
                              agent_id="test", task_id="t1")

    assert result.valid is True
    assert result.repaired is False
    assert result.data["answer"] == "Paris"
    assert s0.Vs() == 1.0


def test_system0_repairs_clamped_confidence() -> None:
    """System0 repairs confidence > 1.0 by clamping, not failing."""
    from mas.kernel.validator import System0
    from mas.kernel.registry import EscalationReason

    s0   = System0()
    data = {"answer": "Rome", "confidence": 1.5, "reasoning": "Capital of Italy."}
    result = s0.validate_dict(data, schema_name="agent_response",
                              agent_id="test", task_id="t2")

    assert result.valid is True
    assert result.repaired is True
    assert result.data["confidence"] <= 1.0
    assert result.predicate == EscalationReason.NONE


def test_budget_manager_tracking() -> None:
    """BudgetManager tracks spend and correctly reports exhaustion."""
    from mas.kernel.budget import BudgetManager

    bm = BudgetManager(omega=1000)
    assert not bm.is_exhausted()

    bm.record("agent_a", 600)
    assert bm.spent == 600
    assert bm.remaining() == 400

    bm.record("agent_b", 400)
    assert bm.is_exhausted()


def test_task_input_and_agent_output_dataclasses() -> None:
    """TaskInput and AgentOutput instantiate and serialise without error."""
    from mas.agents.base import AgentOutput, TaskInput
    from mas.kernel.registry import EscalationReason

    ti = TaskInput(task_id="t1", goal="Test goal", query="Test query")
    assert ti.depth == 0

    ao = AgentOutput(
        agent_id="calc_agent", task_id="t1",
        answer="42", confidence=0.99, reasoning="Computed.",
    )
    assert not ao.needs_escalation
    d = ao.to_dict()
    assert d["answer"] == "42"
    assert d["escalation_reason"] == EscalationReason.NONE.value


def test_conftest_fixtures(sample_run_log, tmp_runs_jsonl) -> None:
    """Conftest fixtures produce the expected shapes."""
    import json

    assert sample_run_log["condition"] == "FULL_FRAMEWORK"
    assert len(sample_run_log["per_step"]) == 4

    lines = tmp_runs_jsonl.read_text().splitlines()
    assert len(lines) == 2
    cascade = json.loads(lines[1])
    assert cascade["cascade_failure"] == 1
    assert any(s["would_have_been_invalid"] for s in cascade["per_step"])
