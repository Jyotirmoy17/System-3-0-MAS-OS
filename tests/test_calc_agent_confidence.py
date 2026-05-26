from __future__ import annotations

from unittest.mock import MagicMock

from mas.agents.s1.calc_agent import CalculatorAgent
from mas.agents.base import TaskInput
from mas.kernel.registry import AgentSpec, Capability, CostProfile, Tier


def test_calc_agent_fallback_coerces_string_confidence_high() -> None:
    spec = AgentSpec(
        agent_id="calc_agent",
        name="Calculator",
        tier=Tier.S1,
        model_name="phi3:mini",
        cost_profile=CostProfile(400, 5.0, 12.0, 1.0),
        capabilities=frozenset({Capability.MATH_ARITHMETIC, Capability.PYTHON_EXEC, Capability.DATA_ANALYSIS}),
        tools=["python_exec"],
        strategy="direct",
    )

    agent = CalculatorAgent(
        spec=spec,
        ollama_client=None,  # force fallback path
        registry=MagicMock(),
        tool_registry=MagicMock(),
    )

    # Force fallback to run by stubbing llm_json (no Ollama calls needed).
    agent.llm_json = MagicMock(return_value=({"answer": "42", "confidence": "high"}, 10))  # type: ignore[method-assign]

    out = agent._fallback(TaskInput(task_id="t1", goal="calc", query="2+2", context={}), ctx="")

    assert out.answer == "42"
    assert isinstance(out.confidence, float)
    assert 0.0 <= out.confidence <= 1.0
