"""Verify that System-2 agents (ReActAgent, VerifierRetryAgent) never call."""
from __future__ import annotations

import inspect
import re
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from mas.agents.base import AgentOutput, TaskInput
from mas.kernel.registry import (
    AgentRegistry, AgentSpec, Capability, CostProfile, EscalationReason, Tier,
)


def _make_react_agent(ollama_client, registry=None, tool_registry=None):
    from mas.agents.s2.react import ReActAgent
    spec = AgentSpec(
        agent_id     = "react_agent",
        name         = "ReAct",
        tier         = Tier.S2,
        model_name   = "llama3.1:8b",
        cost_profile = CostProfile(800, 10.0, 30.0, 8.0),
        capabilities = frozenset({Capability.REACT_LOOP, Capability.REASONING}),
        strategy     = "react_loop",
    )
    return ReActAgent(
        spec=spec,
        ollama_client=ollama_client,
        registry=registry,
        tool_registry=tool_registry,
    )


def _make_verifier_agent(ollama_client, registry=None, tool_registry=None):
    from mas.agents.s2.verifier import VerifierRetryAgent
    spec = AgentSpec(
        agent_id     = "verifier_agent",
        name         = "Verifier",
        tier         = Tier.S2,
        model_name   = "llama3.1:8b",
        cost_profile = CostProfile(900, 12.0, 35.0, 8.0),
        capabilities = frozenset({Capability.VERIFICATION, Capability.REASONING}),
        strategy     = "verifier_retry",
    )
    return VerifierRetryAgent(
        spec=spec,
        ollama_client=ollama_client,
        registry=registry,
        tool_registry=tool_registry,
    )


def _web_agent_spec() -> AgentSpec:
    return AgentSpec(
        agent_id     = "web_agent",
        name         = "Web Agent",
        tier         = Tier.S1,
        model_name   = "phi3:mini",
        cost_profile = CostProfile(400, 8.0, 20.0, 1.0),
        capabilities = frozenset({Capability.WEB_SEARCH, Capability.RESEARCH, Capability.FACTUAL_QA}),
        tools        = ["web_search", "web_fetch"],
        strategy     = "direct",
    )


def _calc_agent_spec() -> AgentSpec:
    return AgentSpec(
        agent_id     = "calc_agent",
        name         = "Calculator",
        tier         = Tier.S1,
        model_name   = "phi3:mini",
        cost_profile = CostProfile(400, 5.0, 12.0, 1.0),
        capabilities = frozenset({Capability.MATH_ARITHMETIC, Capability.PYTHON_EXEC,
                                   Capability.DATA_ANALYSIS}),
        tools        = ["python_exec"],
        strategy     = "direct",
    )


def _canned_spawn_output(agent_id: str = "web_agent") -> AgentOutput:
    return AgentOutput(
        agent_id  = agent_id,
        task_id   = "t_sub",
        answer    = "Canned sub-agent answer.",
        confidence = 0.85,
        reasoning  = "Mocked S1 result.",
    )


def test_react_does_not_call_tools_directly(mock_ollama_client):
    """
    When the ReAct LLM requests a tool action, the agent must route through
    spawn() — not self.tools.call().  We verify this by making tools.call()
    raise and confirming no exception is raised.
    """
    # Registry with a capable S1 agent so the lookup succeeds
    R = AgentRegistry()
    R.register(_web_agent_spec())

    # Tool registry that raises if .call() is ever invoked
    mock_tools = MagicMock()
    mock_tools.call.side_effect = AssertionError(
        "ReActAgent called self.tools.call() directly — Hard Invariant #1 violated"
    )

    mock_ollama_client.generate_json.side_effect = [
        {
            "success": True,
            "parsed": {"thought": "I need web search", "action": "web_search",
                        "action_input": {"query": "transformer attention"}},
            "tokens_used": 30,
        },
        {
            "success": True,
            "parsed": {"answer": "Attention is all you need.",
                        "confidence": 0.9, "reasoning": "Found it.", "complexity": "medium"},
            "tokens_used": 25,
        },
    ]

    agent = _make_react_agent(mock_ollama_client, registry=R, tool_registry=mock_tools)
    task  = TaskInput(task_id="t1", goal="Research transformers", query="transformer attention")

    # spawn() is mocked so no actual S1 LLM call is needed
    with patch.object(agent, "spawn", return_value=_canned_spawn_output("web_agent")):
        out = agent.run(task)

    mock_tools.call.assert_not_called()
    assert out.answer != ""


def test_verifier_does_not_call_tools_directly(mock_ollama_client):
    """
    VerifierRetryAgent's numeric verification must route through calc_agent
    (spawn), not self.tools.call("python_exec", ...).
    """
    mock_tools = MagicMock()
    mock_tools.call.side_effect = AssertionError(
        "VerifierRetryAgent called self.tools.call() directly — Hard Invariant #1 violated"
    )

    # Initial generation returns a numeric answer to trigger the verify path
    mock_ollama_client.generate_json.side_effect = [
        {
            "success": True,
            "parsed": {"answer": "42", "confidence": 0.9,
                        "reasoning": "Computed.", "complexity": "medium"},
            "tokens_used": 30,
        },
        {
            "success": True,
            "parsed": {"is_correct": True, "confidence": 0.95,
                        "error_description": "", "corrected_answer": ""},
            "tokens_used": 20,
        },
    ]

    agent = _make_verifier_agent(mock_ollama_client, tool_registry=mock_tools)
    task  = TaskInput(task_id="t2", goal="Calculate 6*7", query="What is 6 times 7?")

    with patch.object(agent, "spawn", return_value=_canned_spawn_output("calc_agent")):
        out = agent.run(task)

    mock_tools.call.assert_not_called()
    assert out.answer != ""


def test_react_spawn_goes_through_s0(mock_ollama_client):
    """
    After spawning a S1 sub-agent, ReActAgent must validate the output
    through System-0 (_S0.validate_dict) before inserting the observation
    into the ReAct history.
    """
    import mas.agents.s2.react as react_module

    R = AgentRegistry()
    R.register(_web_agent_spec())

    mock_ollama_client.generate_json.side_effect = [
        {
            "success": True,
            "parsed": {"thought": "search needed", "action": "web_search",
                        "action_input": {"query": "test"}},
            "tokens_used": 20,
        },
        {
            "success": True,
            "parsed": {"answer": "Done.", "confidence": 0.9,
                        "reasoning": "ok", "complexity": "easy"},
            "tokens_used": 15,
        },
    ]

    agent = _make_react_agent(mock_ollama_client, registry=R)
    task  = TaskInput(task_id="t3", goal="Test S0", query="test query")

    validate_calls: list = []
    original = react_module._S0.validate_dict

    def _track(*args: Any, **kwargs: Any):
        validate_calls.append(args)
        return original(*args, **kwargs)

    with patch.object(react_module._S0, "validate_dict", side_effect=_track):
        with patch.object(agent, "spawn", return_value=_canned_spawn_output("web_agent")):
            agent.run(task)

    assert len(validate_calls) >= 1, (
        "_S0.validate_dict was never called — spawn result is not being validated "
        "by System-0 before re-entering the ReAct loop."
    )


def test_react_capability_lookup():
    """
    AgentRegistry.cheapest_capable() must return the correct S1 specialist
    for each tool capability used in _TOOL_CAPABILITY.
    """
    from mas.agents.s2.react import _TOOL_CAPABILITY

    R = AgentRegistry()
    R.register(_web_agent_spec())
    R.register(_calc_agent_spec())

    cap_web = _TOOL_CAPABILITY["web_search"]
    found   = R.cheapest_capable({cap_web}, tier=Tier.S1)
    assert found is not None
    assert found.agent_id == "web_agent"

    cap_py = _TOOL_CAPABILITY["python_exec"]
    found  = R.cheapest_capable({cap_py}, tier=Tier.S1)
    assert found is not None
    assert found.agent_id == "calc_agent"

    cap_file = _TOOL_CAPABILITY["file_read"]
    found    = R.cheapest_capable({cap_file}, tier=Tier.S1)
    assert found is None


def test_invariant_documented():
    """
    Grep the source of both S2 agents for any direct self.tools.call() usage.
    This is the definitive invariant check: if this test fails, the fix was
    incomplete or has regressed.
    """
    import mas.agents.s2.react    as react_module
    import mas.agents.s2.verifier as verifier_module

    react_src    = inspect.getsource(react_module)
    verifier_src = inspect.getsource(verifier_module)

    pattern = r"self\.tools\.call\("

    react_matches    = re.findall(pattern, react_src)
    verifier_matches = re.findall(pattern, verifier_src)

    assert not react_matches, (
        f"Found {len(react_matches)} direct self.tools.call() in react.py — "
        "Hard Invariant #1 violated"
    )
    assert not verifier_matches, (
        f"Found {len(verifier_matches)} direct self.tools.call() in verifier.py — "
        "Hard Invariant #1 violated"
    )
