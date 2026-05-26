"""Integration tests verifying that condition wiring (FULL_FRAMEWORK,."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from mas.agents.base import AgentOutput, BaseAgent, TaskInput
from mas.kernel.budget import BudgetManager
from mas.kernel.graph import build_graph
from mas.kernel.registry import (
    AgentRegistry, AgentSpec, Capability, CostProfile, EscalationReason, Tier,
)
from mas.kernel.scheduler import Scheduler
from mas.kernel.state import make_initial_state
from mas.kernel.validator import System0


class GoodAgent(BaseAgent):
    """Always returns a schema-valid AgentOutput."""

    def run_internal(self, task: TaskInput) -> AgentOutput:
        return AgentOutput(
            agent_id  = self.agent_id,
            task_id   = task.task_id,
            answer    = "A valid, correct answer.",
            confidence = 0.90,
            reasoning  = "Solid reasoning, well within length limit.",
        )


class BadAgent(BaseAgent):
    """Returns an AgentOutput whose reasoning exceeds maxLength 4000.

    This cannot be repaired by the System-0 repair heuristic (the heuristic
    only replaces *empty* reasoning strings).  Therefore:
        schema_valid            = False
        would_have_been_invalid = True
    """

    def run_internal(self, task: TaskInput) -> AgentOutput:
        return AgentOutput(
            agent_id  = self.agent_id,
            task_id   = task.task_id,
            answer    = "Answer with schema violation.",
            confidence = 0.88,
            reasoning  = "x" * 5000,   # exceeds maxLength 4000 → fails schema
        )


def _build(
    agent_class,
    condition: str,
    budget: int = 10_000,
):
    """Assemble a minimal compiled graph with one agent and the given condition."""
    R    = AgentRegistry()
    spec = AgentSpec(
        agent_id     = "mock_agent",
        name         = "Mock",
        tier         = Tier.S2,
        model_name   = "llama3.1:8b",
        cost_profile = CostProfile(100, 1.0, 2.0, 8.0),
        capabilities = frozenset({Capability.REASONING}),
        strategy     = "react_loop",
    )
    R.register(spec)   # no agent class — we instantiate manually below

    bm        = BudgetManager(omega=budget)
    s0        = System0(confidence_threshold=0.72)
    scheduler = Scheduler(R, bm)

    # Manually instantiate so we don't need a real OllamaClient.
    agent_instance = agent_class(
        spec         = spec,
        ollama_client = None,
        registry     = R,
        tool_registry = None,
    )
    agents = {"mock_agent": agent_instance}

    graph   = build_graph(R, scheduler, s0, bm, agents)
    initial = make_initial_state("sess_test", "Test goal.", budget)
    return graph, initial


def _invoke(agent_class, condition: str, budget: int = 10_000) -> Dict[str, Any]:
    graph, initial = _build(agent_class, condition, budget)
    return graph.invoke(initial)


def test_full_framework_enforces():
    """
    Under FULL_FRAMEWORK, BadAgent's oversized reasoning fails System-0.
    The per-step log must record would_have_been_invalid=True and
    predicate_fired='P2_schema_invalid_unrepaired'.
    """
    final = _invoke(BadAgent, "FULL_FRAMEWORK")

    per_step = final.get("per_step_log", [])
    assert len(per_step) >= 1, "Expected at least one per-step entry"

    first = per_step[0]
    assert first["would_have_been_invalid"] is True, (
        "BadAgent output must fail schema validation"
    )
    assert first["predicate_fired"] == EscalationReason.P2_SCHEMA.value, (
        f"Expected P2 predicate, got: {first['predicate_fired']}"
    )
    assert first["schema_valid"] is False


def test_runs_jsonl_atomic_append(tmp_path: Path, monkeypatch):
    """
    Two sequential runs must produce exactly two newline-delimited JSON lines
    in runs.jsonl.  Each line must be valid JSON containing required fields.
    """
    log_path = tmp_path / "test_runs.jsonl"
    monkeypatch.setenv("MAS_RUNS_LOG", str(log_path))

    from mas.kernel import MASKernel
    from mas.tools.base import ToolRegistry

    R    = AgentRegistry()
    spec = AgentSpec(
        agent_id     = "good_agent",
        name         = "Good",
        tier         = Tier.S2,
        model_name   = "llama3.1:8b",
        cost_profile = CostProfile(100, 1.0, 2.0, 8.0),
        capabilities = frozenset({Capability.REASONING}),
        strategy     = "react_loop",
    )
    R.register(spec, GoodAgent)

    tool_reg = ToolRegistry(tmp_path)
    kernel   = MASKernel(
        registry  = R,
        tool_reg  = tool_reg,
        ollama    = None,
        budget    = 5_000,
        tau       = 0.72,
        condition = "FULL_FRAMEWORK",
    )

    kernel.run("First test goal.")
    kernel.run("Second test goal.")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, f"Expected 2 JSON lines in runs.jsonl, got {len(lines)}"

    required_keys = {
        "run_id", "condition", "started_at", "completed_at",
        "latency_s", "success", "per_step", "s0", "escalations",
        "model_slm", "model_llm", "tokens",
    }
    for i, line in enumerate(lines):
        entry = json.loads(line)   # raises if not valid JSON
        missing = required_keys - entry.keys()
        assert not missing, f"Line {i+1} missing keys: {missing}"
        assert entry["condition"] == "FULL_FRAMEWORK"
        assert isinstance(entry["per_step"], list)
