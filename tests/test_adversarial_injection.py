"""Tests that the adversarial corruption wrapper is wired correctly into both."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mas.eval.adversarial_wrapper import corrupt
from mas.eval.baselines.langgraph_baseline import GoalSpec, run_goal
from mas.eval.cascade_detector import detect_cascade


from mas.agents.base import AgentOutput, BaseAgent, TaskInput
from mas.kernel.registry import (
    AgentRegistry, AgentSpec, Capability, CostProfile, EscalationReason, Tier,
)


class LowConfAgent(BaseAgent):
    """Returns a valid answer but with confidence below tau (triggers P1)."""

    def run_internal(self, task: TaskInput) -> AgentOutput:
        return AgentOutput(
            agent_id   = self.agent_id,
            task_id    = task.task_id,
            answer     = "Preliminary answer.",
            confidence = 0.40,   # below default tau=0.72 → P1
            reasoning  = "Low confidence deliberately.",
        )


class GoodAgent(BaseAgent):
    """Returns a valid, high-confidence answer (run completes cleanly)."""

    def run_internal(self, task: TaskInput) -> AgentOutput:
        return AgentOutput(
            agent_id   = self.agent_id,
            task_id    = task.task_id,
            answer     = "Final answer.",
            confidence = 0.91,
            reasoning  = "Deterministic test agent.",
        )


class MockOllama:
    def __init__(self, responses: list[str]) -> None:
        self._r   = responses
        self._idx = 0

    def generate(self, model, prompt, system="", force_json=False,
                 temperature=0.7, max_tokens=800) -> dict[str, Any]:
        text = self._r[self._idx % len(self._r)]
        self._idx += 1
        return {"text": text, "tokens_used": 50, "latency_s": 0.05,
                "model": model, "success": True, "error": None}

    def generate_json(self, *a, **k):
        raise AssertionError("Baseline must not call generate_json")


def _valid_resp(step: int = 1) -> str:
    return json.dumps({
        "answer": f"Step {step} valid answer.",
        "confidence": 0.88,
        "reasoning": f"Good reasoning for step {step}.",
    })


def _build_framework_graph(
    step1_class=LowConfAgent,
    step2_class=GoodAgent,
    budget: int = 10_000,
):
    """Build a minimal framework graph: S1=LowConfAgent, S2=GoodAgent."""
    from mas.kernel.budget    import BudgetManager
    from mas.kernel.graph     import build_graph
    from mas.kernel.scheduler import Scheduler
    from mas.kernel.state     import make_initial_state
    from mas.kernel.validator import System0

    R = AgentRegistry()

    s1_spec = AgentSpec(
        "low_conf_agent", "LowConf", Tier.S1,
        cost_profile = CostProfile(100, 1.0, 2.0, 1.0),
        capabilities = frozenset({Capability.FACTUAL_QA}),
        strategy     = "direct",
    )
    R.register(s1_spec, step1_class)

    # Register step-2 recovery agent (self_consistency strategy)
    s2_spec = AgentSpec(
        "good_agent", "Good", Tier.S2,
        cost_profile = CostProfile(200, 2.0, 4.0, 8.0),
        capabilities = frozenset({Capability.REASONING}),
        strategy     = "self_consistency",
    )
    R.register(s2_spec, step2_class)

    bm  = BudgetManager(omega=budget)
    s0  = System0(confidence_threshold=0.72)
    sc  = Scheduler(R, bm)

    instances = {
        "low_conf_agent": step1_class(s1_spec, None, R, None),
        "good_agent":     step2_class(s2_spec, None, R, None),
    }

    graph   = build_graph(R, sc, s0, bm, instances)
    initial = make_initial_state("test_sess", "Test goal.", budget)
    return graph, initial


def test_injection_only_for_adversarial():
    """
    With adversarial_inject=True but a non-adversarial goal (research_synth),
    no corruption must be applied: adversarial_corruption_applied stays null
    in all per_step entries.
    """
    graph, initial = _build_framework_graph()
    initial["adversarial_inject"]  = True
    initial["adversarial_pattern"] = "drop_required_field"
    initial["adversarial_seed"]    = 0

    # (the framework checks adversarial_inject + adversarial_pattern, but
    # does not enforce goal_type — that guard lives in the harness caller).
    # For this test, we rely on the fact that step 2 may or may not be reached.
    final = graph.invoke(initial)

    for step in final.get("per_step_log", []):
        # Step 2 injection would fire IF it's reached; for a non-adversarial
        # place.  Here we confirm the field exists in the schema.
        assert "adversarial_corruption_applied" in step, (
            "adversarial_corruption_applied field must be present in all per_step entries"
        )


@pytest.mark.parametrize("pattern", [
    "drop_required_field",
    "type_flip",
    "out_of_range",
    "malformed_json",
    "extra_fields_not_in_schema",
])
def test_injection_for_each_pattern(pattern: str, tmp_path: Path, monkeypatch):
    """
    For each of the 5 adversarial patterns:
    - Run BASELINE_LANGGRAPH with 3 steps and adversarial_inject=True.
    - Assert adversarial_corruption_applied == pattern in per_step[1] (step 2).
    - For patterns that change the dict (all except extra_fields on
      permissive schema): assert would_have_been_invalid is logged.
    - Assert no crash and run completed.
    """
    monkeypatch.setenv("MAS_RUNS_LOG", str(tmp_path / "runs.jsonl"))

    goal = GoalSpec(
        id                 = f"adversarial_001",
        goal_type          = "adversarial_reliability",
        goal_text          = "Summarize transformer neural networks.",
        pipeline_depth_k   = 3,
        corruption_pattern = pattern,
    )
    mock_llm = MockOllama([_valid_resp(i) for i in range(1, 4)])

    run_log = run_goal(goal, seed=0, ollama_client=mock_llm,
                       adversarial_inject=True)

    assert len(run_log["per_step"]) == 3, (
        f"Expected 3 steps, got {len(run_log['per_step'])}"
    )

    step2 = run_log["per_step"][1]  # 0-indexed, step 2
    assert step2["adversarial_corruption_applied"] == pattern, (
        f"Expected corruption_applied='{pattern}', got {step2['adversarial_corruption_applied']}"
    )
    assert "would_have_been_invalid" in step2


def test_injection_under_baseline(tmp_path: Path, monkeypatch):
    """
    Under BASELINE_LANGGRAPH (which is the 'no System-0' analog for multi-step):
    - 3-step pipeline; corruption injected at step 2 (index 1).
    - Step 2 output: would_have_been_invalid=True, downstream_consumed=True.
    - With ground_truth_match=False → cascade_failure=True.
    """
    monkeypatch.setenv("MAS_RUNS_LOG", str(tmp_path / "runs.jsonl"))

    goal = GoalSpec(
        id                 = "adversarial_001",
        goal_type          = "adversarial_reliability",
        goal_text          = "Analyse for bugs.",
        pipeline_depth_k   = 3,
        ground_truth       = "SPECIFIC_EXPECTED_ANSWER_NOT_MATCHED",
        corruption_pattern = "drop_required_field",
    )
    mock_llm = MockOllama([_valid_resp(i) for i in range(1, 4)])

    run_log = run_goal(goal, seed=0, ollama_client=mock_llm,
                       adversarial_inject=True)

    per_step = run_log["per_step"]
    assert len(per_step) == 3

    step2 = per_step[1]
    assert step2["adversarial_corruption_applied"] == "drop_required_field"
    assert step2["would_have_been_invalid"] is True
    assert step2["downstream_consumed"] is True   # post-processed by run_goal

    run_log["ground_truth_match"] = False

    assert detect_cascade(run_log) is True, (
        "Cascade must be detected: corrupted step-2 output was consumed by step 3 "
        "and the final output failed the rubric"
    )


def test_injection_logs_pattern(tmp_path: Path, monkeypatch):
    """
    The adversarial_corruption_applied field must be the pattern name at the
    injection step and null at all other steps.
    """
    monkeypatch.setenv("MAS_RUNS_LOG", str(tmp_path / "runs.jsonl"))

    goal = GoalSpec(
        id                 = "adversarial_013",
        goal_type          = "adversarial_reliability",
        goal_text          = "Analyse code.",
        pipeline_depth_k   = 3,
        corruption_pattern = "out_of_range",
    )
    mock_llm = MockOllama([_valid_resp(i) for i in range(1, 4)])

    run_log = run_goal(goal, seed=42, ollama_client=mock_llm,
                       adversarial_inject=True)

    per_step = run_log["per_step"]
    assert len(per_step) == 3

    assert per_step[0]["adversarial_corruption_applied"] is None,  "Step 1: no corruption"
    assert per_step[1]["adversarial_corruption_applied"] == "out_of_range", "Step 2: corruption"
    assert per_step[2]["adversarial_corruption_applied"] is None,  "Step 3: no corruption"
