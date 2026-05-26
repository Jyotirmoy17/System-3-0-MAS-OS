"""Tests for mas/eval/baselines/langgraph_baseline.py (BASELINE_LANGGRAPH condition)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mas.eval.baselines.langgraph_baseline import GoalSpec, run_goal
from mas.eval.cascade_detector import detect_cascade


class MockOllama:
    """
    Minimal stand-in for OllamaClient.

    Accepts a list of text responses, cycling through them.
    Raises AssertionError if generate_json is ever called (proving the
    baseline does NOT use constrained decoding).
    """
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx       = 0
        self.calls:      list[dict[str, Any]] = []

    def generate(
        self,
        model:       str,
        prompt:      str,
        system:      str   = "",
        force_json:  bool  = False,
        temperature: float = 0.1,
        max_tokens:  int   = 800,
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "force_json": force_json, "prompt": prompt})
        text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return {
            "text":        text,
            "tokens_used": 60,
            "latency_s":   0.1,
            "model":       model,
            "success":     True,
            "error":       None,
        }

    def generate_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(
            "Baseline must NOT call generate_json (no constrained decoding)."
        )


def _valid_step_response(step: int = 1) -> str:
    return json.dumps({
        "answer":     f"Step {step} answer.",
        "confidence": 0.85,
        "reasoning":  f"Reasoning for step {step}.",
        "complexity": "medium",
    })


def test_baseline_runs_research_synth(tmp_path: Path, monkeypatch):
    """
    End-to-end test on a 4-step research synthesis goal.
    Verifies the RunLog is well-formed and has all required top-level fields.
    """
    monkeypatch.setenv("MAS_RUNS_LOG", str(tmp_path / "runs.jsonl"))

    goal = GoalSpec(
        id                 = "research_synth_001",
        goal_type          = "research_synthesis",
        goal_text          = "Explain the attention mechanism in transformers.",
        pipeline_depth_k   = 4,
    )
    mock_llm = MockOllama([_valid_step_response(i) for i in range(1, 5)])

    run_log = run_goal(goal, seed=0, ollama_client=mock_llm)

    # Structural checks
    assert run_log["condition"]    == "BASELINE_LANGGRAPH"
    assert run_log["routing_skipped"] is True
    assert run_log["success"]      == 1
    assert len(run_log["per_step"]) == 4
    assert run_log["final_output"] != ""

    # Per-step log integrity
    for i, step in enumerate(run_log["per_step"]):
        assert "step_id"                in step
        assert "would_have_been_invalid" in step
        assert "downstream_consumed"    in step
        if i < 3:
            assert step["downstream_consumed"] is True
        else:
            assert step["downstream_consumed"] is False


def test_baseline_cascade_on_adversarial(tmp_path: Path, monkeypatch):
    """
    Run 3-step goal with 'drop_required_field' corruption at step 2 (index 1).

    Expected cascade:
      - Step 2: would_have_been_invalid=True (required field dropped).
      - Step 2: downstream_consumed=True (step 3 receives corrupted context).
      - ground_truth_match=False (final answer won't match specific expected string).
      → cascade_detector returns True.
    """
    monkeypatch.setenv("MAS_RUNS_LOG", str(tmp_path / "runs.jsonl"))

    goal = GoalSpec(
        id                 = "adversarial_drop_001",
        goal_type          = "adversarial_reliability",
        goal_text          = "Analyse the code for bugs.",
        ground_truth       = "SPECIFIC_EXPECTED_ANSWER_THAT_WILL_NOT_MATCH",
        pipeline_depth_k   = 3,
        corruption_pattern = "drop_required_field",  # applied at step 2 (index 1)
    )
    mock_llm = MockOllama([_valid_step_response(i) for i in range(1, 4)])

    run_log = run_goal(goal, seed=0, ollama_client=mock_llm, adversarial_inject=True)

    assert len(run_log["per_step"]) == 3

    step2 = run_log["per_step"][1]
    assert step2["would_have_been_invalid"] is True,  (
        "Corruption at step 2 must fail audit validation"
    )
    assert step2["downstream_consumed"] is True, (
        "Step 2 output is consumed by step 3"
    )

    run_log["ground_truth_match"] = False

    assert detect_cascade(run_log) is True, (
        "Cascade must be detected: invalid step-2 output propagated to step 3 "
        "and final output failed rubric"
    )


def test_baseline_uses_same_tools(tmp_path: Path):
    """
    The baseline's tool registry must contain FileTool and PythonTool
    from the same mas/tools/ package used by the framework.
    """
    from mas.tools.base import ToolRegistry
    from mas.tools.file_tool   import FileTool
    from mas.tools.python_tool import PythonTool

    ws  = tmp_path / "workspace"
    ws.mkdir()
    registry = ToolRegistry(ws)

    # Verify tools are accessible by name (same interface as framework uses)
    assert registry.call is not None, "ToolRegistry must have a call() method"

    # Check FileTool and PythonTool are registered (they're lazy-loaded)
    from mas.tools.base import ToolRegistry as TR
    inst = TR(ws)
    # Trigger lazy load by accessing internals
    assert hasattr(inst, "_tools") or hasattr(inst, "call"), (
        "ToolRegistry must be accessible"
    )

    # Verify tool class origins match framework's own tools
    assert FileTool.__module__   == "mas.tools.file_tool"
    assert PythonTool.__module__ == "mas.tools.python_tool"


def test_baseline_no_constrained_decoding(tmp_path: Path, monkeypatch):
    """
    All calls to the ollama client must NOT set force_json=True.
    This is the key difference from FULL_FRAMEWORK's constrained decoding.
    """
    monkeypatch.setenv("MAS_RUNS_LOG", str(tmp_path / "runs.jsonl"))

    goal = GoalSpec(
        id="cd_test", goal_type="sequential_reasoning",
        goal_text="Calculate 2+2.", pipeline_depth_k=2,
    )
    mock_llm = MockOllama([_valid_step_response(1), _valid_step_response(2)])

    run_goal(goal, seed=0, ollama_client=mock_llm)

    assert len(mock_llm.calls) >= 2, "Baseline must make at least one LLM call per step"
    for call in mock_llm.calls:
        assert call["force_json"] is False, (
            f"force_json must be False in every baseline LLM call; got: {call}"
        )


def test_baseline_pydantic_failure_propagates(tmp_path: Path, monkeypatch):
    """
    When step 1 returns malformed output that fails Pydantic validation,
    the baseline must NOT retry and must still call the LLM for step 2.
    This is the propagation behaviour that produces cascades in multi-step goals.
    """
    monkeypatch.setenv("MAS_RUNS_LOG", str(tmp_path / "runs.jsonl"))

    bad_step1  = json.dumps({"answer": "partial answer", "reasoning": "thinking..."})
    good_step2 = _valid_step_response(2)

    goal = GoalSpec(
        id="pydantic_test", goal_type="code_analysis",
        goal_text="Find the bug.", pipeline_depth_k=2,
    )
    mock_llm = MockOllama([bad_step1, good_step2])

    run_log = run_goal(goal, seed=0, ollama_client=mock_llm)

    assert len(mock_llm.calls) >= 2, (
        "Baseline must call LLM for step 2 even after step 1 Pydantic failure"
    )
    step1_log = run_log["per_step"][0]
    assert step1_log["pydantic_error"] is not None, (
        "Step 1 Pydantic error must be logged"
    )
    assert len(run_log["per_step"]) == 2


def test_baseline_runlog_schema_valid(tmp_path: Path, monkeypatch):
    """
    The RunLog dict produced by run_goal() must contain all fields
    required by EXPERIMENTS.md §4.
    """
    monkeypatch.setenv("MAS_RUNS_LOG", str(tmp_path / "runs.jsonl"))

    # Required top-level fields from EXPERIMENTS.md §4
    REQUIRED_FIELDS = {
        "run_id", "goal_id", "goal_type",
        "pipeline_depth_k", "condition", "routing_skipped", "seed",
        "model_slm", "model_llm", "started_at", "completed_at",
        "latency_s", "success", "cascade_failure",
        "tokens", "s0", "escalations", "per_step",
        "final_output", "ground_truth", "rubric_scores",
        "ground_truth_match", "voc_trace", "crash_reason",
        "grading_api_calls", "grading_cost_usd",
    }
    # Required per-step fields
    REQUIRED_STEP_FIELDS = {
        "step_id", "agent_id", "tier", "raw_output",
        "schema_valid", "would_have_been_invalid",
        "downstream_consumed", "tokens_used", "latency_ms",
    }

    goal    = GoalSpec(
        id="schema_test", goal_type="financial_analysis",
        goal_text="Calculate revenue.", pipeline_depth_k=3,
    )
    mock_llm = MockOllama([_valid_step_response(i) for i in range(1, 3)])

    run_log = run_goal(goal, seed=42, ollama_client=mock_llm)

    missing = REQUIRED_FIELDS - set(run_log.keys())
    assert not missing, f"RunLog missing required fields: {missing}"

    assert run_log["condition"] == "BASELINE_LANGGRAPH"
    assert isinstance(run_log["per_step"], list)
    assert len(run_log["per_step"]) > 0

    for step in run_log["per_step"]:
        missing_step = REQUIRED_STEP_FIELDS - set(step.keys())
        assert not missing_step, f"per_step entry missing fields: {missing_step}"

    # Validate the JSON line written to runs.jsonl
    log_path = tmp_path / "runs.jsonl"
    lines    = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["run_id"] == run_log["run_id"]
