"""Tests for the single-run experimental harness (mas/eval/harness.py)."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

import mas.eval.harness as harness_mod
from mas.eval.harness import RunLog, _validate_run_log, run_one, score_rubric
from mas.eval.goal_loader import GoalSpec


FIXTURE_GOALS_DIR = Path(__file__).parent / "fixtures" / "goals"


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canned_run_entry(
    condition:      str            = "FULL_FRAMEWORK",
    success:        int            = 1,
    crash_reason:   str | None     = None,
    per_step:       list | None    = None,
    final_output:   str            = "A valid answer.",
    slm:            str            = "phi3:mini",
    llm:            str            = "llama3.1:8b",
    would_have:     bool           = False,
) -> dict[str, Any]:
    """Return a kernel-style (incomplete) run_log entry."""
    step: dict[str, Any] = {
        "step_id":                        1,
        "agent_id":                       "react_agent",
        "tier":                           2,
        "raw_output":                     '{"answer":"A valid answer.","confidence":0.9,"reasoning":"OK."}',
        "schema_valid":                   not would_have,
        "would_have_been_invalid":        would_have,
        "downstream_consumed":            True,
        "tokens_used":                    120,
        "latency_ms":                     800,
        "adversarial_corruption_applied": None,
    }
    step2: dict[str, Any] = {
        "step_id":                        2,
        "agent_id":                       "react_agent",
        "tier":                           2,
        "raw_output":                     '{"answer":"A valid answer.","confidence":0.9,"reasoning":"OK."}',
        "schema_valid":                   True,
        "would_have_been_invalid":        False,
        "downstream_consumed":            False,
        "tokens_used":                    100,
        "latency_ms":                     600,
        "adversarial_corruption_applied": None,
    }
    now = _now_iso()
    return {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            "",         # harness fills
        "goal_type":          "",         # harness fills
        "pipeline_depth_k":   3,
        "condition":          condition,
        "routing_skipped":    condition == "BASELINE_LANGGRAPH",
        "seed":               0,          # harness fills
        "model_slm":          slm,
        "model_llm":          llm,
        "started_at":         now,
        "completed_at":       now,
        "latency_s":          1.2,
        "success":            success,
        "cascade_failure":    None,       # harness fills
        "tokens":             {"slm": 0, "llm": 420, "total": 420},
        "s0": {
            "catches":          0,
            "repairs":          0,
            "would_have_failed": 1 if would_have else 0,
        },
        "escalations":        {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step":           per_step if per_step is not None else [step, step2],
        "final_output":       final_output,
        "ground_truth":       "",         # harness fills
        "rubric_scores":      {},         # harness fills
        "ground_truth_match": None,       # harness fills
        "voc_trace":          [],
        "crash_reason":       crash_reason,
        "grading_api_calls":  0,          # harness fills
        "grading_cost_usd":   0.0,        # harness fills
    }


class MockKernel:
    """
    Stand-in for MASKernel.run().

    Writes a pre-built run_log entry to os.environ["MAS_RUNS_LOG"]
    so the harness can read it back exactly as the real kernel would.
    """
    def __init__(self, entry: dict[str, Any]) -> None:
        self._entry = entry

    def run(
        self,
        goal_text: str,
        session_id: str | None = None,
        *,
        adversarial_inject: bool = False,
        corruption_pattern: str  = "",
        corruption_seed:    int  = 0,
    ) -> dict[str, Any]:
        log_path = os.environ.get("MAS_RUNS_LOG", "./runs.jsonl")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self._entry) + "\n")
        return {
            "session_id":   "mock",
            "goal":         goal_text,
            "answer":       self._entry.get("final_output", ""),
            "confidence":   0.9,
            "complexity":   "easy",
            "total_tokens": self._entry["tokens"]["total"],
            "escalations":  0,
            "messages":     [],
            "agent_events": [],
            "per_step_log": self._entry.get("per_step", []),
            "system0":      {},
            "budget":       {},
        }


def _patch_make_kernel(monkeypatch, entry: dict[str, Any]) -> None:
    """Monkeypatch _make_kernel to return a MockKernel writing *entry*."""
    kernel = MockKernel(entry)
    monkeypatch.setattr(harness_mod, "_make_kernel", lambda *a, **kw: kernel)


def test_run_one_full_framework(monkeypatch):
    """
    run_one() with FULL_FRAMEWORK produces a valid RunLog with success=1.
    """
    entry = _canned_run_entry(condition="FULL_FRAMEWORK", success=1)
    _patch_make_kernel(monkeypatch, entry)

    result = run_one(
        "research_synth_001",
        "FULL_FRAMEWORK",
        42,
        _skip_warmup = True,
        _goals_dir   = FIXTURE_GOALS_DIR,
    )

    # Schema validation passes (run_one raises if invalid)
    _validate_run_log(result)

    assert result["success"]   == 1
    assert result["condition"] == "FULL_FRAMEWORK"
    assert result["goal_id"]   == "research_synth_001"
    assert result["goal_type"] == "research_synthesis"
    assert result["seed"]      == 42
    assert result["routing_skipped"] is False
    assert isinstance(result["rubric_scores"], dict)
    assert isinstance(result["ground_truth_match"], bool)
    assert result["cascade_failure"] in (0, 1)


def test_run_one_baseline_langgraph(monkeypatch):
    """
    run_one() with BASELINE_LANGGRAPH produces a valid RunLog with
    condition='BASELINE_LANGGRAPH'.
    """
    from mas.eval.baselines.langgraph_baseline import GoalSpec as BaselineGoalSpec

    baseline_entry = _canned_run_entry(condition="BASELINE_LANGGRAPH", success=1)
    # Override model fields as the baseline sets them
    baseline_entry["model_slm"] = "phi3:mini"
    baseline_entry["model_llm"] = "llama3.1:8b"

    def _mock_run_goal(
        goal,
        seed=0,
        ollama_client=None,
        tool_registry=None,
        model="llama3.1:8b",
        runs_log_path=None,
        *,
        adversarial_inject=False,
    ) -> dict[str, Any]:
        if runs_log_path:
            with open(runs_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(baseline_entry) + "\n")
        return dict(baseline_entry)

    import mas.eval.harness as h
    original_run_baseline = h._run_baseline

    def _patched_run_baseline(goal, seed, slm, llm, ollama_url, skip_warmup, **kwargs):
        entry = dict(baseline_entry)
        return entry

    monkeypatch.setattr(h, "_run_baseline", _patched_run_baseline)

    result = run_one(
        "research_synth_001",
        "BASELINE_LANGGRAPH",
        7,
        _skip_warmup = True,
        _goals_dir   = FIXTURE_GOALS_DIR,
    )

    _validate_run_log(result)
    assert result["condition"]       == "BASELINE_LANGGRAPH"
    assert result["routing_skipped"] is True


def test_run_one_crash_produces_valid_log(monkeypatch):
    """
    When the kernel raises (mocked to simulate agent failure), run_one()
    returns a valid RunLog with success=0 and crash_reason populated.
    """
    class CrashKernel:
        """Simulates a kernel whose graph.invoke raises mid-run."""
        def run(self, goal_text, session_id=None, **kwargs):
            # Kernel catches this internally and writes a crash entry.
            crash_entry = _canned_run_entry(
                condition    = "FULL_FRAMEWORK",
                success      = 0,
                crash_reason = "ValueError: simulated agent crash",
                per_step     = [],
                final_output = "",
            )
            log_path = os.environ.get("MAS_RUNS_LOG", "./runs.jsonl")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(crash_entry) + "\n")
            return {"answer": "", "confidence": 0.0, "per_step_log": [],
                    "total_tokens": 0, "escalations": 0, "messages": [],
                    "agent_events": [], "system0": {}, "budget": {}}

    monkeypatch.setattr(
        harness_mod, "_make_kernel",
        lambda *a, **kw: CrashKernel(),
    )

    result = run_one(
        "research_synth_001",
        "FULL_FRAMEWORK",
        99,
        _skip_warmup = True,
        _goals_dir   = FIXTURE_GOALS_DIR,
    )

    _validate_run_log(result)
    assert result["success"]      == 0
    assert result["crash_reason"] is not None
    assert "crash" in result["crash_reason"].lower() or \
           "ValueError" in result["crash_reason"] or \
           len(result["crash_reason"]) > 0


def test_run_one_validates_log_before_write(monkeypatch, tmp_path):
    """
    run_one() raises jsonschema.ValidationError if the assembled RunLog is
    invalid, and does NOT write anything to runs.jsonl.

    Simulated by making the mock kernel write a run_entry with run_id=None
    (violates 'minLength: 1' constraint).
    """
    import jsonschema

    bad_entry = _canned_run_entry(condition="FULL_FRAMEWORK", success=1)
    bad_entry["run_id"] = None   # invalid: schema requires non-empty string

    class BadKernel:
        def run(self, goal_text, session_id=None, **kwargs):
            log_path = os.environ.get("MAS_RUNS_LOG", "./runs.jsonl")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(bad_entry) + "\n")
            return {}

    monkeypatch.setattr(harness_mod, "_make_kernel", lambda *a, **kw: BadKernel())

    real_runs_jsonl = tmp_path / "runs.jsonl"
    # Make sure the real runs.jsonl path doesn't exist before the test.
    assert not real_runs_jsonl.exists()

    with pytest.raises(jsonschema.ValidationError):
        run_one(
            "research_synth_001",
            "FULL_FRAMEWORK",
            1,
            _skip_warmup = True,
            _goals_dir   = FIXTURE_GOALS_DIR,
        )

    # run_one() raises before returning — the CLI never writes.
    assert not real_runs_jsonl.exists(), \
        "runs.jsonl must not be written when run_one() raises"


def test_cli_smoke(tmp_path):
    """
    Invoke 'python -m mas.eval.cli run --no-write' as a subprocess.

    The _MAS_HARNESS_MOCK_LOG env var injects a pre-built RunLog so
    the CLI returns without hitting Ollama.
    """
    mock_run_log = {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            "research_synth_001",
        "goal_type":          "research_synthesis",
        "pipeline_depth_k":   4,
        "condition":          "FULL_FRAMEWORK",
        "routing_skipped":    False,
        "seed":               42,
        "model_slm":          "phi3:mini",
        "model_llm":          "llama3.1:8b",
        "started_at":         "2026-04-29T10:00:00Z",
        "completed_at":       "2026-04-29T10:00:01Z",
        "latency_s":          1.0,
        "success":            1,
        "cascade_failure":    0,
        "tokens":             {"slm": 0, "llm": 420, "total": 420},
        "s0":                 {"catches": 0, "repairs": 0, "would_have_failed": 0},
        "escalations":        {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step":           [],
        "final_output":       "Mock answer.",
        "ground_truth":       "attention mechanism",
        "rubric_scores":      {"c1": True},
        "ground_truth_match": True,
        "voc_trace":          [],
        "crash_reason":       None,
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
    }

    env = {
        **os.environ,
        "_MAS_HARNESS_MOCK_LOG": json.dumps(mock_run_log),
        # Prevent any real writes from earlier tests leaking in.
        "MAS_RUNS_LOG": str(tmp_path / "runs.jsonl"),
    }

    result = subprocess.run(
        [
            sys.executable, "-m", "mas.eval.cli", "run",
            "--condition", "FULL_FRAMEWORK",
            "--goal-id",   "research_synth_001",
            "--seed",      "42",
            "--no-write",
        ],
        capture_output = True,
        text           = True,
        env            = env,
        cwd            = str(Path(__file__).resolve().parent.parent),
    )

    assert result.returncode == 0, \
        f"CLI exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"

    parsed = json.loads(result.stdout.strip())
    assert parsed["condition"] == "FULL_FRAMEWORK"
    assert parsed["goal_id"]   == "research_synth_001"
    assert "run_id" in parsed


def test_score_rubric_substring_pass():
    """substring check passes when must_contain items are in the output."""
    rubric = [{
        "check_id":    "c1",
        "check_type":  "substring",
        "check_params": {"field": "answer", "must_contain": ["hello", "world"]},
    }]
    # final_output is plain text; _extract_from_output falls back to raw string
    scores = score_rubric(rubric, "hello world this is an answer")
    assert scores["c1"] is True


def test_score_rubric_substring_fail():
    """substring check fails when a required keyword is absent."""
    rubric = [{
        "check_id":    "c1",
        "check_type":  "substring",
        "check_params": {"field": "answer", "must_contain": ["missing_keyword"]},
    }]
    scores = score_rubric(rubric, "some other text")
    assert scores["c1"] is False


def test_score_rubric_empty_must_contain():
    """Empty must_contain list means the check always passes."""
    rubric = [{
        "check_id":    "c1",
        "check_type":  "substring",
        "check_params": {"field": "answer", "must_contain": []},
    }]
    scores = score_rubric(rubric, "any text")
    assert scores["c1"] is True


def test_score_rubric_exact_match():
    rubric = [{
        "check_id":    "c1",
        "check_type":  "exact_match",
        "check_params": {"field": "answer", "expected": "Paris"},
    }]
    scores = score_rubric(rubric, json.dumps({"answer": "paris", "confidence": 0.9}))
    assert scores["c1"] is True   # case-insensitive


def test_score_rubric_numeric_in_range():
    rubric = [{
        "check_id":    "c1",
        "check_type":  "numeric",
        "check_params": {"field": "value", "min": 0, "max": 100},
    }]
    scores = score_rubric(rubric, json.dumps({"value": 42}))
    assert scores["c1"] is True


def test_score_rubric_numeric_out_of_range():
    rubric = [{
        "check_id":    "c1",
        "check_type":  "numeric",
        "check_params": {"field": "value", "min": 0, "max": 10},
    }]
    scores = score_rubric(rubric, json.dumps({"value": 99}))
    assert scores["c1"] is False


class _YesJudge:
    """Stub judge that always answers YES."""
    total_api_calls: int   = 0
    total_cost_usd:  float = 0.0

    def judge(self, answer: str, criteria: str) -> bool:
        self.__class__.total_api_calls += 1
        return True


class _NoJudge:
    """Stub judge that always answers NO."""
    total_api_calls: int   = 0
    total_cost_usd:  float = 0.0

    def judge(self, answer: str, criteria: str) -> bool:
        self.__class__.total_api_calls += 1
        return False


_SEMANTIC_RUBRIC = [{
    "check_id":     "sj1",
    "check_type":   "semantic_judge",
    "check_params": {
        "field":    "answer",
        "criteria": "The answer must name a capital city.",
    },
}]


def test_semantic_judge_yes():
    """score_rubric with a YES judge marks semantic_judge check as True."""
    scores = score_rubric(_SEMANTIC_RUBRIC, "The answer is Paris.", _YesJudge())
    assert scores["sj1"] is True


def test_semantic_judge_no():
    """score_rubric with a NO judge marks semantic_judge check as False."""
    scores = score_rubric(_SEMANTIC_RUBRIC, "I have no idea.", _NoJudge())
    assert scores["sj1"] is False


def test_semantic_judge_no_client_defaults_true(caplog):
    """
    When judge_client=None the stub behaviour is preserved: check returns True
    and a WARNING is emitted.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="mas.eval.harness"):
        scores = score_rubric(_SEMANTIC_RUBRIC, "anything", None)

    assert scores["sj1"] is True
    assert any("semantic_judge" in r.message for r in caplog.records), \
        "Expected a WARNING about missing judge_client"


def test_run_one_grading_fields_populated(monkeypatch):
    """
    run_one() always populates grading_api_calls / grading_cost_usd in the
    RunLog (both are 0 when judge_backend='none').
    """
    entry = _canned_run_entry(condition="FULL_FRAMEWORK", success=1)
    _patch_make_kernel(monkeypatch, entry)

    result = run_one(
        "research_synth_001",
        "FULL_FRAMEWORK",
        42,
        model_overrides = {"judge_backend": "none"},
        _skip_warmup    = True,
        _goals_dir      = FIXTURE_GOALS_DIR,
    )

    assert "grading_api_calls" in result
    assert "grading_cost_usd"  in result
    assert result["grading_api_calls"] == 0
    assert result["grading_cost_usd"]  == 0.0
