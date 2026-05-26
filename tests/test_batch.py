"""Tests for the batch orchestrator (mas/eval/batch.py)."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mas.eval.batch import (
    RunSpec,
    append_completed,
    build_run_queue,
    format_progress_line,
    load_completed,
    run_batch,
)


# 5 goal IDs used across all tests (no real goal files needed)
_INSTANCE_IDS = [
    "research_synth_001",
    "research_synth_002",
    "research_synth_003",
    "research_synth_004",
    "research_synth_005",
]

_ALL_CONDITIONS = [
    "FULL_FRAMEWORK",
    "BASELINE_LANGGRAPH",
]


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canned_run_log(
    goal_id:   str   = "research_synth_001",
    condition: str   = "FULL_FRAMEWORK",
    seed:      int   = 0,
    success:   int   = 1,
) -> dict[str, Any]:
    """Return a minimal but schema-valid RunLog dict."""
    now = _now_iso()
    return {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            goal_id,
        "goal_type":          "research_synthesis",
        "pipeline_depth_k":   2,
        "condition":          condition,
        "routing_skipped":    condition == "BASELINE_LANGGRAPH",
        "seed":               seed,
        "model_slm":          "phi3:mini",
        "model_llm":          "llama3.1:8b",
        "started_at":         now,
        "completed_at":       now,
        "latency_s":          1.2,
        "success":            success,
        "cascade_failure":    0,
        "tokens":             {"slm": 0, "llm": 100, "total": 100},
        "s0":                 {"catches": 0, "repairs": 0, "would_have_failed": 0},
        "escalations":        {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step":           [],
        "final_output":       "Mock answer.",
        "ground_truth":       "expected",
        "rubric_scores":      {"c1": True},
        "ground_truth_match": True,
        "voc_trace":          [],
        "crash_reason":       None,
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
    }


def _mock_run_fn(spec: RunSpec, overrides: dict) -> dict[str, Any]:
    """Mock run_one that returns a canned RunLog without hitting Ollama."""
    return _canned_run_log(spec.goal_id, spec.condition, spec.seed)


def _pass_preflight(goals_dir: Path | None) -> SimpleNamespace:
    """Preflight stub that always passes."""
    return SimpleNamespace(passed=True, errors=[], warnings=[])


def _fail_preflight(goals_dir: Path | None) -> SimpleNamespace:
    """Preflight stub that always fails with two hard errors."""
    return SimpleNamespace(
        passed=False,
        errors=["research_synth_001: _placeholder=true", "research_synth_002: _placeholder=true"],
        warnings=[],
    )


def test_batch_dry_run(capsys):
    """
    2 conditions × 5 instances × 1 seed = 10 expected runs.
    Dry-run must list them all and return an empty list (no execution).
    """
    queue = build_run_queue(
        conditions         = _ALL_CONDITIONS,
        instance_ids       = _INSTANCE_IDS,
        seeds_per_instance = 1,
    )
    assert len(queue) == 10, f"Expected 10 runs in queue, got {len(queue)}"

    result = run_batch(
        queue          = queue,
        runs_jsonl     = Path("/dev/null"),     # never written in dry-run
        completed_path = Path("/dev/null"),
        dry_run        = True,
    )

    assert result == [], "dry-run must return empty list"

    captured = capsys.readouterr()
    # Count lines that start with "  [" (the per-run listing lines)
    listing_lines = [
        ln for ln in captured.out.splitlines()
        if ln.strip().startswith("[") and "|" in ln
    ]
    assert len(listing_lines) == 10, (
        f"Expected 10 listing lines in dry-run output, found {len(listing_lines)}.\n"
        f"Output was:\n{captured.out}"
    )


def test_batch_max_runs(tmp_path):
    """
    With max_runs=3 and 10 runs in the queue, exactly 3 runs execute.
    """
    queue = [RunSpec(g, c, 0) for g in _INSTANCE_IDS for c in _ALL_CONDITIONS]
    assert len(queue) == 10

    executed: list[RunSpec] = []

    def tracking_run_fn(spec: RunSpec, overrides: dict) -> dict[str, Any]:
        executed.append(spec)
        return _canned_run_log(spec.goal_id, spec.condition, spec.seed)

    run_batch(
        queue          = queue,
        runs_jsonl     = tmp_path / "runs.jsonl",
        completed_path = tmp_path / "runs_completed.txt",
        max_runs       = 3,
        _run_one_fn    = tracking_run_fn,
        _preflight_fn  = _pass_preflight,
    )

    assert len(executed) == 3, f"Expected 3 runs, got {len(executed)}"

    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3, f"Expected 3 lines in runs.jsonl, got {len(lines)}"


def test_batch_resume(tmp_path):
    """
    With 5 triples pre-recorded in runs_completed.txt, only 5 run.
    """
    queue = [RunSpec(g, c, 0) for g in _INSTANCE_IDS for c in _ALL_CONDITIONS]
    assert len(queue) == 10

    completed_path = tmp_path / "runs_completed.txt"

    # Record first 5 as done
    for spec in queue[:5]:
        append_completed(completed_path, spec)

    done_before = load_completed(completed_path)
    assert len(done_before) == 5

    executed: list[RunSpec] = []

    def tracking_run_fn(spec: RunSpec, overrides: dict) -> dict[str, Any]:
        executed.append(spec)
        return _canned_run_log(spec.goal_id, spec.condition, spec.seed)

    run_batch(
        queue          = queue,
        runs_jsonl     = tmp_path / "runs.jsonl",
        completed_path = completed_path,
        resume         = True,
        _run_one_fn    = tracking_run_fn,
        _preflight_fn  = _pass_preflight,
    )

    assert len(executed) == 5, f"Expected 5 runs after resume, got {len(executed)}"

    pre_done_set = set(tuple(s) for s in queue[:5])
    for spec in executed:
        assert tuple(spec) not in pre_done_set, (
            f"Spec {spec} was in the pre-completed set but was re-run"
        )


def test_batch_preflight_blocks(tmp_path):
    """
    When preflight returns errors, run_batch raises SystemExit(1) and
    no runs are executed.
    """
    queue = [RunSpec(g, c, 0) for g in _INSTANCE_IDS[:2] for c in _ALL_CONDITIONS[:1]]

    executed: list[RunSpec] = []

    def tracking_run_fn(spec: RunSpec, overrides: dict) -> dict[str, Any]:
        executed.append(spec)
        return _canned_run_log(spec.goal_id, spec.condition, spec.seed)

    with pytest.raises(SystemExit) as exc_info:
        run_batch(
            queue          = queue,
            runs_jsonl     = tmp_path / "runs.jsonl",
            completed_path = tmp_path / "runs_completed.txt",
            _run_one_fn    = tracking_run_fn,
            _preflight_fn  = _fail_preflight,
        )

    assert exc_info.value.code == 1, "Expected exit code 1 on preflight failure"
    assert len(executed) == 0, "No runs should execute when preflight fails"

    assert not (tmp_path / "runs.jsonl").exists(), \
        "runs.jsonl must not be created when preflight aborts the batch"


def test_batch_progress_format(tmp_path, capsys):
    """
    Each completed-run line on stdout matches:
        [N/M] goal_id condition success=X cascade=Y latency=Z.Zs
    """
    queue = [RunSpec(g, "FULL_FRAMEWORK", 0) for g in _INSTANCE_IDS]  # 5 runs

    run_batch(
        queue          = queue,
        runs_jsonl     = tmp_path / "runs.jsonl",
        completed_path = tmp_path / "runs_completed.txt",
        _run_one_fn    = _mock_run_fn,
        _preflight_fn  = _pass_preflight,
    )

    captured = capsys.readouterr()
    # Collect progress lines (those starting with "[")
    progress_lines = [
        ln for ln in captured.out.splitlines()
        if ln.startswith("[")
    ]

    assert len(progress_lines) == 5, (
        f"Expected 5 progress lines, got {len(progress_lines)}.\n"
        f"stdout was:\n{captured.out}"
    )

    # Pattern: [N/M] goal_id condition success=INT cascade=INT latency=FLOAT s
    pattern = re.compile(
        r"^\[\d+/\d+\] \S+ \S+ success=[01] cascade=[01] latency=\d+\.\d+s$"
    )
    for line in progress_lines:
        assert pattern.match(line), (
            f"Progress line does not match expected format.\n"
            f"Line:    {line!r}\n"
            f"Pattern: {pattern.pattern}"
        )


def test_batch_summary_format(tmp_path, capsys):
    """
    After a batch covering all 4 conditions, the summary table must
    contain a row for each condition with Runs, Success%, CFR, and
    Mean tokens columns.
    """
    # 1 instance × 4 conditions × 1 seed = 4 runs
    queue = [RunSpec("research_synth_001", c, 0) for c in _ALL_CONDITIONS]

    run_batch(
        queue          = queue,
        runs_jsonl     = tmp_path / "runs.jsonl",
        completed_path = tmp_path / "runs_completed.txt",
        _run_one_fn    = _mock_run_fn,
        _preflight_fn  = _pass_preflight,
    )

    captured = capsys.readouterr()

    for condition in _ALL_CONDITIONS:
        assert condition in captured.out, (
            f"Condition '{condition}' missing from batch summary.\n"
            f"stdout was:\n{captured.out}"
        )

    # Check that the summary header keywords are present
    assert "Success%" in captured.out
    assert "CFR"       in captured.out
    assert "tokens"    in captured.out.lower()


def test_build_run_queue_count():
    """Cross-product is correct: N_conditions × N_instances × N_seeds."""
    queue = build_run_queue(
        conditions         = _ALL_CONDITIONS[:2],
        instance_ids       = _INSTANCE_IDS[:3],
        seeds_per_instance = 2,
    )
    # 2 conditions × 3 instances × 2 seeds = 12
    assert len(queue) == 12
    assert all(isinstance(s, RunSpec) for s in queue)
    assert all(s.seed in (0, 1) for s in queue)


def test_build_run_queue_ordering():
    """Queue ordering: goal_id (outer) × condition × seed (inner)."""
    queue = build_run_queue(
        conditions         = ["FULL_FRAMEWORK", "BASELINE_LANGGRAPH"],
        instance_ids       = ["research_synth_001", "research_synth_002"],
        seeds_per_instance = 1,
    )
    assert len(queue) == 4
    # goal_id stays contiguous: first 2 entries are for research_synth_001
    assert queue[0].goal_id == "research_synth_001"
    assert queue[1].goal_id == "research_synth_001"
    assert queue[2].goal_id == "research_synth_002"
    assert queue[3].goal_id == "research_synth_002"


def test_completed_round_trip(tmp_path):
    """append_completed + load_completed is a correct round-trip."""
    completed_path = tmp_path / "runs_completed.txt"

    specs = [
        RunSpec("research_synth_001", "FULL_FRAMEWORK", 0),
        RunSpec("research_synth_002", "BASELINE_LANGGRAPH", 7),
    ]
    for spec in specs:
        append_completed(completed_path, spec)

    done = load_completed(completed_path)
    assert len(done) == 2
    for spec in specs:
        assert (spec.goal_id, spec.condition, spec.seed) in done


def test_format_progress_line():
    """format_progress_line produces the documented one-liner."""
    run_log = _canned_run_log("research_synth_001", "FULL_FRAMEWORK", 42)
    line    = format_progress_line(3, 20, run_log)
    assert line.startswith("[3/20]")
    assert "FULL_FRAMEWORK" in line
    assert "success=1"      in line
    assert "cascade=0"      in line
    assert "latency="       in line
    assert line.endswith("s")
