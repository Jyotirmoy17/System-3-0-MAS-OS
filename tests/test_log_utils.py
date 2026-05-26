"""Tests for mas/eval/log_utils.py."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from mas.eval.log_utils import (
    ValidationReport,
    deduplicate_runs,
    filter_runs,
    format_summary,
    runs_summary,
    validate_runs_jsonl,
)


def _run_log(
    goal_id:   str  = "research_synth_001",
    condition: str  = "FULL_FRAMEWORK",
    seed:      int  = 0,
    success:   int  = 1,
    goal_type: str  = "research_synthesis",
    pipeline_depth_k: int = 3,
    tokens:    int  = 80,
    latency:   float = 1.0,
    cascade:   int  = 0,
) -> dict[str, Any]:
    return {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            goal_id,
        "goal_type":          goal_type,
        "pipeline_depth_k":   pipeline_depth_k,
        "condition":          condition,
        "routing_skipped":    condition == "BASELINE_LANGGRAPH",
        "seed":               seed,
        "model_slm":          "phi3:mini",
        "model_llm":          "llama3.1:8b",
        "started_at":         "2026-05-01T10:00:00Z",
        "completed_at":       "2026-05-01T10:00:01Z",
        "latency_s":          latency,
        "success":            success,
        "cascade_failure":    cascade,
        "tokens":             {"slm": 0, "llm": tokens, "total": tokens},
        "s0":                 {"catches": 0, "repairs": 0, "would_have_failed": 0},
        "escalations":        {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step":           [],
        "final_output":       "Answer.",
        "ground_truth":       "expected",
        "rubric_scores":      {"c1": True},
        "ground_truth_match": True,
        "voc_trace":          [],
        "crash_reason":       None,
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
    }


def _write_jsonl(path: Path, entries: list) -> None:
    """Write entries (dicts or raw strings) as JSONL."""
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            if isinstance(e, str):
                fh.write(e + "\n")
            else:
                fh.write(json.dumps(e) + "\n")


def test_validate_clean_file(tmp_path):
    """Three valid run logs → no invalid lines, no duplicates, ok=True."""
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [
        _run_log("research_synth_001", "FULL_FRAMEWORK", 0),
        _run_log("research_synth_002", "BASELINE_LANGGRAPH", 0),
        _run_log("research_synth_003", "FULL_FRAMEWORK", 1),
    ])

    report = validate_runs_jsonl(p)

    assert report.total_lines        == 3
    assert report.valid_lines        == 3
    assert report.invalid_lines      == []
    assert report.duplicate_groups   == []
    assert report.ok                 is True
    assert report.duplicate_count    == 0


def test_validate_catches_malformed_json(tmp_path):
    """A broken JSON line is flagged with the correct 1-based line number."""
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [
        _run_log("research_synth_001", "FULL_FRAMEWORK", 0),
        "{not: valid json",                                      # line 2 — bad
        _run_log("research_synth_003", "BASELINE_LANGGRAPH", 0),
    ])

    report = validate_runs_jsonl(p)

    assert report.total_lines   == 3
    assert report.valid_lines   == 2
    assert len(report.invalid_lines) == 1

    inv = report.invalid_lines[0]
    assert inv.line_no == 2
    assert inv.reason  == "malformed_json"
    assert report.ok   is False


def test_validate_catches_schema_violation(tmp_path):
    """A line missing a required field (run_id) is flagged as invalid."""
    bad = _run_log("research_synth_001", "FULL_FRAMEWORK", 0)
    del bad["run_id"]               # violates minLength:1 / required

    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [
        _run_log("research_synth_002", "BASELINE_LANGGRAPH", 0),
        bad,
    ])

    report = validate_runs_jsonl(p)

    assert report.total_lines        == 2
    assert report.valid_lines        == 1
    assert len(report.invalid_lines) == 1
    assert report.invalid_lines[0].line_no == 2
    assert report.invalid_lines[0].reason in ("missing_fields", "schema_violation")
    assert report.ok is False


def test_validate_detects_duplicates(tmp_path):
    """Three entries for the same triple → one duplicate group with count=3."""
    triple_entry = _run_log("research_synth_001", "FULL_FRAMEWORK", 42)
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [triple_entry, triple_entry, triple_entry])

    report = validate_runs_jsonl(p)

    assert report.valid_lines          == 3
    assert len(report.duplicate_groups) == 1
    assert report.duplicate_groups[0].count == 3
    assert report.duplicate_count          == 2   # 3 entries − 1 unique = 2 extra
    assert report.ok is False


def test_dedupe_latest(tmp_path):
    """Three entries for same triple; 'latest' strategy keeps the last one."""
    p = tmp_path / "runs.jsonl"
    early  = _run_log("research_synth_001", "FULL_FRAMEWORK", 0, success=0)
    middle = _run_log("research_synth_001", "FULL_FRAMEWORK", 0, success=0)
    last   = _run_log("research_synth_001", "FULL_FRAMEWORK", 0, success=1)

    # Assign unique run_ids so we can identify which was kept
    early["run_id"]  = "aaa"
    middle["run_id"] = "bbb"
    last["run_id"]   = "ccc"

    _write_jsonl(p, [early, middle, last])

    removed = deduplicate_runs(p, strategy="latest")

    assert removed == 2

    lines = p.read_text().splitlines()
    assert len(lines) == 1
    kept = json.loads(lines[0])
    assert kept["run_id"] == "ccc", "latest strategy must keep the last-written entry"


def test_dedupe_successful(tmp_path):
    """With success=0 then success=1 for same triple, 'successful' keeps the winner."""
    p = tmp_path / "runs.jsonl"
    fail_entry = _run_log("research_synth_001", "FULL_FRAMEWORK", 0, success=0)
    win_entry  = _run_log("research_synth_001", "FULL_FRAMEWORK", 0, success=1)
    fail_entry["run_id"] = "fail"
    win_entry["run_id"]  = "win"

    _write_jsonl(p, [fail_entry, win_entry])

    removed = deduplicate_runs(p, strategy="successful")

    assert removed == 1

    lines = p.read_text().splitlines()
    assert len(lines) == 1
    kept = json.loads(lines[0])
    assert kept["run_id"] == "win", "successful strategy must keep success=1"


def test_dedupe_earliest(tmp_path):
    """'earliest' strategy keeps the first occurrence."""
    p = tmp_path / "runs.jsonl"
    first  = _run_log("research_synth_001", "FULL_FRAMEWORK", 0)
    second = _run_log("research_synth_001", "FULL_FRAMEWORK", 0)
    first["run_id"]  = "first"
    second["run_id"] = "second"

    _write_jsonl(p, [first, second])

    removed = deduplicate_runs(p, strategy="earliest")

    assert removed == 1
    kept = json.loads(p.read_text().strip())
    assert kept["run_id"] == "first"


def test_dedupe_no_duplicates(tmp_path):
    """When there are no duplicates, removed count is 0 and file is unchanged."""
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [
        _run_log("research_synth_001", "FULL_FRAMEWORK", 0),
        _run_log("research_synth_002", "FULL_FRAMEWORK", 0),
    ])
    before = p.read_text()

    removed = deduplicate_runs(p, strategy="latest")

    assert removed   == 0
    assert p.read_text() == before


def test_filter_by_condition(tmp_path):
    """filter_runs returns only entries matching the given condition."""
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [
        _run_log("research_synth_001", "FULL_FRAMEWORK", 0),
        _run_log("research_synth_001", "BASELINE_LANGGRAPH", 0),
        _run_log("research_synth_002", "FULL_FRAMEWORK", 0),
        _run_log("research_synth_002", "BASELINE_LANGGRAPH", 0),
    ])

    results = filter_runs(p, condition="FULL_FRAMEWORK")

    assert len(results) == 2
    assert all(r["condition"] == "FULL_FRAMEWORK" for r in results)


def test_filter_multi_field(tmp_path):
    """filter_runs with two kwargs applies AND logic."""
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [
        _run_log("research_synth_001", "FULL_FRAMEWORK", 0, success=1),
        _run_log("research_synth_001", "FULL_FRAMEWORK", 0, success=0),
        _run_log("research_synth_001", "BASELINE_LANGGRAPH", 0, success=1),
    ])

    results = filter_runs(p, condition="FULL_FRAMEWORK", success=1)

    assert len(results) == 1
    assert results[0]["success"]   == 1
    assert results[0]["condition"] == "FULL_FRAMEWORK"


def test_summary_counts(tmp_path):
    """runs_summary numbers match what was written to the file."""
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [
        # FULL_FRAMEWORK: 2 runs, 1 success, 1 cascade
        _run_log("research_synth_001", "FULL_FRAMEWORK", 0, success=1, cascade=0),
        _run_log("research_synth_002", "FULL_FRAMEWORK", 0, success=0, cascade=1),
        # BASELINE_LANGGRAPH: 1 run, 0 success, 0 cascade
        _run_log("research_synth_001", "BASELINE_LANGGRAPH", 0, success=0, cascade=0),
    ])

    s = runs_summary(p)

    assert s["total_runs"]   == 3
    assert s["schema_valid"] == 3

    ff = s["by_condition"]["FULL_FRAMEWORK"]
    assert ff["runs"]             == 2
    assert ff["successes"]        == 1
    assert ff["success_rate"]     == 0.5
    assert ff["cascade_failures"] == 1
    assert ff["cfr"]              == 0.5

    bl = s["by_condition"]["BASELINE_LANGGRAPH"]
    assert bl["runs"]         == 1
    assert bl["successes"]    == 0
    assert bl["success_rate"] == 0.0


def test_summary_by_goal_type_and_pipeline_depth(tmp_path):
    """runs_summary breaks down correctly by goal_type and pipeline_depth_k."""
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [
        _run_log("research_synth_001", "FULL_FRAMEWORK", 0,
                 goal_type="research_synthesis", pipeline_depth_k=3, success=1),
        _run_log("adversarial_001",    "FULL_FRAMEWORK", 0,
                 goal_type="adversarial_reliability", pipeline_depth_k=5, success=0),
    ])

    s = runs_summary(p)

    assert s["by_goal_type"]["research_synthesis"]["runs"] == 1
    assert s["by_goal_type"]["adversarial_reliability"]["runs"] == 1

    assert s["by_pipeline_depth"]["3"]["success_rate"] == 1.0
    assert s["by_pipeline_depth"]["5"]["success_rate"] == 0.0


def test_format_summary_output(tmp_path):
    """format_summary produces human-readable output with all key headers."""
    p = tmp_path / "runs.jsonl"
    p2 = tmp_path / "runs2.jsonl"
    _write_jsonl(p2, [
        _run_log("research_synth_001", c, 0)
        for c in ("FULL_FRAMEWORK", "BASELINE_LANGGRAPH")
    ])

    s   = runs_summary(p2)
    txt = format_summary(s)

    assert "FULL_FRAMEWORK"     in txt
    assert "BASELINE_LANGGRAPH" in txt
    assert "Success%"           in txt
    assert "CFR"                in txt


def test_dedupe_invalid_strategy(tmp_path):
    """Passing an unknown strategy raises ValueError immediately."""
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [_run_log()])

    with pytest.raises(ValueError, match="Unknown deduplication strategy"):
        deduplicate_runs(p, strategy="bogus")


def test_validate_empty_file(tmp_path):
    """An empty file is valid (0 lines, 0 valid, ok=True)."""
    p = tmp_path / "runs.jsonl"
    p.write_text("")

    report = validate_runs_jsonl(p)

    assert report.total_lines == 0
    assert report.valid_lines == 0
    assert report.ok          is True


def test_filter_missing_file(tmp_path):
    """filter_runs returns [] when the file does not exist."""
    result = filter_runs(tmp_path / "nonexistent.jsonl", condition="FULL_FRAMEWORK")
    assert result == []


def test_validation_report_to_text(tmp_path):
    """ValidationReport.to_text() includes error details and status."""
    p = tmp_path / "runs.jsonl"
    _write_jsonl(p, [
        "{broken",
        _run_log("research_synth_001", "FULL_FRAMEWORK", 0),
    ])

    report = validate_runs_jsonl(p)
    text   = report.to_text()

    assert "Line 1"          in text
    assert "malformed_json"  in text
    assert "[FAIL]"          in text
