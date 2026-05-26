"""Tests for mas/eval/failure_analysis.py."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest

from mas.eval.failure_analysis import (
    CATEGORIES,
    categorize_failures,
    failure_section_markdown,
    _assign_category,
)


def _make_run(
    *,
    condition:          str   = "FULL_FRAMEWORK",
    goal_type:          str   = "research_synthesis",
    success:            int   = 1,
    crash_reason:       str | None = None,
    ground_truth_match: bool | None = True,
    would_have_been_invalid: bool   = False,
    downstream_consumed:     bool   = False,
    s0_catches:         int   = 0,
) -> dict:
    """Build a minimal valid run-log dict for a specific failure scenario."""
    return {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            f"goal_{uuid.uuid4().hex[:6]}",
        "goal_type":          goal_type,
        "pipeline_depth_k":   3,
        "condition":          condition,
        "routing_skipped":    False,
        "seed":               42,
        "model_slm":          "phi3:mini",
        "model_llm":          "llama3.1:8b",
        "started_at":         "2026-01-01T00:00:00Z",
        "completed_at":       "2026-01-01T00:01:00Z",
        "latency_s":          5.0,
        "success":            success,
        "cascade_failure":    0,
        "tokens":             {"slm": 100, "llm": 900, "total": 1000},
        "s0":                 {"catches": s0_catches, "repairs": 0, "would_have_failed": 0},
        "escalations":        {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step":           [
            {
                "step_id":                       1,
                "agent_id":                      "WebAgent",
                "tier":                          1,
                "raw_output":                    "...",
                "schema_valid":                  True,
                "would_have_been_invalid":       would_have_been_invalid,
                "downstream_consumed":           downstream_consumed,
                "tokens_used":                   1000,
                "latency_ms":                    2000,
                "adversarial_corruption_applied": None,
            }
        ],
        "final_output":       "result",
        "ground_truth":       "result",
        "rubric_scores":      {"check_1": ground_truth_match is not False},
        "ground_truth_match": ground_truth_match,
        "voc_trace":          [0.8],
        "crash_reason":       crash_reason,
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
    }


def _write_runs(tmp_path: Path, runs: list[dict]) -> str:
    """Write *runs* to a temp JSONL file and return its path as str."""
    p = tmp_path / "runs.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for run in runs:
            fh.write(json.dumps(run) + "\n")
    return str(p)


class TestCategorizeEachType:
    """One run per failure category; assert each is classified correctly."""

    def test_agent_crash(self, tmp_path: Path) -> None:
        run = _make_run(success=0, crash_reason="RuntimeError: agent panicked")
        path = _write_runs(tmp_path, [run])
        cats = categorize_failures(path)
        assert "agent_crash" in cats
        assert cats["agent_crash"]["count"] == 1

    def test_budget_exhausted(self, tmp_path: Path) -> None:
        run = _make_run(success=0, crash_reason="budget exceeded: max token limit reached")
        path = _write_runs(tmp_path, [run])
        cats = categorize_failures(path)
        assert "budget_exhausted" in cats
        assert cats["budget_exhausted"]["count"] == 1

    def test_infinite_loop(self, tmp_path: Path) -> None:
        run = _make_run(success=0, crash_reason="max_steps reached: infinite loop detected")
        path = _write_runs(tmp_path, [run])
        cats = categorize_failures(path)
        assert "infinite_loop" in cats
        assert cats["infinite_loop"]["count"] == 1

    def test_schema_violation_uncaught_baseline(self, tmp_path: Path) -> None:
        """BASELINE_LANGGRAPH + cascade → schema_violation_uncaught."""
        run = _make_run(
            condition           = "BASELINE_LANGGRAPH",
            success             = 0,
            ground_truth_match  = False,
            would_have_been_invalid = True,
            downstream_consumed = True,
        )
        path = _write_runs(tmp_path, [run])
        cats = categorize_failures(path)
        assert "schema_violation_uncaught" in cats

    def test_cascade_failure_full_framework(self, tmp_path: Path) -> None:
        """FULL_FRAMEWORK + cascade → cascade_failure (not schema_violation_uncaught)."""
        run = _make_run(
            condition           = "FULL_FRAMEWORK",
            success             = 0,
            ground_truth_match  = False,
            would_have_been_invalid = True,
            downstream_consumed = True,
        )
        path = _write_runs(tmp_path, [run])
        cats = categorize_failures(path)
        assert "cascade_failure" in cats
        assert "schema_violation_uncaught" not in cats

    def test_rubric_failed_no_cascade(self, tmp_path: Path) -> None:
        """Failed rubric, no crash, no cascade → rubric_failed_no_cascade."""
        run = _make_run(
            success             = 0,
            ground_truth_match  = False,
            would_have_been_invalid = False,
            downstream_consumed = False,
        )
        path = _write_runs(tmp_path, [run])
        cats = categorize_failures(path)
        assert "rubric_failed_no_cascade" in cats
        assert cats["rubric_failed_no_cascade"]["count"] == 1

    def test_schema_violation_caught(self, tmp_path: Path) -> None:
        """S0 caught violation but run succeeded → schema_violation_caught."""
        run = _make_run(
            condition           = "FULL_FRAMEWORK",
            success             = 1,
            ground_truth_match  = True,
            s0_catches          = 2,
        )
        path = _write_runs(tmp_path, [run])
        cats = categorize_failures(path)
        assert "schema_violation_caught" in cats
        assert cats["schema_violation_caught"]["count"] == 1

    def test_clean_success_excluded(self, tmp_path: Path) -> None:
        """Fully successful run with no violations is excluded from all categories."""
        run = _make_run(success=1, ground_truth_match=True, s0_catches=0)
        path = _write_runs(tmp_path, [run])
        cats = categorize_failures(path)
        total = sum(d["count"] for d in cats.values())
        assert total == 0


class TestMutuallyExclusive:
    def test_each_run_in_exactly_one_category(self, tmp_path: Path) -> None:
        """For each synthetic run, _assign_category returns exactly one label."""
        scenarios = [
            _make_run(success=0, crash_reason="RuntimeError: panic"),
            _make_run(success=0, crash_reason="budget exceeded"),
            _make_run(success=0, crash_reason="max_steps hit"),
            _make_run(
                condition="BASELINE_LANGGRAPH", success=0,
                ground_truth_match=False,
                would_have_been_invalid=True, downstream_consumed=True,
            ),
            _make_run(
                condition="FULL_FRAMEWORK", success=0,
                ground_truth_match=False,
                would_have_been_invalid=True, downstream_consumed=True,
            ),
            _make_run(success=0, ground_truth_match=False),
            _make_run(success=1, s0_catches=3),
        ]
        for run in scenarios:
            cat = _assign_category(run)
            # Must be one of the registered categories (or None for clean success)
            assert cat is None or cat in CATEGORIES, \
                f"Unknown category {cat!r} for run {run['run_id']!r}"

    def test_crash_takes_priority_over_cascade(self, tmp_path: Path) -> None:
        """When crash_reason is set AND cascade criteria hold, crash wins."""
        run = _make_run(
            success=0,
            crash_reason="RuntimeError: everything broke",
            ground_truth_match=False,
            would_have_been_invalid=True,
            downstream_consumed=True,
        )
        cat = _assign_category(run)
        assert cat == "agent_crash"


class TestExamplesCappedAtFive:
    def test_ten_crashes_give_five_examples(self, tmp_path: Path) -> None:
        """10 runs of the same category → examples list has exactly 5 entries."""
        runs = [
            _make_run(success=0, crash_reason="RuntimeError: boom")
            for _ in range(10)
        ]
        path = _write_runs(tmp_path, runs)
        cats = categorize_failures(path)
        assert "agent_crash" in cats
        assert cats["agent_crash"]["count"] == 10
        assert len(cats["agent_crash"]["examples"]) == 5

    def test_fewer_than_five_shows_all(self, tmp_path: Path) -> None:
        """3 runs of a category → all 3 run_ids appear in examples."""
        runs = [
            _make_run(success=0, crash_reason="budget exceeded")
            for _ in range(3)
        ]
        path = _write_runs(tmp_path, runs)
        cats = categorize_failures(path)
        assert len(cats["budget_exhausted"]["examples"]) == 3

    def test_example_run_ids_unique(self, tmp_path: Path) -> None:
        """All example run IDs must be unique (no duplicate from factory bug)."""
        runs = [
            _make_run(success=0, crash_reason="max_steps exceeded")
            for _ in range(7)
        ]
        path = _write_runs(tmp_path, runs)
        cats = categorize_failures(path)
        examples = cats["infinite_loop"]["examples"]
        assert len(examples) == len(set(examples)), "Duplicate run_ids in examples"


class TestBreakdownCounts:
    def test_by_condition_counts_correctly(self, tmp_path: Path) -> None:
        """by_condition reflects the actual condition distribution."""
        runs = (
            [_make_run(condition="FULL_FRAMEWORK",  success=0,
                       crash_reason="RuntimeError: a") for _ in range(3)]
            + [_make_run(condition="BASELINE_LANGGRAPH", success=0,
                         crash_reason="RuntimeError: b") for _ in range(2)]
        )
        path = _write_runs(tmp_path, runs)
        cats = categorize_failures(path)
        by_cond = cats["agent_crash"]["by_condition"]
        assert by_cond.get("FULL_FRAMEWORK") == 3
        assert by_cond.get("BASELINE_LANGGRAPH") == 2

    def test_by_goal_type_counts_correctly(self, tmp_path: Path) -> None:
        """by_goal_type reflects the actual goal type distribution."""
        runs = (
            [_make_run(goal_type="code_analysis",       success=0,
                       ground_truth_match=False) for _ in range(4)]
            + [_make_run(goal_type="financial_analysis", success=0,
                         ground_truth_match=False) for _ in range(1)]
        )
        path = _write_runs(tmp_path, runs)
        cats = categorize_failures(path)
        by_gt = cats["rubric_failed_no_cascade"]["by_goal_type"]
        assert by_gt.get("code_analysis")      == 4
        assert by_gt.get("financial_analysis") == 1


class TestSectionMarkdownFormat:
    def _write_mixed(self, tmp_path: Path) -> str:
        """Write a representative set of failure types."""
        runs = [
            _make_run(success=1, ground_truth_match=True, s0_catches=0),   # clean success
            _make_run(success=0, crash_reason="RuntimeError: panic"),        # agent_crash
            _make_run(success=0, crash_reason="budget exceeded"),            # budget_exhausted
            _make_run(
                condition="BASELINE_LANGGRAPH", success=0, ground_truth_match=False,
                would_have_been_invalid=True, downstream_consumed=True,
            ),  # schema_violation_uncaught
            _make_run(success=0, ground_truth_match=False),                  # rubric_failed_no_cascade
            _make_run(success=1, s0_catches=1),                             # schema_violation_caught
        ]
        return _write_runs(tmp_path, runs)

    def test_section_starts_with_heading(self, tmp_path: Path) -> None:
        """Markdown section must start with '### Failure Mode Analysis'."""
        path = self._write_mixed(tmp_path)
        md   = failure_section_markdown(path)
        assert md.strip().startswith("### Failure Mode Analysis")

    def test_section_contains_summary_table(self, tmp_path: Path) -> None:
        """Markdown section must contain a summary table header."""
        path = self._write_mixed(tmp_path)
        md   = failure_section_markdown(path)
        assert "#### Summary" in md
        assert "| Category |" in md

    def test_section_contains_breakdown(self, tmp_path: Path) -> None:
        """Markdown section must contain per-category breakdown."""
        path = self._write_mixed(tmp_path)
        md   = failure_section_markdown(path)
        assert "#### Breakdown by Category" in md

    def test_no_failures_gives_placeholder(self, tmp_path: Path) -> None:
        """When all runs succeed cleanly, no-failure placeholder is emitted."""
        runs = [_make_run(success=1, ground_truth_match=True, s0_catches=0)
                for _ in range(5)]
        path = _write_runs(tmp_path, runs)
        md   = failure_section_markdown(path)
        assert "No failures" in md

    def test_all_category_labels_present(self, tmp_path: Path) -> None:
        """Present categories appear in the markdown by their human label."""
        path = self._write_mixed(tmp_path)
        md   = failure_section_markdown(path)
        from mas.eval.failure_analysis import CATEGORY_LABELS
        found = [label for label in CATEGORY_LABELS.values() if label in md]
        assert len(found) >= 3, f"Expected >= 3 labels in md; found: {found}"

    def test_example_run_ids_in_markdown(self, tmp_path: Path) -> None:
        """run_id strings must appear in the markdown output."""
        path = self._write_mixed(tmp_path)
        cats = categorize_failures(path)
        md   = failure_section_markdown(path)
        # Check that at least one example ID from agent_crash appears in the md
        if "agent_crash" in cats and cats["agent_crash"]["examples"]:
            run_id = cats["agent_crash"]["examples"][0]
            assert run_id in md


class TestMissingFile:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            categorize_failures(str(tmp_path / "nonexistent.jsonl"))

    def test_markdown_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            failure_section_markdown(str(tmp_path / "nonexistent.jsonl"))
