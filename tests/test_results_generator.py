"""Tests for mas/eval/results_generator.py."""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, Optional

import pytest

from mas.eval.results_generator import generate


_CONDITIONS = [
    "FULL_FRAMEWORK",
    "BASELINE_LANGGRAPH",
]

_GOAL_TYPES = [
    "research_synthesis",
    "financial_analysis",
    "sequential_reasoning",
    "code_analysis",
    "adversarial_reliability",
]


def _make_run(
    goal_id: str,
    condition: str,
    seed: int = 0,
    goal_type: str = "research_synthesis",
    success: int = 1,
    cascade_failure: int = 0,
    pipeline_depth_k: int = 3,
    tokens_total: int = 2000,
    model_llm: str = "llama3.1:8b",
) -> dict[str, Any]:
    """Build a minimal valid run-log dict matching the RunLog schema in mas/eval/harness.py."""
    return {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            goal_id,
        "goal_type":          goal_type,
        "pipeline_depth_k":   pipeline_depth_k,
        "condition":          condition,
        "routing_skipped":    False,
        "seed":               seed,
        "model_slm":          "phi3:mini",
        "model_llm":          model_llm,
        "started_at":         "2026-01-01T00:00:00Z",
        "completed_at":       "2026-01-01T00:01:00Z",
        "latency_s":          10.0,
        "success":            success,
        "cascade_failure":    cascade_failure,
        "tokens": {
            "slm":   400,
            "llm":   max(0, tokens_total - 400),
            "total": tokens_total,
        },
        "s0": {"catches": 0, "repairs": 0, "would_have_failed": 0},
        "escalations": {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step": [
            {
                "step_id": 1,
                "agent_id": "WebAgent",
                "tier": 1,
                "raw_output": "...",
                "schema_valid": True,
                "would_have_been_invalid": bool(cascade_failure),
                "downstream_consumed": bool(cascade_failure),
                "tokens_used": tokens_total,
                "latency_ms": 1000,
            }
        ],
        "final_output":       "result",
        "ground_truth":       "expected",
        "rubric_scores":      {"c1": bool(success)},
        "ground_truth_match": bool(success),
        "voc_trace":          [0.8],
        "crash_reason":       None,
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
    }


def _write_jsonl(runs: list[dict], tmp_path: Path) -> str:
    p = tmp_path / "runs.jsonl"
    with open(p, "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    return str(p)


def _build_paired_runs(
    n_per_cond: int = 30,
    seed: int = 42,
) -> list[dict]:
    """
    Build a paired synthetic dataset with the 2-condition protocol and all 5 goal types.

    Uses two distinct pipeline_depth_k values (3 and 5).
    Each (goal_id, seed_val) tuple exists in both conditions.
    """
    rng = random.Random(seed)
    runs: list[dict] = []
    gid_counter = 0

    for gtype in _GOAL_TYPES:
        for i in range(n_per_cond // len(_GOAL_TYPES) + 1):
            gid = f"{gtype[:8]}_{gid_counter:04d}"
            gid_counter += 1
            depth = 3 if i % 2 == 0 else 5
            seed_val = i

            for cond in _CONDITIONS:
                # FULL: low failure, others: higher failure
                cf_prob = 0.05 if cond == "FULL_FRAMEWORK" else 0.30
                is_cf = rng.random() < cf_prob
                runs.append(_make_run(
                    goal_id=gid,
                    condition=cond,
                    seed=seed_val,
                    goal_type=gtype,
                    success=int(not is_cf),
                    cascade_failure=int(is_cf),
                    pipeline_depth_k=depth,
                    tokens_total=rng.randint(1500, 6000),
                ))

    return runs


_REQUIRED_HEADERS = [
    "## Experimental Results",
    "### Sample Sizes",
    "### Primary Outcomes",
    "### CFR by Condition",
    "### Capability & Performance",
    "### Cost-Quality",
    "### Frontier Replication",
    "### Robustness Checks",
]


class TestGeneratesMarkdown:
    def test_creates_results_md(self, tmp_path: Path) -> None:
        """generate() creates RESULTS.md from synthetic paired runs."""
        runs = _build_paired_runs(n_per_cond=30)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        assert Path(out).exists(), "RESULTS.md was not created"
        content = Path(out).read_text()
        assert len(content) > 500, "RESULTS.md is too short"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """generate() creates nested output directories if they don't exist."""
        runs = _build_paired_runs(n_per_cond=10)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "nested" / "deep" / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        assert Path(out).exists()

    def test_total_runs_in_header(self, tmp_path: Path) -> None:
        """Header line reports the correct total run count."""
        runs = _build_paired_runs(n_per_cond=20)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        # results_generator filters runs.jsonl to FULL_FRAMEWORK + BASELINE_LANGGRAPH
        n = sum(1 for r in runs if r["condition"] in ("FULL_FRAMEWORK", "BASELINE_LANGGRAPH"))
        assert f"Total runs: {n}" in content, (
            f"Expected 'Total runs: {n}' in RESULTS.md header"
        )


class TestContainsAllSections:
    def test_all_section_headers_present(self, tmp_path: Path) -> None:
        """
        Every required section header appears in RESULTS.md.

        Checks for the exact strings defined in _REQUIRED_HEADERS.
        """
        runs = _build_paired_runs(n_per_cond=20)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        missing = [h for h in _REQUIRED_HEADERS if h not in content]
        assert not missing, (
            f"Missing section headers in RESULTS.md: {missing}"
        )

    def test_alpha_mentioned(self, tmp_path: Path) -> None:
        """RESULTS.md mentions the (uncorrected) alpha threshold."""
        runs = _build_paired_runs(n_per_cond=20)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        assert "Significance threshold" in content and "0.05" in content, (
            "Alpha threshold not mentioned in RESULTS.md"
        )

    def test_robustness_placeholder_present(self, tmp_path: Path) -> None:
        """Robustness section contains the placeholder text."""
        runs = _build_paired_runs(n_per_cond=10)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        assert "Robustness Checks" in content

    def test_cfr_table_has_both_conditions(self, tmp_path: Path) -> None:
        """CFR table lists System 0/3 and Baseline conditions."""
        runs = _build_paired_runs(n_per_cond=20)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        for cond in ("System 0/3", "Baseline"):
            assert cond in content, f"Condition label '{cond}' not found in RESULTS.md"

    def test_sample_sizes_table_present(self, tmp_path: Path) -> None:
        """Sample Sizes table lists goal-type column headers."""
        runs = _build_paired_runs(n_per_cond=20)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        # Short column names from the generator
        for label in ("Research", "Financial", "Sequential", "Code", "Adversarial"):
            assert label in content, f"Goal-type column '{label}' missing from Sample Sizes table"


class TestHandlesMissingFigures:
    def test_generates_without_figures(self, tmp_path: Path) -> None:
        """
        RESULTS.md is created even when figure files do not exist and
        generate_figures=False is passed.

        All [figure not available] notes must appear for each missing figure.
        """
        runs = _build_paired_runs(n_per_cond=20)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        assert Path(out).exists()
        content = Path(out).read_text()

        assert "not available" in content, (
            "Expected [figure not available] notes in RESULTS.md when figures are absent"
        )

    def test_all_section_headers_still_present(self, tmp_path: Path) -> None:
        """Section headers must appear even when figures are missing."""
        runs = _build_paired_runs(n_per_cond=10)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        missing = [h for h in _REQUIRED_HEADERS if h not in content]
        assert not missing, f"Missing headers when figures absent: {missing}"

    def test_no_exception_raised(self, tmp_path: Path) -> None:
        """generate() must not raise even with no figures available."""
        runs = _build_paired_runs(n_per_cond=10)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)  # must not raise


class TestHandlesZeroRuns:
    def _empty_jsonl(self, tmp_path: Path) -> str:
        p = tmp_path / "runs.jsonl"
        p.write_text("")
        return str(p)

    def test_generates_with_empty_runs(self, tmp_path: Path) -> None:
        """generate() creates RESULTS.md even when runs.jsonl is empty."""
        runs_path = self._empty_jsonl(tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        assert Path(out).exists(), "RESULTS.md not created for empty runs.jsonl"

    def test_zero_runs_total_in_header(self, tmp_path: Path) -> None:
        """Header reports 'Total runs: 0' for an empty log."""
        runs_path = self._empty_jsonl(tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        assert "Total runs: 0" in content

    def test_zero_runs_all_section_headers_present(self, tmp_path: Path) -> None:
        """All section headers appear even for empty runs.jsonl."""
        runs_path = self._empty_jsonl(tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        missing = [h for h in _REQUIRED_HEADERS if h not in content]
        assert not missing, f"Missing headers for empty runs: {missing}"

    def test_zero_runs_no_exception(self, tmp_path: Path) -> None:
        """generate() must not raise for an empty runs.jsonl."""
        runs_path = self._empty_jsonl(tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)  # must not raise

    def test_zero_runs_no_data_note(self, tmp_path: Path) -> None:
        """'No data' note appears somewhere in the report."""
        runs_path = self._empty_jsonl(tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        assert "No data" in content or "no data" in content.lower()


class TestFormatting:
    def test_p_value_formatted_as_less_than_for_tiny(self, tmp_path: Path) -> None:
        """
        When a test is highly significant, p-value cell shows '< 0.001'
        not a raw floating-point number with many decimals.
        """
        from mas.eval.results_generator import _fmt_p
        assert _fmt_p(0.0001) == "< 0.001"
        assert _fmt_p(0.00000001) == "< 0.001"

    def test_p_value_two_decimals(self) -> None:
        """p-values >= 0.001 use exactly 2 decimal places."""
        from mas.eval.results_generator import _fmt_p
        assert _fmt_p(0.04) == "0.04"
        assert _fmt_p(0.12) == "0.12"

    def test_prop_three_decimals(self) -> None:
        """Proportions are formatted to 3 decimal places."""
        from mas.eval.results_generator import _fmt_prop
        assert _fmt_prop(0.1234) == "0.123"
        assert _fmt_prop(0.0) == "0.000"
        assert _fmt_prop(1.0) == "1.000"

    def test_ci_format(self) -> None:
        """CI formatted as [lo, hi] with 3 decimal places."""
        from mas.eval.results_generator import _fmt_ci_pair
        assert _fmt_ci_pair(0.1, 0.4) == "[0.100, 0.400]"

    def test_results_md_is_valid_markdown_tables(self, tmp_path: Path) -> None:
        """
        Every table row in RESULTS.md starts with '|' and ends with '|',
        confirming correct markdown table syntax.
        """
        runs = _build_paired_runs(n_per_cond=20)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        content = Path(out).read_text()
        table_lines = [l for l in content.splitlines() if l.startswith("|")]
        assert table_lines, "No markdown table rows found in RESULTS.md"
        for line in table_lines:
            assert line.endswith("|"), (
                f"Table row does not end with '|': {line!r}"
            )


class TestCLI:
    def test_results_subcommand_exists(self, tmp_path: Path) -> None:
        """
        'python -m mas.eval.cli results' is a registered subcommand.
        Verify by calling the parser directly (no subprocess).
        """
        from mas.eval.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "results",
            "--runs-path", str(tmp_path / "runs.jsonl"),
            "--output", str(tmp_path / "RESULTS.md"),
            "--no-figures",
        ])
        assert args.command == "results"
        assert args.no_figures is True

    def test_results_subcommand_generates_file(self, tmp_path: Path) -> None:
        """
        Calling _cmd_results via the CLI dispatcher produces RESULTS.md.
        """
        import argparse
        from mas.eval.cli import _cmd_results

        runs = _build_paired_runs(n_per_cond=10)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        args = argparse.Namespace(
            runs_path=runs_path,
            output=out,
            no_figures=True,
        )
        _cmd_results(args)

        assert Path(out).exists()


class TestCapabilityPerformanceSection:
    """Tests for the simplified ### Capability & Performance section."""

    def test_section_renders(self, tmp_path: Path) -> None:
        runs = _build_paired_runs(n_per_cond=10)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False, capability_bootstrap_n=200)
        text = Path(out).read_text()

        assert "### Capability & Performance" in text
        assert "Accuracy" in text
        assert "JPI" in text
        assert "Execution Completion Rate (CR)" in text
        assert "System 0/3" in text
        assert "Baseline" in text

    def test_empty_runs(self, tmp_path: Path) -> None:
        runs_path = _write_jsonl([], tmp_path)
        out = str(tmp_path / "RESULTS.md")
        generate(runs_path, out, generate_figures=False, capability_bootstrap_n=50)
        text = Path(out).read_text()
        assert "### Capability & Performance" in text
        assert "*No data.*" in text


class TestExecutionCompletionRelabeling:
    """Presentation uses CR labels; data still comes from runs[].success."""

    def test_cost_quality_table_uses_cr_column_header(self, tmp_path: Path) -> None:
        runs = _build_paired_runs(n_per_cond=12)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        text = Path(out).read_text()
        assert "Execution Completion Rate (CR)" in text
        assert "Success Rate" not in text.split("### Cost-Quality", 1)[1].split("### Frontier", 1)[0]

    def test_completion_footnote_under_cost_quality_table(self, tmp_path: Path) -> None:
        runs = _build_paired_runs(n_per_cond=12)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        cq = Path(out).read_text().split("### Cost-Quality", 1)[1].split("### Frontier", 1)[0]
        assert "Note on Execution Completion vs. Task Accuracy" in cq
        assert "System 0/3" in cq
        assert "Accuracy" in cq and "JPI" in cq

    def test_primary_outcomes_use_completion_not_success_label(self, tmp_path: Path) -> None:
        runs = _build_paired_runs(n_per_cond=20)
        runs_path = _write_jsonl(runs, tmp_path)
        out = str(tmp_path / "RESULTS.md")

        generate(runs_path, out, generate_figures=False)

        primary = Path(out).read_text().split("### Primary Outcomes", 1)[1].split("### CFR", 1)[0]
        assert "Completion (McNemar)" in primary
        assert "Success (McNemar)" not in primary

    def test_mean_execution_completion_reads_success_field(self) -> None:
        from mas.eval.results_generator import _mean_execution_completion

        runs = [
            _make_run("g1", "FULL_FRAMEWORK", success=1),
            _make_run("g2", "FULL_FRAMEWORK", success=0),
        ]
        assert _mean_execution_completion(runs) == 0.5
