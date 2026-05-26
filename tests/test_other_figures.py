"""Tests for the three supplementary figure modules:."""
from __future__ import annotations

import json
import random
import unittest.mock as mock
import uuid
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from mas.eval.figures.cost_quality_pareto import generate as pareto_generate
from mas.eval.figures.per_step_survival import generate as survival_generate, fit_epsilon
from mas.eval.figures.frontier_vs_local import generate as frontier_generate


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


def _base_run(
    condition: str,
    goal_type: str,
    goal_id: Optional[str] = None,
    success: int = 1,
    tokens_total: int = 1000,
    latency_s: float = 10.0,
    model_llm: str = "llama3.1:8b",
    per_step: Optional[list] = None,
) -> dict[str, Any]:
    """Minimal valid run-log dict matching EXPERIMENTS.md §4 schema."""
    run_id = str(uuid.uuid4())
    gid = goal_id or f"g_{run_id[:8]}"
    return {
        "run_id":             run_id,
        "goal_id":            gid,
        "goal_type":          goal_type,
        "pipeline_depth_k":   3,
        "condition":          condition,
        "routing_skipped":    False,
        "seed":               42,
        "model_slm":          "phi3:mini",
        "model_llm":          model_llm,
        "started_at":         "2026-01-01T00:00:00Z",
        "completed_at":       "2026-01-01T00:01:00Z",
        "latency_s":          latency_s,
        "success":            success,
        "cascade_failure":    0,
        "tokens": {
            "slm": 200,
            "llm": max(0, tokens_total - 200),
            "total": tokens_total,
        },
        "s0": {"catches": 0, "repairs": 0, "would_have_failed": 0},
        "escalations": {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step": per_step or [
            {
                "step_id": 1,
                "agent_id": "WebAgent",
                "tier": 1,
                "raw_output": "...",
                "schema_valid": True,
                "would_have_been_invalid": False,
                "downstream_consumed": False,
                "tokens_used": tokens_total,
                "latency_ms": int(latency_s * 1000),
            }
        ],
        "final_output":       "result",
        "ground_truth":       "expected",
        "rubric_scores":      {"check_1": bool(success)},
        "ground_truth_match": bool(success),
        "voc_trace":          [0.8],
        "crash_reason":       None,
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
    }


def _write_jsonl(runs: list[dict], tmp_path: Path) -> str:
    """Write runs to tmp_path/runs.jsonl and return the path string."""
    path = tmp_path / "runs.jsonl"
    with open(path, "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _capture_figure(generate_fn, *args, **kwargs):
    """
    Call generate_fn and capture the Figure before plt.close() is called.

    Returns:
        (fig, return_value) — the captured Figure and generate_fn's return value.
    """
    captured: list[plt.Figure] = []
    original_close = plt.close

    def _side_effect(fig=None):
        if fig is not None:
            captured.append(fig)
        original_close(fig)

    with mock.patch("matplotlib.pyplot.close", side_effect=_side_effect):
        retval = generate_fn(*args, **kwargs)

    fig = captured[-1] if captured else None
    return fig, retval


def _figure_texts(fig: plt.Figure) -> str:
    """Collect all text strings from the figure as a single lowercased string."""
    texts: list[str] = []
    for ax in fig.get_axes():
        for child in ax.get_children():
            if hasattr(child, "get_text"):
                texts.append(child.get_text())
    for txt in fig.texts:
        texts.append(txt.get_text())
    return " ".join(texts).lower()


def _build_pareto_runs(n_per_cell: int = 10) -> list[dict]:
    """
    Build synthetic runs covering all 20 (condition × goal_type) cells.

    Uses a fixed random seed for reproducibility.
    """
    rng = random.Random(99)
    runs: list[dict] = []
    for cond in _CONDITIONS:
        for gtype in _GOAL_TYPES:
            for _ in range(n_per_cell):
                runs.append(_base_run(
                    condition=cond,
                    goal_type=gtype,
                    success=rng.randint(0, 1),
                    tokens_total=rng.randint(500, 5000),
                ))
    return runs


class TestPareto:
    def test_pareto_generates(self, tmp_path: Path) -> None:
        """generate() saves both a PDF and a PNG for complete 20-cell data."""
        runs = _build_pareto_runs()
        runs_path = _write_jsonl(runs, tmp_path)
        out_pdf = str(tmp_path / "pareto.pdf")

        pareto_generate(runs_path, out_pdf)

        assert Path(out_pdf).exists(), "PDF not created"
        assert Path(out_pdf).stat().st_size > 0, "PDF is empty"
        png = Path(out_pdf).with_suffix(".png")
        assert png.exists(), "PNG not created"
        assert png.stat().st_size > 0, "PNG is empty"

    def test_pareto_has_20_points(self, tmp_path: Path) -> None:
        """
        The scatter plot has exactly 20 PathCollection objects — one per
        (condition × goal_type) cell.

        Each ax.scatter() call creates one PathCollection, which is added to
        ax.collections.  The Pareto frontier (ax.step) creates a Line2D and
        does NOT appear in ax.collections.
        """
        runs = _build_pareto_runs(n_per_cell=10)
        runs_path = _write_jsonl(runs, tmp_path)
        out_pdf = str(tmp_path / "pareto_20.pdf")

        fig, _ = _capture_figure(pareto_generate, runs_path, out_pdf)
        assert fig is not None, "Figure was not captured"

        ax = fig.get_axes()[0]
        n_scatter = len(ax.collections)
        assert n_scatter == 20, (
            f"Expected exactly 20 scatter markers (4 conditions × 5 goal types), "
            f"got {n_scatter}"
        )


def _make_multistep_step(
    step_id: int,
    fails: bool,
    condition: str,
) -> dict:
    """Build one per_step entry for a multi-step run."""
    schema_valid = True if condition == "FULL_FRAMEWORK" else not fails
    return {
        "step_id":                step_id,
        "agent_id":               "WebAgent",
        "tier":                   1,
        "raw_output":             "...",
        "schema_valid":           schema_valid,
        "would_have_been_invalid": fails,
        "downstream_consumed":    True,
        "tokens_used":            100,
        "latency_ms":             200,
    }


def _build_survival_runs(
    n: int = 300,
    epsilon: float = 0.2,
    n_steps: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Build BASELINE_LANGGRAPH + FULL_FRAMEWORK runs with multi-step per_step.

    BASELINE: each step independently fails with probability epsilon.
    FULL: all steps always succeed (System-0 guarantees schema_valid=True).
    """
    rng = random.Random(seed)
    runs: list[dict] = []

    for i in range(n):
        gid = f"goal_{i:04d}"

        # BASELINE — each step fails independently
        bl_steps = [
            _make_multistep_step(k + 1, rng.random() < epsilon, "BASELINE_LANGGRAPH")
            for k in range(n_steps)
        ]
        first_fail = next(
            (k for k, s in enumerate(bl_steps) if s["would_have_been_invalid"]),
            None,
        )
        runs.append(_base_run(
            condition="BASELINE_LANGGRAPH",
            goal_type="research_synthesis",
            goal_id=gid,
            success=int(first_fail is None),
            per_step=bl_steps,
        ))

        # FULL — all steps always succeed
        full_steps = [
            _make_multistep_step(k + 1, False, "FULL_FRAMEWORK")
            for k in range(n_steps)
        ]
        runs.append(_base_run(
            condition="FULL_FRAMEWORK",
            goal_type="research_synthesis",
            goal_id=gid,
            success=1,
            per_step=full_steps,
        ))

    return runs


class TestSurvival:
    def test_survival_two_lines(self, tmp_path: Path) -> None:
        """
        The figure contains at least two distinct Line2D objects
        (one for FULL_FRAMEWORK, one for BASELINE_LANGGRAPH).

        fill_between CI bands are PolyCollections (ax.collections), not
        Line2D, so ax.lines reflects only the data lines and the
        theoretical-curve overlay.
        """
        runs = _build_survival_runs(n=50)
        runs_path = _write_jsonl(runs, tmp_path)
        out_pdf = str(tmp_path / "surv_lines.pdf")

        fig, _ = _capture_figure(survival_generate, runs_path, out_pdf)
        assert fig is not None

        ax = fig.get_axes()[0]
        n_lines = len(ax.lines)
        assert n_lines >= 2, (
            f"Expected ≥2 Line2D objects (FULL + BASELINE + optional theory), "
            f"got {n_lines}"
        )

    def test_survival_decay_shape(self, tmp_path: Path) -> None:
        """
        Fitted epsilon from BASELINE data with known true ε=0.2 is
        within 0.05 of the true value.

        Uses 300 independent runs (seed=42) for sufficient statistical power;
        0.8^k theoretical decay makes the log-linear fit well-constrained.
        """
        runs = _build_survival_runs(n=300, epsilon=0.2, n_steps=5, seed=42)
        runs_path = _write_jsonl(runs, tmp_path)
        out_pdf = str(tmp_path / "surv_decay.pdf")

        _, result = _capture_figure(survival_generate, runs_path, out_pdf)

        assert result is not None, "generate() should return a dict"
        eps = result.get("fitted_epsilon")
        assert eps is not None, (
            "fitted_epsilon should not be None when BASELINE has multi-step data"
        )
        assert abs(eps - 0.2) < 0.05, (
            f"Fitted ε = {eps:.4f}, true ε = 0.20, "
            f"|diff| = {abs(eps - 0.2):.4f} ≥ 0.05"
        )

    def test_fit_epsilon_unit(self) -> None:
        """
        fit_epsilon() recovers (1-ε) from exact (1-ε)^k values.
        """
        import math
        eps_true = 0.15
        steps = list(range(1, 8))
        survival = [(1 - eps_true) ** k for k in steps]

        eps_hat = fit_epsilon(steps, survival)
        assert eps_hat is not None
        assert abs(eps_hat - eps_true) < 1e-6, (
            f"Expected {eps_true}, got {eps_hat}"
        )

    def test_fit_epsilon_returns_none_for_flat_survival(self) -> None:
        """
        fit_epsilon() returns None when all survival values are 1.0
        (zero variance — cannot distinguish from perfect reliability).
        """
        steps = [1, 2, 3, 4, 5]
        survival = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert fit_epsilon(steps, survival) is None

    def test_survival_pdf_and_png_created(self, tmp_path: Path) -> None:
        """generate() saves both PDF and PNG."""
        runs = _build_survival_runs(n=30)
        runs_path = _write_jsonl(runs, tmp_path)
        out_pdf = str(tmp_path / "surv.pdf")

        survival_generate(runs_path, out_pdf)

        assert Path(out_pdf).exists()
        assert Path(out_pdf).with_suffix(".png").exists()


def _build_frontier_runs(
    n_local: int = 50,
    include_frontier: bool = True,
    n_frontier: int = 30,
    seed: int = 7,
) -> list[dict]:
    """Build FULL_FRAMEWORK runs with local and (optionally) frontier models."""
    rng = random.Random(seed)
    runs: list[dict] = []

    for i in range(n_local):
        runs.append(_base_run(
            condition="FULL_FRAMEWORK",
            goal_type="research_synthesis",
            goal_id=f"g_local_{i:03d}",
            success=rng.randint(0, 1),
            model_llm="llama3.1:8b",
            tokens_total=rng.randint(2000, 6000),
            latency_s=rng.uniform(5.0, 30.0),
        ))

    if include_frontier:
        for i in range(n_frontier):
            runs.append(_base_run(
                condition="FULL_FRAMEWORK",
                goal_type="research_synthesis",
                goal_id=f"g_frontier_{i:03d}",
                success=rng.randint(0, 1),
                model_llm="claude-3-5-sonnet-20241022",
                tokens_total=rng.randint(1500, 5000),
                latency_s=rng.uniform(2.0, 15.0),
            ))

    return runs


class TestFrontier:
    def test_frontier_handles_missing_data(self, tmp_path: Path) -> None:
        """
        When no frontier runs exist, a placeholder PDF is created and
        the figure text includes "frontier replication not available".
        """
        runs = _build_frontier_runs(include_frontier=False)
        runs_path = _write_jsonl(runs, tmp_path)
        out_pdf = str(tmp_path / "fvl_missing.pdf")

        fig, _ = _capture_figure(frontier_generate, runs_path, out_pdf)

        assert Path(out_pdf).exists(), "PDF should be created even without frontier data"
        assert Path(out_pdf).with_suffix(".png").exists(), "PNG should also be created"

        assert fig is not None
        combined = _figure_texts(fig)
        assert "frontier replication not available" in combined, (
            f"Expected placeholder message in figure text.\n"
            f"Got: {combined[:500]}"
        )

    def test_frontier_generates_bars_when_data_present(self, tmp_path: Path) -> None:
        """
        With frontier runs, the figure has exactly 2 bar groups
        (local and frontier), each with 4 bars (one per metric).
        """
        runs = _build_frontier_runs(include_frontier=True)
        runs_path = _write_jsonl(runs, tmp_path)
        out_pdf = str(tmp_path / "fvl_full.pdf")

        fig, _ = _capture_figure(frontier_generate, runs_path, out_pdf)

        assert Path(out_pdf).exists()
        assert fig is not None

        ax = fig.get_axes()[0]
        # ax.containers: one BarContainer per ax.bar() call
        n_groups = len(ax.containers)
        assert n_groups == 2, (
            f"Expected 2 BarContainers (local + frontier), got {n_groups}"
        )
        for container in ax.containers:
            assert len(container) == 4, (
                f"Expected 4 bars per backend, got {len(container)}"
            )

    def test_frontier_pdf_and_png_created(self, tmp_path: Path) -> None:
        """generate() saves both PDF and PNG regardless of frontier presence."""
        runs = _build_frontier_runs(include_frontier=True, n_local=20, n_frontier=10)
        runs_path = _write_jsonl(runs, tmp_path)
        out_pdf = str(tmp_path / "fvl.pdf")

        frontier_generate(runs_path, out_pdf)

        assert Path(out_pdf).exists()
        assert Path(out_pdf).with_suffix(".png").exists()

    def test_frontier_missing_does_not_raise(self, tmp_path: Path) -> None:
        """generate() must not raise when no frontier runs are present."""
        runs = _build_frontier_runs(include_frontier=False, n_local=10)
        runs_path = _write_jsonl(runs, tmp_path)
        out_pdf = str(tmp_path / "fvl_safe.pdf")

        frontier_generate(runs_path, out_pdf)   # must not raise
        assert Path(out_pdf).exists()
