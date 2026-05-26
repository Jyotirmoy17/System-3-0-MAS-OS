"""Tests for mas/eval/figures/cascade_bar.py."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from mas.eval.figures.cascade_bar import generate
from mas.eval.cascade_detector import detect_cascade


def _make_run(
    goal_id: str,
    condition: str,
    seed: int = 42,
    *,
    cascade_failure: int = 0,
    success: int = 1,
    tokens_total: int = 1000,
    ground_truth_match: bool = True,
    would_have_been_invalid: bool = False,
    downstream_consumed: bool = False,
) -> dict:
    run = {
        "run_id": str(uuid.uuid4()),
        "goal_id": goal_id,
        "goal_type": "research_synthesis",
        "pipeline_depth_k": 3,
        "condition": condition,
        "routing_skipped": False,
        "seed": seed,
        "model_slm": "phi3:mini",
        "model_llm": "llama3.1:8b",
        "started_at": "2026-01-01T00:00:00",
        "completed_at": "2026-01-01T00:01:00",
        "latency_s": 10.0,
        "success": success,
        "cascade_failure": cascade_failure,
        "tokens": {
            "slm": 200,
            "llm": max(0, tokens_total - 200),
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
                "schema_valid": not would_have_been_invalid,
                "would_have_been_invalid": would_have_been_invalid,
                "downstream_consumed": downstream_consumed,
                "tokens_used": tokens_total,
                "latency_ms": 1000,
                "adversarial_corruption_applied": None,
            }
        ],
        "final_output": "result",
        "ground_truth": "expected",
        "rubric_scores": {"check_1": ground_truth_match},
        "ground_truth_match": ground_truth_match,
        "voc_trace": [0.8],
        "crash_reason": None,
        "grading_api_calls": 0,
        "grading_cost_usd": 0.0,
    }
    run["cascade_failure"] = int(detect_cascade(run))
    return run


_CONDITIONS = (
    "FULL_FRAMEWORK",
    "BASELINE_LANGGRAPH",
)


def _write_jsonl(runs: list[dict], tmpdir: str) -> str:
    path = os.path.join(tmpdir, "runs.jsonl")
    with open(path, "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    return path


def _build_runs(
    n: int = 150,
    *,
    cfr_full: float = 0.0,
    cfr_bl:   float = 0.0,
) -> list[dict]:
    """
    Build a paired synthetic dataset.

    For cascade detection to work the per_step flags must be set.
    When cfr_* > 0, the first cfr_* * n instances for that condition
    are cascade failures (would_have_been_invalid + downstream_consumed
    + ground_truth_match=False).
    """
    runs: list[dict] = []
    cfr_map = {
        "FULL_FRAMEWORK":     cfr_full,
        "BASELINE_LANGGRAPH": cfr_bl,
    }
    for i in range(n):
        gid = f"goal_{i:03d}"
        for cond in _CONDITIONS:
            n_cf = round(cfr_map[cond] * n)
            is_cf = i < n_cf
            runs.append(_make_run(
                gid, cond,
                cascade_failure=int(is_cf),
                success=int(not is_cf),
                ground_truth_match=not is_cf,
                would_have_been_invalid=is_cf,
                downstream_consumed=is_cf,
            ))
    return runs


class TestGeneratesPdf:
    def test_creates_pdf_and_png(self, tmp_path):
        """generate() creates both PDF and PNG files."""
        runs = _build_runs(n=150)
        runs_path = _write_jsonl(runs, str(tmp_path))
        out_pdf = str(tmp_path / "cascade_bar.pdf")

        generate(runs_path, out_pdf)

        assert Path(out_pdf).exists(), "PDF not created"
        assert Path(out_pdf).stat().st_size > 0, "PDF is empty"
        png_path = str(tmp_path / "cascade_bar.png")
        assert Path(png_path).exists(), "PNG not created"
        assert Path(png_path).stat().st_size > 0, "PNG is empty"

    def test_creates_parent_directories(self, tmp_path):
        """generate() creates nested output directories if they don't exist."""
        runs = _build_runs(n=50)
        runs_path = _write_jsonl(runs, str(tmp_path))
        out_pdf = str(tmp_path / "nested" / "deep" / "cascade_bar.pdf")

        generate(runs_path, out_pdf)

        assert Path(out_pdf).exists()
        assert Path(out_pdf).with_suffix(".png").exists()

    def test_no_full_framework_raises(self, tmp_path):
        """ValueError if FULL_FRAMEWORK has no runs."""
        runs = [_make_run(f"g{i}", "BASELINE_LANGGRAPH") for i in range(10)]
        runs_path = _write_jsonl(runs, str(tmp_path))
        out_pdf = str(tmp_path / "out.pdf")

        with pytest.raises(ValueError, match="FULL_FRAMEWORK"):
            generate(runs_path, out_pdf)


class TestPdfHasCorrectData:
    def _get_figure_texts(self, runs_path: str, out_pdf: str) -> list[str]:
        """
        Call generate() and intercept the figure before it is closed
        by patching plt.close. Returns a list of all text strings on the
        saved figure axes.
        """
        closed_fig: list[plt.Figure] = []
        original_close = plt.close

        def capture_close(fig=None):
            if fig is not None:
                closed_fig.append(fig)
            original_close(fig)

        import unittest.mock as mock
        with mock.patch("matplotlib.pyplot.close", side_effect=capture_close):
            generate(runs_path, out_pdf)

        if not closed_fig:
            return []
        fig = closed_fig[-1]
        texts = []
        for ax in fig.get_axes():
            for txt in ax.get_children():
                if hasattr(txt, "get_text"):
                    texts.append(txt.get_text())
        for txt in fig.texts:
            texts.append(txt.get_text())
        return texts

    def test_n150_in_caption(self, tmp_path):
        """The caption mentions n=150."""
        runs = _build_runs(n=150)
        runs_path = _write_jsonl(runs, str(tmp_path))
        out_pdf = str(tmp_path / "cb.pdf")

        texts = self._get_figure_texts(runs_path, out_pdf)
        combined = " ".join(texts)
        assert "n=150" in combined, f"n=150 not found in figure texts: {combined[:300]}"

    def test_title_present(self, tmp_path):
        """The figure title contains 'Cascade Failure Rate'."""
        runs = _build_runs(n=20)
        runs_path = _write_jsonl(runs, str(tmp_path))
        out_pdf = str(tmp_path / "cb.pdf")

        texts = self._get_figure_texts(runs_path, out_pdf)
        combined = " ".join(texts)
        assert "Cascade Failure Rate" in combined, \
            f"Title not found in figure texts: {combined[:300]}"


class TestHandlesNoSignificance:
    def test_ns_annotation_when_no_significant_tests(self, tmp_path):
        """
        When FULL and ablations have identical CFR (0%), no '*' stars appear.
        'ns' labels are present instead.
        """
        # All conditions: 0% CFR, 150 instances — no discordant pairs.
        runs = _build_runs(n=150, cfr_full=0.0, cfr_bl=0.0)
        runs_path = _write_jsonl(runs, str(tmp_path))
        out_pdf = str(tmp_path / "cb.pdf")

        import unittest.mock as mock
        closed_fig: list[plt.Figure] = []

        def cap(fig=None):
            if fig is not None: closed_fig.append(fig)
            plt.close.__wrapped__(fig) if hasattr(plt.close, "__wrapped__") else None

        original_close = plt.close
        with mock.patch("matplotlib.pyplot.close",
                        side_effect=lambda f=None: closed_fig.append(f) or original_close(f)):
            generate(runs_path, out_pdf)

        fig = closed_fig[-1]
        all_text = " ".join(
            t.get_text()
            for ax in fig.get_axes()
            for t in ax.get_children()
            if hasattr(t, "get_text")
        )
        assert "***" not in all_text
        assert "**" not in all_text
        assert "ns" in all_text


class TestHandlesSignificance:
    def test_triple_star_when_strong_effect(self, tmp_path):
        """
        FULL=0% CFR, ablations=50% CFR over 150 instances → *** annotations.
        """
        runs = _build_runs(n=150, cfr_full=0.0, cfr_bl=0.50)
        runs_path = _write_jsonl(runs, str(tmp_path))
        out_pdf = str(tmp_path / "cb.pdf")

        import unittest.mock as mock
        closed_fig: list[plt.Figure] = []
        original_close = plt.close
        with mock.patch("matplotlib.pyplot.close",
                        side_effect=lambda f=None: closed_fig.append(f) or original_close(f)):
            generate(runs_path, out_pdf)

        fig = closed_fig[-1]
        all_text = " ".join(
            t.get_text()
            for ax in fig.get_axes()
            for t in ax.get_children()
            if hasattr(t, "get_text")
        )
        assert "***" in all_text, \
            f"Expected '***' in figure text for strong effect, got: {all_text[:400]}"

    def test_double_star_mid_effect(self, tmp_path):
        """
        A moderate effect (p between 0.001 and 0.01) → '**' annotation.

        We engineer exactly 12 discordant pairs (FULL=0, ABL=1) out of 150.
        binomtest(0, 12, 0.5) p ≈ 0.0005 → *** actually.
        Use 4 discordant pairs instead: binomtest(0, 4) = 0.125 → ns.
        Use 8 pairs:  binomtest(0, 8) = 0.0039 → **.
        """
        n = 150
        # 8 instances where FULL=0, BASELINE=1
        runs: list[dict] = []
        for i in range(n):
            gid = f"g{i:03d}"
            runs.append(_make_run(gid, "FULL_FRAMEWORK", cascade_failure=0,
                                  would_have_been_invalid=False, downstream_consumed=False,
                                  ground_truth_match=True))
            is_cf = i < 8
            runs.append(_make_run(gid, "BASELINE_LANGGRAPH",
                                  would_have_been_invalid=is_cf,
                                  downstream_consumed=is_cf,
                                  ground_truth_match=not is_cf))

        runs_path = _write_jsonl(runs, str(tmp_path))
        out_pdf = str(tmp_path / "cb.pdf")

        import unittest.mock as mock
        closed_fig: list[plt.Figure] = []
        original_close = plt.close
        with mock.patch("matplotlib.pyplot.close",
                        side_effect=lambda f=None: closed_fig.append(f) or original_close(f)):
            generate(runs_path, out_pdf)

        fig = closed_fig[-1]
        all_text = " ".join(
            t.get_text()
            for ax in fig.get_axes()
            for t in ax.get_children()
            if hasattr(t, "get_text")
        )
        assert "**" in all_text, \
            f"Expected '**' for 8 discordant pairs, got: {all_text[:400]}"

    def test_reproducible_output(self, tmp_path):
        """generate() is reproducible: same input → identical file bytes."""
        runs = _build_runs(n=50, cfr_full=0.0, cfr_bl=0.30)
        runs_path = _write_jsonl(runs, str(tmp_path))

        out1 = str(tmp_path / "run1" / "cb.pdf")
        out2 = str(tmp_path / "run2" / "cb.pdf")
        generate(runs_path, out1)
        generate(runs_path, out2)

        assert Path(out1).read_bytes() == Path(out2).read_bytes(), \
            "Two runs with identical input produced different PDF bytes"
