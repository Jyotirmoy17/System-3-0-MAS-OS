"""Tests for mas/eval/statistics.py."""
from __future__ import annotations

import json
import math
import os
import tempfile
import uuid

import numpy as np
import pytest

from mas.eval.statistics import (
    bootstrap_ci_paired,
    cliffs_delta,
    cohens_h,
    compute_cfr_by_condition,
    mcnemar_paired,
    primary_analysis,
    wilcoxon_paired,
)


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
    """
    Build a minimal synthetic run-log dict.

    For detect_cascade() to return True the caller must pass:
        ground_truth_match=False,
        would_have_been_invalid=True,
        downstream_consumed=True.
    """
    return {
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
                "schema_valid": True,
                "would_have_been_invalid": would_have_been_invalid,
                "downstream_consumed": downstream_consumed,
                "tokens_used": tokens_total,
                "latency_ms": 1000,
                "adversarial_corruption_applied": None,
            }
        ],
        "final_output": "result",
        "ground_truth": "result",
        "rubric_scores": {"check_1": True},
        "ground_truth_match": ground_truth_match,
        "voc_trace": [0.8],
        "crash_reason": None,
        "grading_api_calls": 0,
        "grading_cost_usd": 0.0,
    }


class TestComputeCFRBasic:
    def test_cfr_ten_percent(self):
        """10 runs with true cascade conditions out of 100 → CFR = 0.10."""
        runs: list[dict] = []
        # 10 true cascade failures: ground_truth_match=False + contaminated step
        for i in range(10):
            runs.append(_make_run(
                f"goal_{i:03d}", "FULL_FRAMEWORK",
                ground_truth_match=False,
                would_have_been_invalid=True,
                downstream_consumed=True,
            ))
        # 90 clean runs
        for i in range(10, 100):
            runs.append(_make_run(f"goal_{i:03d}", "FULL_FRAMEWORK"))

        result = compute_cfr_by_condition(runs)

        assert "FULL_FRAMEWORK" in result
        r = result["FULL_FRAMEWORK"]
        assert r["n"] == 100
        assert abs(r["cfr"] - 0.10) < 1e-9
        assert r["ci_low"] < 0.10
        assert r["ci_high"] > 0.10

    def test_zero_cfr(self):
        """All clean runs → CFR = 0.0; CI lower bound is effectively 0."""
        runs = [_make_run(f"g{i}", "BASELINE_LANGGRAPH") for i in range(50)]
        result = compute_cfr_by_condition(runs)
        assert result["BASELINE_LANGGRAPH"]["cfr"] == 0.0
        assert result["BASELINE_LANGGRAPH"]["ci_low"] == pytest.approx(0.0, abs=1e-12)

    def test_multiple_conditions_grouped(self):
        """Runs from two conditions are grouped independently."""
        runs = (
            [_make_run(f"g{i}", "FULL_FRAMEWORK") for i in range(20)]
            + [_make_run(f"g{i}", "BASELINE_LANGGRAPH") for i in range(20)]
        )
        result = compute_cfr_by_condition(runs)
        assert set(result.keys()) == {"FULL_FRAMEWORK", "BASELINE_LANGGRAPH"}
        assert result["FULL_FRAMEWORK"]["n"] == 20
        assert result["BASELINE_LANGGRAPH"]["n"] == 20


class TestMcNemarSignificant:
    def test_significant(self):
        """FULL=0% vs BASELINE=50% cascade_failure over 100 pairs → p < 0.001."""
        full_runs: list[dict] = []
        base_runs: list[dict] = []

        for i in range(100):
            gid = f"goal_{i:03d}"
            full_runs.append(_make_run(gid, "FULL_FRAMEWORK", cascade_failure=0))
            if i < 50:
                base_runs.append(
                    _make_run(gid, "BASELINE_LANGGRAPH", cascade_failure=1)
                )
            else:
                base_runs.append(_make_run(gid, "BASELINE_LANGGRAPH", cascade_failure=0))

        result = mcnemar_paired(full_runs, base_runs, "cascade_failure")

        assert result["p_value"] < 0.001
        assert result["n"] == 100
        assert result["discordant_pairs"] == (0, 50)


class TestMcNemarNotSignificant:
    def test_not_significant(self):
        """FULL=10% vs BASELINE=12% cascade_failure over 100 pairs → p > 0.05."""
        full_runs: list[dict] = []
        base_runs: list[dict] = []

        for i in range(100):
            gid = f"goal_{i:03d}"
            full_cf = 1 if i < 10 else 0
            base_cf = 1 if i < 12 else 0
            full_runs.append(_make_run(gid, "FULL_FRAMEWORK", cascade_failure=full_cf))
            base_runs.append(_make_run(gid, "BASELINE_LANGGRAPH", cascade_failure=base_cf))

        result = mcnemar_paired(full_runs, base_runs, "cascade_failure")

        assert result["p_value"] > 0.05


class TestWilcoxonTokenDifference:
    def test_median_difference_recovered(self):
        """FULL=1000 tokens, BASELINE=1500 tokens: median diff = −500, p < 0.001."""
        full_runs: list[dict] = []
        base_runs: list[dict] = []

        for i in range(50):
            gid = f"goal_{i:03d}"
            full_runs.append(_make_run(gid, "FULL_FRAMEWORK", tokens_total=1000))
            base_runs.append(_make_run(gid, "BASELINE_LANGGRAPH", tokens_total=1500))

        result = wilcoxon_paired(full_runs, base_runs, "tokens.total")

        assert result["p_value"] < 0.001
        assert abs(result["median_diff"] - (-500.0)) < 1e-6
        assert result["n"] == 50

    def test_no_difference_returns_p_one(self):
        """Identical token counts → all diffs zero → p_value = 1.0."""
        runs_a = [_make_run(f"g{i}", "FULL_FRAMEWORK", tokens_total=800) for i in range(30)]
        runs_b = [_make_run(f"g{i}", "BASELINE_LANGGRAPH", tokens_total=800) for i in range(30)]

        result = wilcoxon_paired(runs_a, runs_b, "tokens.total")

        assert result["p_value"] == 1.0
        assert result["median_diff"] == 0.0

    def test_nested_dot_notation(self):
        """tokens.total is accessed via dot notation without error."""
        runs_a = [_make_run(f"g{i}", "FULL_FRAMEWORK", tokens_total=500) for i in range(10)]
        runs_b = [_make_run(f"g{i}", "BASELINE_LANGGRAPH", tokens_total=700) for i in range(10)]

        result = wilcoxon_paired(runs_a, runs_b, "tokens.total")
        assert result["median_diff"] == pytest.approx(-200.0)


class TestBootstrapCICoverage:
    def test_ci_contains_true_mean_diff(self):
        """
        Bootstrap CI must contain the true mean difference.

        True diff = 300 tokens (FULL=1300, COMP=1000), so CI should
        straddle 300.
        """
        n = 200
        full_runs = [_make_run(f"g{i}", "FULL_FRAMEWORK", tokens_total=1300) for i in range(n)]
        comp_runs = [_make_run(f"g{i}", "BASELINE_LANGGRAPH", tokens_total=1000) for i in range(n)]

        lower, upper = bootstrap_ci_paired(
            full_runs, comp_runs, "tokens.total", n_resamples=5_000, seed=0
        )
        true_diff = 300.0
        # When all diffs are identical the CI collapses to a point; use <=.
        assert lower <= true_diff <= upper

    def test_ci_is_reproducible(self):
        """Same seed → identical CI."""
        runs_a = [_make_run(f"g{i}", "FULL_FRAMEWORK", tokens_total=1200) for i in range(50)]
        runs_b = [_make_run(f"g{i}", "BASELINE_LANGGRAPH", tokens_total=1000) for i in range(50)]

        ci1 = bootstrap_ci_paired(runs_a, runs_b, "tokens.total", n_resamples=1_000, seed=7)
        ci2 = bootstrap_ci_paired(runs_a, runs_b, "tokens.total", n_resamples=1_000, seed=7)
        assert ci1 == ci2


class TestCohensHKnown:
    def test_equal_proportions_gives_zero(self):
        """cohens_h(p, p) == 0 for any p."""
        assert cohens_h(0.5, 0.5) == pytest.approx(0.0, abs=1e-12)
        assert cohens_h(0.1, 0.1) == pytest.approx(0.0, abs=1e-12)
        assert cohens_h(0.9, 0.9) == pytest.approx(0.0, abs=1e-12)

    def test_extreme_proportions_give_pi(self):
        """cohens_h(1.0, 0.0) == π (maximum possible effect)."""
        h = cohens_h(1.0, 0.0)
        assert h == pytest.approx(math.pi, abs=1e-10)
        assert abs(h) > 0.8  # well above "large" threshold

    def test_sign_convention(self):
        """h is positive when p1 > p2."""
        assert cohens_h(0.8, 0.2) > 0
        assert cohens_h(0.2, 0.8) < 0


class TestPairingFailsLoudly:
    def test_missing_partner_raises_value_error(self):
        """Runs with disjoint goal_ids raise ValueError naming the missing key."""
        runs_a = [_make_run("goal_001", "FULL_FRAMEWORK")]
        runs_b = [_make_run("goal_002", "BASELINE_LANGGRAPH")]

        with pytest.raises(ValueError, match="goal_001"):
            mcnemar_paired(runs_a, runs_b, "cascade_failure")

    def test_duplicate_in_runs_b_raises(self):
        """Duplicate (goal_id, seed) in runs_b raises ValueError."""
        runs_a = [_make_run("goal_001", "FULL_FRAMEWORK")]
        runs_b = [
            _make_run("goal_001", "BASELINE_LANGGRAPH"),
            _make_run("goal_001", "BASELINE_LANGGRAPH"),  # duplicate
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            mcnemar_paired(runs_a, runs_b, "cascade_failure")

    def test_extra_runs_in_b_raises(self):
        """runs_b has more unique keys than runs_a → ValueError."""
        runs_a = [_make_run("goal_001", "FULL_FRAMEWORK")]
        runs_b = [
            _make_run("goal_001", "BASELINE_LANGGRAPH"),
            _make_run("goal_002", "BASELINE_LANGGRAPH"),
        ]
        with pytest.raises(ValueError):
            mcnemar_paired(runs_a, runs_b, "cascade_failure")

    def test_wilcoxon_also_raises_on_bad_pairing(self):
        """wilcoxon_paired propagates pairing errors too."""
        runs_a = [_make_run("goal_A", "FULL_FRAMEWORK")]
        runs_b = [_make_run("goal_B", "BASELINE_LANGGRAPH")]

        with pytest.raises(ValueError):
            wilcoxon_paired(runs_a, runs_b, "tokens.total")


class TestPrimaryAnalysisFull:
    def _build_synthetic_runs(self, n_instances: int = 150) -> list[dict]:
        """
        Build a synthetic 300-run dataset (2 conditions × 150 instances).

        FULL_FRAMEWORK: cascade_failure=0, success=1, tokens=1000.
        BASELINE_LANGGRAPH: cascade_failure=1 for first 15, success=0 for first 15,
                            tokens=1500 for all.
        """
        runs: list[dict] = []
        for i in range(n_instances):
            gid = f"goal_{i:03d}"
            runs.append(_make_run(
                gid, "FULL_FRAMEWORK",
                cascade_failure=0, success=1, tokens_total=1000,
            ))
            cf = 1 if i < 15 else 0
            sc = 0 if i < 15 else 1
            runs.append(_make_run(
                gid, "BASELINE_LANGGRAPH",
                cascade_failure=cf, success=sc, tokens_total=1500,
            ))
        return runs

    def test_three_tests_returned(self):
        """primary_analysis returns exactly 3 paired test results."""
        runs = self._build_synthetic_runs()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for run in runs:
                f.write(json.dumps(run) + "\n")
            tmppath = f.name

        try:
            result = primary_analysis(tmppath)
        finally:
            os.unlink(tmppath)

        assert len(result["tests"]) == 3

    def test_required_keys_present(self):
        """Every test result has p_value, statistic, effect_size, ci."""
        runs = self._build_synthetic_runs()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for run in runs:
                f.write(json.dumps(run) + "\n")
            tmppath = f.name

        try:
            result = primary_analysis(tmppath)
        finally:
            os.unlink(tmppath)

        expected_test_keys = {
            "BASELINE_LANGGRAPH_cfr",
            "BASELINE_LANGGRAPH_success",
            "BASELINE_LANGGRAPH_tokens",
        }
        assert set(result["tests"].keys()) == expected_test_keys

        for key, test in result["tests"].items():
            assert "p_value" in test,       f"Missing p_value in {key}"
            assert "statistic" in test,     f"Missing statistic in {key}"
            assert "effect_size" in test,   f"Missing effect_size in {key}"
            assert "ci" in test,            f"Missing ci in {key}"
            assert isinstance(test["ci"], tuple), f"ci is not a tuple in {key}"
            assert len(test["ci"]) == 2,    f"ci must have 2 elements in {key}"
            assert "significant" in test,   f"Missing significant in {key}"

    def test_alpha_correct(self):
        """alpha == 0.05 in the output."""
        runs = self._build_synthetic_runs(n_instances=10)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for run in runs:
                f.write(json.dumps(run) + "\n")
            tmppath = f.name

        try:
            result = primary_analysis(tmppath)
        finally:
            os.unlink(tmppath)

        assert abs(result["alpha"] - 0.05) < 1e-12

    def test_cfr_by_condition_in_output(self):
        """primary_analysis output includes cfr_by_condition for both conditions."""
        runs = self._build_synthetic_runs()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for run in runs:
                f.write(json.dumps(run) + "\n")
            tmppath = f.name

        try:
            result = primary_analysis(tmppath)
        finally:
            os.unlink(tmppath)

        assert "cfr_by_condition" in result
        for cond in ("FULL_FRAMEWORK", "BASELINE_LANGGRAPH"):
            assert cond in result["cfr_by_condition"]

    def test_significant_token_difference_detected(self):
        """FULL=1000 tokens vs BASELINE=1500 → Wilcoxon p < 0.05."""
        runs = self._build_synthetic_runs()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for run in runs:
                f.write(json.dumps(run) + "\n")
            tmppath = f.name

        try:
            result = primary_analysis(tmppath)
        finally:
            os.unlink(tmppath)

        test = result["tests"]["BASELINE_LANGGRAPH_tokens"]
        assert test["p_value"] < 0.05
        assert test["significant"] is True


class TestCliffsDelta:
    def test_all_a_greater_gives_plus_one(self):
        """If every a > every b, delta = +1."""
        a = [10, 20, 30]
        b = [1, 2, 3]
        assert cliffs_delta(a, b) == pytest.approx(1.0)

    def test_all_b_greater_gives_minus_one(self):
        """If every b > every a, delta = −1."""
        a = [1, 2, 3]
        b = [10, 20, 30]
        assert cliffs_delta(a, b) == pytest.approx(-1.0)

    def test_identical_distributions_gives_zero(self):
        """Identical distributions → delta ≈ 0."""
        vals = list(range(1, 21))
        assert cliffs_delta(vals, vals) == pytest.approx(0.0)
