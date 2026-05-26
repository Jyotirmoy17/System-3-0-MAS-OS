"""Tests for mas/eval/robustness.py."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest

from mas.eval.robustness import (
    append_robustness_section,
    drop_adversarial_sensitivity,
    seed_sweep,
    temperature_sweep,
    threshold_sensitivity,
    _compute_cfr,
    _select_stratified,
)


def _make_run(
    goal_id: str,
    condition: str = "FULL_FRAMEWORK",
    seed: int = 0,
    *,
    cascade_failure: bool = False,
    goal_type: str = "research_synthesis",
) -> dict:
    """
    Build a minimal valid run-log dict (EXPERIMENTS.md §4).

    For detect_cascade() to return True the caller must also pass
    would_have_been_invalid=True and downstream_consumed=True on a step
    that has ground_truth_match=False.  The simpler cascade_failure=True
    flag is stored in the 'cascade_failure' field (int 0/1).
    """
    return {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            goal_id,
        "goal_type":          goal_type,
        "pipeline_depth_k":   3,
        "condition":          condition,
        "routing_skipped":    False,
        "seed":               seed,
        "model_slm":          "phi3:mini",
        "model_llm":          "llama3.1:8b",
        "started_at":         "2026-01-01T00:00:00Z",
        "completed_at":       "2026-01-01T00:01:00Z",
        "latency_s":          5.0,
        "success":            0 if cascade_failure else 1,
        "cascade_failure":    1 if cascade_failure else 0,
        "tokens":             {"slm": 100, "llm": 900, "total": 1000},
        "s0":                 {"catches": 0, "repairs": 0, "would_have_failed": 0},
        "escalations":        {"P1": 1 if cascade_failure else 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step":           [
            {
                "step_id":                       1,
                "agent_id":                      "WebAgent",
                "tier":                          1,
                "raw_output":                    "...",
                "schema_valid":                  True,
                "would_have_been_invalid":       cascade_failure,
                "downstream_consumed":           cascade_failure,
                "tokens_used":                   1000,
                "latency_ms":                    2000,
                "adversarial_corruption_applied": None,
            }
        ],
        "final_output":       "result",
        "ground_truth":       "result",
        "rubric_scores":      {"check_1": not cascade_failure},
        "ground_truth_match": not cascade_failure,
        "voc_trace":          [0.8],
        "crash_reason":       None,
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
    }


def _make_mock_run_one(cfr_flag: bool = False, goal_type: str = "research_synthesis"):
    """Return a mock run_one function that always returns a fixed RunLog."""
    def _mock(
        goal_id: str,
        condition: str,
        seed: int,
        model_overrides=None,
        *,
        _skip_warmup: bool = True,
        _goals_dir=None,
    ) -> dict:
        return _make_run(goal_id, condition, seed,
                         cascade_failure=cfr_flag, goal_type=goal_type)
    return _mock


def _write_goals_fixture(tmp_dir: Path, n_per_type: int = 3) -> None:
    """
    Write minimal goal JSON files under *tmp_dir*/goals/<type>/*.json.

    Creates a bare-minimum file for each goal so _select_stratified() can
    enumerate them without loading real goal content.
    """
    goal_types = [
        "research_synthesis",
        "financial_analysis",
        "sequential_reasoning",
        "code_analysis",
        "adversarial_reliability",
    ]
    for gt in goal_types:
        gt_dir = tmp_dir / "goals" / gt
        gt_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_type):
            goal_id = f"{gt[:8]}_{i:03d}"
            (gt_dir / f"{goal_id}.json").write_text(
                json.dumps({
                    "id": goal_id,
                    "goal_type": gt,
                    "goal_text": f"test goal {i}",
                    "inputs": {},
                    "ground_truth": "ok",
                    "rubric": [],
                    "pipeline_depth_k": 3,
                    "corruption_pattern": None,
                }),
                encoding="utf-8",
            )


class TestTemperatureSweep:
    def test_correct_cfr_computed(self, tmp_path: Path) -> None:
        """
        Mock run_one returns cascade_failure=False for t=0.0, True for t=0.7.
        Assert CFR for each temperature is computed correctly.
        """
        _write_goals_fixture(tmp_path)

        call_counts: dict[float, int] = {}

        def _mock(goal_id, condition, seed, model_overrides=None, *,
                  _skip_warmup=True, _goals_dir=None):
            temp = (model_overrides or {}).get("temperature", 0.0)
            call_counts[temp] = call_counts.get(temp, 0) + 1
            # t=0.7 always fails; t=0.0 always succeeds
            fail = temp == 0.7
            return _make_run(goal_id, condition, seed, cascade_failure=fail)

        result = temperature_sweep(
            n_instances=5,
            temperatures=[0.0, 0.7],
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=0,
            _run_one_fn=_mock,
        )

        assert "t=0.0" in result
        assert "t=0.7" in result
        assert result["t=0.0"]["cfr"] == pytest.approx(0.0)
        assert result["t=0.7"]["cfr"] == pytest.approx(1.0)
        assert result["delta_preserved"] is True   # 0.0 <= 1.0

    def test_returns_n_equal_to_instances(self, tmp_path: Path) -> None:
        """Each temperature bucket should have n == number of instances selected."""
        _write_goals_fixture(tmp_path, n_per_type=4)

        result = temperature_sweep(
            n_instances=10,
            temperatures=[0.0],
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=1,
            _run_one_fn=_make_mock_run_one(cfr_flag=False),
        )

        assert result["t=0.0"]["n"] == 10

    def test_resumable_skips_completed(self, tmp_path: Path) -> None:
        """Pre-populating the log means zero new run_one calls for those IDs."""
        _write_goals_fixture(tmp_path, n_per_type=2)

        # Pre-populate the log for t=0.0
        log_path = tmp_path / "runs_robustness_temperature_t0.0.jsonl"
        instance_ids = [
            "research_:0",  # placeholder — real IDs come from _select_stratified
        ]
        # Actually select what _select_stratified would pick to pre-populate correctly
        from mas.eval.robustness import _select_stratified as _sel
        ids = _sel(10, tmp_path / "goals", rng_seed=99)
        for gid in ids:
            log_path.write_text(
                json.dumps(_make_run(gid, "FULL_FRAMEWORK", 0)) + "\n",
                encoding="utf-8",
            )
            break  # Pre-populate just the first ID

        call_log: list[str] = []

        def _mock(goal_id, condition, seed, overrides=None, *,
                  _skip_warmup=True, _goals_dir=None):
            call_log.append(goal_id)
            return _make_run(goal_id, condition, seed)

        temperature_sweep(
            n_instances=10,
            temperatures=[0.0],
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=99,
            _run_one_fn=_mock,
        )

        assert ids[0] not in call_log


class TestSeedSweep:
    def test_correct_structure(self, tmp_path: Path) -> None:
        """seed_sweep returns keys 'seed=0', 'seed=1000', 'delta_preserved'."""
        _write_goals_fixture(tmp_path)

        result = seed_sweep(
            n_instances=5,
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=0,
            _run_one_fn=_make_mock_run_one(cfr_flag=False),
        )

        assert "seed=0"    in result
        assert "seed=1000" in result
        assert "delta_preserved" in result

    def test_cfr_values_match_mock(self, tmp_path: Path) -> None:
        """When all runs succeed, both seed buckets have CFR == 0."""
        _write_goals_fixture(tmp_path, n_per_type=2)

        result = seed_sweep(
            n_instances=5,
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=7,
            _run_one_fn=_make_mock_run_one(cfr_flag=False),
        )

        assert result["seed=0"]["cfr"]    == pytest.approx(0.0)
        assert result["seed=1000"]["cfr"] == pytest.approx(0.0)
        assert result["delta_preserved"]  is True

    def test_delta_preserved_false_when_large_gap(self, tmp_path: Path) -> None:
        """When CFR differs by > 20pp, delta_preserved is False."""
        _write_goals_fixture(tmp_path, n_per_type=4)

        call_counter: list[int] = [0]

        def _mock(goal_id, condition, seed, overrides=None, *,
                  _skip_warmup=True, _goals_dir=None):
            # seed=1000 always fails; seed=0 always succeeds
            fail = seed == 1000
            return _make_run(goal_id, condition, seed, cascade_failure=fail)

        result = seed_sweep(
            n_instances=10,
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=3,
            _run_one_fn=_mock,
        )

        assert result["seed=0"]["cfr"]    == pytest.approx(0.0)
        assert result["seed=1000"]["cfr"] == pytest.approx(1.0)
        assert result["delta_preserved"]  is False


class TestDropAdversarial:
    def _write_synthetic_runs(
        self,
        path: Path,
        n_non_adv: int = 4,
        n_adv: int = 4,
    ) -> None:
        """Write synthetic runs.jsonl with adversarial and non-adversarial runs."""
        with path.open("w", encoding="utf-8") as fh:
            for cond in ("FULL_FRAMEWORK", "BASELINE_LANGGRAPH"):
                for i in range(n_non_adv):
                    fh.write(json.dumps(_make_run(
                        f"goal_{i:03d}", cond, 42,
                        goal_type="research_synthesis",
                    )) + "\n")
                for i in range(n_adv):
                    fh.write(json.dumps(_make_run(
                        f"adv_{i:03d}", cond, 42,
                        goal_type="adversarial_reliability",
                        cascade_failure=True,
                    )) + "\n")

    def test_excludes_adversarial_runs(self, tmp_path: Path) -> None:
        """After exclusion, n_remaining does not include adversarial_reliability."""
        runs_path = tmp_path / "runs.jsonl"
        self._write_synthetic_runs(runs_path, n_non_adv=3, n_adv=3)

        result = drop_adversarial_sensitivity(str(runs_path))

        assert result["note"]       == "excluding adversarial"
        assert result["n_excluded"] == 6    # 3 adv × 2 conditions
        assert result["n_remaining"] == 6   # 3 non-adv × 2 conditions

    def test_cfr_by_condition_present(self, tmp_path: Path) -> None:
        """Result always contains 'cfr_by_condition'."""
        runs_path = tmp_path / "runs.jsonl"
        self._write_synthetic_runs(runs_path, n_non_adv=4, n_adv=2)

        result = drop_adversarial_sensitivity(str(runs_path))

        assert "cfr_by_condition" in result

    def test_non_adversarial_cfr_not_inflated(self, tmp_path: Path) -> None:
        """
        All non-adversarial runs succeed; all adversarial fail (cascade).
        Excluding adversarial should bring FULL_FRAMEWORK CFR to 0.
        """
        runs_path = tmp_path / "runs.jsonl"
        # Write: FULL_FRAMEWORK non-adv succeed; FULL_FRAMEWORK adv fail
        with runs_path.open("w", encoding="utf-8") as fh:
            for i in range(5):
                fh.write(json.dumps(_make_run(
                    f"g_{i:03d}", "FULL_FRAMEWORK", 42,
                    goal_type="research_synthesis", cascade_failure=False,
                )) + "\n")
                fh.write(json.dumps(_make_run(
                    f"adv_{i:03d}", "FULL_FRAMEWORK", 42,
                    goal_type="adversarial_reliability", cascade_failure=True,
                )) + "\n")

        result = drop_adversarial_sensitivity(str(runs_path))

        cfr_full = result["cfr_by_condition"].get("FULL_FRAMEWORK", {}).get("cfr", -1)
        assert cfr_full == pytest.approx(0.0), (
            f"Expected CFR=0 after excluding adversarial runs; got {cfr_full}"
        )

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Passing a non-existent runs.jsonl raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            drop_adversarial_sensitivity(str(tmp_path / "nonexistent.jsonl"))


class TestThresholdSensitivity:
    def test_all_three_taus_present(self, tmp_path: Path) -> None:
        """threshold_sensitivity returns keys for all requested taus."""
        _write_goals_fixture(tmp_path)

        result = threshold_sensitivity(
            taus=[0.5, 0.7, 0.9],
            n_instances=5,
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=0,
            _run_one_fn=_make_mock_run_one(cfr_flag=False),
        )

        assert "tau=0.5" in result
        assert "tau=0.7" in result
        assert "tau=0.9" in result

    def test_escalations_p1_in_output(self, tmp_path: Path) -> None:
        """Each tau bucket contains 'escalations_p1'."""
        _write_goals_fixture(tmp_path)

        result = threshold_sensitivity(
            taus=[0.5],
            n_instances=5,
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=0,
            _run_one_fn=_make_mock_run_one(cfr_flag=False),
        )

        assert "escalations_p1" in result["tau=0.5"]

    def test_n_correct_per_tau(self, tmp_path: Path) -> None:
        """Each tau bucket has n == number of instances requested."""
        _write_goals_fixture(tmp_path, n_per_type=4)

        result = threshold_sensitivity(
            taus=[0.9],
            n_instances=10,
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=2,
            _run_one_fn=_make_mock_run_one(cfr_flag=False),
        )

        assert result["tau=0.9"]["n"] == 10

    def test_cfr_varies_with_tau(self, tmp_path: Path) -> None:
        """When mock returns different CFR per tau, sweep reflects it."""
        _write_goals_fixture(tmp_path, n_per_type=4)

        def _mock(goal_id, condition, seed, model_overrides=None, *,
                  _skip_warmup=True, _goals_dir=None):
            tau = (model_overrides or {}).get("tau", 0.72)
            fail = tau >= 0.9
            return _make_run(goal_id, condition, seed, cascade_failure=fail)

        result = threshold_sensitivity(
            taus=[0.5, 0.9],
            n_instances=10,
            log_dir=tmp_path,
            goals_dir=tmp_path / "goals",
            rng_seed=5,
            _run_one_fn=_mock,
        )

        assert result["tau=0.5"]["cfr"] < result["tau=0.9"]["cfr"]


class TestAppendRobustnessSection:
    def test_creates_section_when_absent(self, tmp_path: Path) -> None:
        """append_robustness_section creates the section if RESULTS.md has none."""
        md_path = tmp_path / "RESULTS.md"
        md_path.write_text("# Experimental Results\n\nSome content.\n", encoding="utf-8")

        result = {"t=0.0": {"cfr": 0.1, "n": 50}, "delta_preserved": True}
        append_robustness_section(md_path, "Temperature Sweep", result)

        content = md_path.read_text(encoding="utf-8")
        assert "#### Temperature Sweep" in content
        assert "0.100" in content  # CFR formatted to 3dp

    def test_appends_under_existing_heading(self, tmp_path: Path) -> None:
        """If '### Robustness Checks' is present, sub-section is added under it."""
        md_path = tmp_path / "RESULTS.md"
        md_path.write_text(
            "# Results\n\n### Robustness Checks\n\n> Placeholder\n",
            encoding="utf-8",
        )

        result = {"seed=0": {"cfr": 0.05, "n": 50}, "seed=1000": {"cfr": 0.07, "n": 50},
                  "delta_preserved": True}
        append_robustness_section(md_path, "Seed Sweep", result)

        content = md_path.read_text(encoding="utf-8")
        assert "#### Seed Sweep" in content
        assert "### Robustness Checks" in content

    def test_replaces_existing_sub_section(self, tmp_path: Path) -> None:
        """Calling twice with same check_name replaces (not duplicates) the block."""
        md_path = tmp_path / "RESULTS.md"
        md_path.write_text(
            "# Results\n\n### Robustness Checks\n\n#### Drop-Adversarial\n\nOld text.\n\n",
            encoding="utf-8",
        )

        result = {"note": "excluding adversarial", "n_excluded": 30, "n_remaining": 120,
                  "cfr_by_condition": {}, "alpha": 0.05, "tests": {}}
        append_robustness_section(md_path, "Drop-Adversarial", result)
        append_robustness_section(md_path, "Drop-Adversarial", result)  # second call

        content = md_path.read_text(encoding="utf-8")
        assert content.count("#### Drop-Adversarial") == 1
        assert "Old text." not in content

    def test_creates_results_md_if_absent(self, tmp_path: Path) -> None:
        """append_robustness_section creates RESULTS.md if the file does not exist."""
        md_path = tmp_path / "RESULTS.md"
        assert not md_path.exists()

        result = {"tau=0.5": {"cfr": 0.2, "n": 50, "escalations_p1": 0.1}}
        append_robustness_section(md_path, "Threshold Sensitivity", result)

        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "#### Threshold Sensitivity" in content


class TestHelpers:
    def test_compute_cfr_empty(self) -> None:
        """_compute_cfr on an empty list returns (0.0, 0)."""
        cfr, n = _compute_cfr([])
        assert cfr == 0.0
        assert n == 0

    def test_compute_cfr_all_cascade(self) -> None:
        """_compute_cfr with all cascade failures returns (1.0, k)."""
        runs = [
            _make_run(f"g{i}", cascade_failure=True)
            for i in range(10)
        ]
        cfr, n = _compute_cfr(runs)
        assert cfr == pytest.approx(1.0)
        assert n == 10

    def test_select_stratified_count(self, tmp_path: Path) -> None:
        """_select_stratified returns exactly n_instances IDs (when enough goals)."""
        _write_goals_fixture(tmp_path, n_per_type=12)

        ids = _select_stratified(50, tmp_path / "goals", rng_seed=0)

        assert len(ids) == 50

    def test_select_stratified_fewer_goals(self, tmp_path: Path) -> None:
        """_select_stratified caps at available IDs when fewer than requested."""
        _write_goals_fixture(tmp_path, n_per_type=2)  # 10 total across 5 types

        ids = _select_stratified(50, tmp_path / "goals", rng_seed=0)

        # Should get at most 2 per type = 10 total
        assert len(ids) <= 10
