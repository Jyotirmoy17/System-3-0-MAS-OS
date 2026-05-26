"""Tests for mas/eval/preflight.py::run_preflight()."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mas.eval.preflight import run_preflight

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_SCHEMA_SRC = _REPO_ROOT / "goals" / "_schema.json"

_SUBDIRS = {
    "research_synthesis":    "research_synthesis",
    "financial_analysis":    "financial_analysis",
    "sequential_reasoning":  "sequential_reasoning",
    "code_analysis":         "code_analysis",
    "adversarial_reliability": "adversarial_reliability",
}

_ID_PREFIXES = {
    "research_synthesis":    "research_synth",
    "financial_analysis":    "finance",
    "sequential_reasoning":  "sequential",
    "code_analysis":         "code",
    "adversarial_reliability": "adversarial",
}

_FIVE_PATTERN_NAMES: list[str] = [
    "drop_required_field",
    "type_flip",
    "out_of_range",
    "malformed_json",
    "extra_fields_not_in_schema",
]


def _adv_pattern(idx: int) -> str:
    """Corruption pattern for adversarial instance index *idx* (1-based, EXPERIMENTS §3)."""
    return _FIVE_PATTERN_NAMES[(idx - 1) % 5]


def _make_instance(
    goal_type: str,
    depth: int,
    idx: int,
    *,
    placeholder: bool = False,
    ground_truth: str = "",
    rubric: list | None = None,
    goal_text: str = "",
    corruption_pattern: str | None = None,
) -> dict:
    """Build a minimal valid goal instance dict."""
    prefix = _ID_PREFIXES[goal_type]
    inst_id = f"{prefix}_{idx:03d}"

    # Sensible defaults for a complete (non-placeholder) instance
    if not placeholder:
        if not goal_text:
            goal_text = f"Test goal for {goal_type} instance {idx}."
        if not ground_truth:
            if goal_type in ("financial_analysis", "sequential_reasoning"):
                ground_truth = str(float(idx * 10))   # numeric gt
            else:
                ground_truth = f"Expected answer for {inst_id}."
        if rubric is None:
            rubric = [
                {
                    "check_id":    "c1",
                    "check_text":  "Mentions key term",
                    "check_type":  "substring",
                    "check_params": {"field": "answer", "must_contain": ["answer"]},
                },
                {
                    "check_id":    "c2",
                    "check_text":  "Non-empty answer",
                    "check_type":  "substring",
                    "check_params": {"field": "answer", "must_contain": [""]},
                },
            ]

    # Adversarial instances need a corruption pattern
    if goal_type == "adversarial_reliability" and corruption_pattern is None and not placeholder:
        corruption_pattern = _adv_pattern(idx)

    result: dict = {
        "id":                 inst_id,
        "goal_type":          goal_type,
        "pipeline_depth_k":   depth,
        "goal_text":          goal_text,
        "inputs":             {"key": "value"},
        "ground_truth":       ground_truth,
        "rubric":             rubric or [],
        "corruption_pattern": corruption_pattern,
    }
    if placeholder:
        result["_placeholder"] = True

    return result


def _write_instance(goals_dir: Path, data: dict) -> None:
    """Write one instance JSON file into the correct subdirectory."""
    gt      = data["goal_type"]
    subdir  = goals_dir / _SUBDIRS[gt]
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / f"{data['id']}.json").write_text(json.dumps(data, indent=2))


def _setup_goals_dir(tmp_path: Path) -> Path:
    """Create a temp goals/ with _schema.json."""
    goals = tmp_path / "goals"
    goals.mkdir()
    shutil.copy(_SCHEMA_SRC, goals / "_schema.json")
    return goals


def _build_suite(
    goals_dir: Path,
    depths_per_type: dict[str, list[tuple[int, int]]],
) -> None:
    """
    Populate goals_dir with instances according to depths_per_type.

    depths_per_type: {goal_type: [(pipeline_depth_k, count), ...]}
    """
    global_idx: dict[str, int] = {gt: 1 for gt in _SUBDIRS}

    for goal_type, specs in depths_per_type.items():
        for depth, count in specs:
            for _ in range(count):
                idx  = global_idx[goal_type]
                data = _make_instance(goal_type, depth, idx)
                _write_instance(goals_dir, data)
                global_idx[goal_type] += 1


# Canonical depth spec for the 150-instance CFR-stress suite (EXPERIMENTS.md §3)
_UNIFIED_SUITE_SPEC = {
    "sequential_reasoning":     [(3, 9), (4, 9), (5, 9)],
    "code_analysis":            [(3, 11), (4, 11), (5, 10)],
    "adversarial_reliability":  [(3, 31), (4, 30), (5, 30)],
}


def test_preflight_passes_complete_suite(tmp_path: Path) -> None:
    """
    A minimal valid suite — one fully authored instance per goal type —
    must pass (no hard errors).  Distribution is incomplete (5 < 150)
    so distribution checks are skipped (soft warning only).
    """
    goals = _setup_goals_dir(tmp_path)

    depths = [3, 4, 5, 3, 4]
    for i, (goal_type, depth) in enumerate(zip(list(_SUBDIRS), depths), start=1):
        data = _make_instance(goal_type, depth, idx=i)
        _write_instance(goals, data)

    report = run_preflight(goals_dir=goals)

    assert report.passed is True, (
        f"Expected passed=True for complete 5-instance suite; "
        f"errors: {report.errors}"
    )
    # The only acceptable warning is the incomplete-suite distribution notice
    for w in report.warnings:
        assert "incomplete" in w.lower() or "distribution" in w.lower(), (
            f"Unexpected warning: {w}"
        )


def test_preflight_fails_missing_field(tmp_path: Path) -> None:
    """Instance with empty ground_truth must generate a hard error."""
    goals = _setup_goals_dir(tmp_path)

    bad = _make_instance("research_synthesis", 3, 1)
    bad["ground_truth"] = ""   # clear the populated value
    _write_instance(goals, bad)

    report = run_preflight(goals_dir=goals)

    assert report.passed is False
    assert any("ground_truth" in e for e in report.errors), (
        f"Expected ground_truth error; got: {report.errors}"
    )


def test_preflight_fails_shallow_depth(tmp_path: Path) -> None:
    """pipeline_depth_k < 3 must produce a hard error."""
    goals = _setup_goals_dir(tmp_path)

    shallow = _make_instance("research_synthesis", 2, 1)
    _write_instance(goals, shallow)

    report = run_preflight(goals_dir=goals)

    assert report.passed is False
    assert any("pipeline_depth_k=2" in e for e in report.errors)


def test_preflight_warns_incomplete_suite(tmp_path: Path) -> None:
    """A partial suite warns about incomplete distribution checks."""
    goals = _setup_goals_dir(tmp_path)

    for gt in list(_SUBDIRS)[:2]:
        _write_instance(goals, _make_instance(gt, 3, 1))

    report = run_preflight(goals_dir=goals)

    assert any("incomplete" in w.lower() for w in report.warnings)
    depth_errors = [e for e in report.errors if "pipeline_depth_k" in e]
    assert not depth_errors


def test_preflight_fails_invalid_corruption_distribution(tmp_path: Path) -> None:
    """
    Skewed adversarial corruption distribution (not 15×5) must error.
    """
    goals = _setup_goals_dir(tmp_path)

    # Build a full suite (correct strata for no strata errors)
    _build_suite(goals, _UNIFIED_SUITE_SPEC)

    # Overwrite adversarial instances with wrong pattern distribution:
    # drop_required_field: 5 instead of 6
    # out_of_range: 7 instead of 6
    adv_dir = goals / "adversarial_reliability"
    for f in sorted(adv_dir.glob("*.json")):
        d = json.loads(f.read_text())
        idx = int(d["id"].split("_")[-1])
        if idx == 6:
            # Remove one drop_required_field and reassign to out_of_range
            d["corruption_pattern"] = "out_of_range"
        f.write_text(json.dumps(d, indent=2))

    report = run_preflight(goals_dir=goals)

    assert report.passed is False
    pat_errors = [e for e in report.errors if "pattern" in e or "adversarial" in e.lower()]
    assert len(pat_errors) >= 2, (
        f"Expected ≥2 pattern distribution errors; got: {pat_errors}"
    )
    assert any("drop_required_field" in e for e in pat_errors)
    assert any("out_of_range" in e for e in pat_errors)
