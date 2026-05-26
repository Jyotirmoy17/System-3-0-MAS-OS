"""Tests for mas/eval/goal_loader.py."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema
import pytest

from mas.eval.goal_loader import (
    GoalSpec,
    ValidationReport,
    load_all_goals,
    load_goal,
    validate_goal_suite,
)

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_GOALS_DIR  = _REPO_ROOT / "goals"
_SCHEMA_SRC = _GOALS_DIR / "_schema.json"


def _copy_schema(dest_dir: Path) -> None:
    """Copy the canonical _schema.json into a temp goals directory."""
    shutil.copy(_SCHEMA_SRC, dest_dir / "_schema.json")


def _make_goal_dir(tmp_path: Path) -> Path:
    """Return a temporary goals/ root with the schema copied in."""
    goals = tmp_path / "goals"
    goals.mkdir()
    _copy_schema(goals)
    return goals


def _write_instance(goals_dir: Path, subdir: str, data: dict) -> Path:
    """Write a goal instance JSON file and return its path."""
    d = goals_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{data['id']}.json"
    p.write_text(json.dumps(data, indent=2))
    return p


def _complete_instance(
    inst_id:            str   = "research_synth_001",
    goal_type:          str   = "research_synthesis",
    corruption_pattern        = None,
    pipeline_depth_k:   int   = 4,
) -> dict:
    """Return a fully authored (non-TBD) goal instance dict."""
    return {
        "id":                 inst_id,
        "goal_type":          goal_type,
        "pipeline_depth_k":   pipeline_depth_k,
        "goal_text":          "Explain the attention mechanism in transformers.",
        "inputs":             {"context": "test input"},
        "ground_truth":       "The attention mechanism weights tokens.",
        "rubric": [
            {
                "check_id":    "c1",
                "check_text":  "Mentions attention",
                "check_type":  "substring",
                "check_params": {"field": "answer", "must_contain": ["attention"]},
            },
            {
                "check_id":    "c2",
                "check_text":  "Confidence reported",
                "check_type":  "substring",
                "check_params": {"field": "answer", "must_contain": ["weight"]},
            },
        ],
        "corruption_pattern": corruption_pattern,
    }


def test_load_goal_valid(tmp_path: Path) -> None:
    """A complete, schema-valid instance loads without error."""
    goals = _make_goal_dir(tmp_path)
    data  = _complete_instance()
    _write_instance(goals, "research_synthesis", data)

    spec = load_goal("research_synth_001", goals_dir=goals)

    assert isinstance(spec, GoalSpec)
    assert spec.id               == "research_synth_001"
    assert spec.goal_type        == "research_synthesis"
    assert spec.pipeline_depth_k == 4
    assert spec.goal_text        != ""
    assert spec.ground_truth     != ""
    assert len(spec.rubric)      == 2
    assert spec.corruption_pattern is None
    assert not spec.is_tbd()


def test_load_goal_invalid_schema(tmp_path: Path) -> None:
    """Instance missing a required field must raise jsonschema.ValidationError."""
    goals = _make_goal_dir(tmp_path)
    data  = _complete_instance()
    del data["goal_text"]       # violate required field
    _write_instance(goals, "research_synthesis", data)

    with pytest.raises(jsonschema.ValidationError, match="goal_text"):
        load_goal("research_synth_001", goals_dir=goals)


def test_validate_suite_reports_tbd() -> None:
    """
    After synchronization to runs.jsonl, all 100 instances have _placeholder: false.
    The suite must report 0 placeholders, 0 TBD, and 0 hard violations.
    """
    from mas.eval.goal_loader import EXPECTED_TOTAL

    report = validate_goal_suite(goals_dir=_GOALS_DIR)

    assert report.total == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL} instances, found {report.total}"
    )
    assert report.placeholder_count == 0,   f"Expected 0 placeholders after authoring, found {report.placeholder_count}"
    assert report.tbd_count         == 0,   f"Expected 0 TBD after authoring, found {report.tbd_count}"
    assert report.hard_violation_count == 0, (
        f"Unexpected hard violations: {report.schema_errors + report.pattern_errors + report.goal_distribution_errors}"
    )


def test_corruption_pattern_distribution() -> None:
    """Each of the 5 adversarial corruption patterns must appear exactly 15 times."""
    report = validate_goal_suite(goals_dir=_GOALS_DIR)

    assert report.pattern_errors == [], (
        f"Pattern distribution violations: {report.pattern_errors}"
    )

    # Double-check by loading all adversarial instances directly
    adv_specs = [
        s for s in load_all_goals(goals_dir=_GOALS_DIR)
        if s.goal_type == "adversarial_reliability"
    ]
    from mas.eval.goal_loader import (
        EXPECTED_ADV_PATTERN_COUNTS,
        EXPECTED_GOAL_TYPE_COUNTS,
    )

    assert len(adv_specs) == EXPECTED_GOAL_TYPE_COUNTS["adversarial_reliability"]

    from collections import Counter
    counts = Counter(s.corruption_pattern for s in adv_specs)
    for pattern, expected in EXPECTED_ADV_PATTERN_COUNTS.items():
        assert counts[pattern] == expected, (
            f"Pattern '{pattern}': expected {expected}, got {counts[pattern]}"
        )


def test_non_adversarial_corruption_null(tmp_path: Path) -> None:
    """
    A research_synth instance with corruption_pattern != null must fail
    schema validation (only adversarial goals may have a non-null pattern).
    """
    goals = _make_goal_dir(tmp_path)
    data  = _complete_instance(
        inst_id            = "research_synth_001",
        goal_type          = "research_synthesis",
        corruption_pattern = "drop_required_field",   # must be null for non-adversarial
    )
    _write_instance(goals, "research_synthesis", data)

    with pytest.raises(jsonschema.ValidationError):
        load_goal("research_synth_001", goals_dir=goals)


def test_adversarial_corruption_required(tmp_path: Path) -> None:
    """
    An adversarial instance with corruption_pattern=null must fail schema
    validation (adversarial goals MUST have a corruption pattern).
    """
    goals = _make_goal_dir(tmp_path)
    data  = _complete_instance(
        inst_id            = "adversarial_001",
        goal_type          = "adversarial_reliability",
        corruption_pattern = None,   # must be non-null for adversarial
    )
    _write_instance(goals, "adversarial_reliability", data)

    with pytest.raises(jsonschema.ValidationError):
        load_goal("adversarial_001", goals_dir=goals)


def test_load_all_goals_count() -> None:
    """load_all_goals must return exactly EXPECTED_TOTAL GoalSpec instances."""
    from mas.eval.goal_loader import EXPECTED_TOTAL

    specs = load_all_goals(goals_dir=_GOALS_DIR)
    assert len(specs) == EXPECTED_TOTAL
    ids = [s.id for s in specs]
    assert len(ids) == len(set(ids))


def test_is_tbd_scaffolded(tmp_path: Path) -> None:
    """Scaffolded (empty) instances are flagged as TBD; complete ones are not."""
    goals = _make_goal_dir(tmp_path)

    # Scaffolded TBD instance
    tbd_data = {
        "id": "finance_001", "goal_type": "financial_analysis",
        "pipeline_depth_k": 0,
        "goal_text": "", "inputs": {}, "ground_truth": "",
        "rubric": [], "corruption_pattern": None,
    }
    _write_instance(goals, "financial_analysis", tbd_data)
    tbd_spec = load_goal("finance_001", goals_dir=goals)
    assert tbd_spec.is_tbd()

    # Complete authored instance (reuse same subdir, different id)
    complete_data = _complete_instance("finance_002", "financial_analysis")
    _write_instance(goals, "financial_analysis", complete_data)
    complete_spec = load_goal("finance_002", goals_dir=goals)
    assert not complete_spec.is_tbd()


def test_placeholder_field_recognized(tmp_path: Path) -> None:
    """
    An instance with '_placeholder': true must load with is_placeholder()=True
    and is_tbd()=False (placeholder takes precedence over TBD check).
    """
    goals = _make_goal_dir(tmp_path)
    data  = {
        "id":                 "research_synth_001",
        "goal_type":          "research_synthesis",
        "pipeline_depth_k":   4,
        "goal_text":          "Summarize transformer neural networks.",
        "inputs":             {"topic": "transformer neural networks"},
        "ground_truth":       "",           # empty — placeholder convention
        "rubric": [
            {
                "check_id":    "c1",
                "check_type":  "substring",
                "check_params": {"field": "answer", "must_contain": ["transformer"]},
            }
        ],
        "corruption_pattern": None,
        "_placeholder":       True,
    }
    _write_instance(goals, "research_synthesis", data)
    spec = load_goal("research_synth_001", goals_dir=goals)

    assert spec.is_placeholder() is True
    assert spec.is_tbd() is False, (
        "Placeholder instances must NOT be flagged as TBD even with empty ground_truth"
    )


def test_placeholder_count_in_validation_report(tmp_path: Path) -> None:
    """
    A suite with one placeholder and one real TBD must report them separately.
    """
    goals = _make_goal_dir(tmp_path)

    # Placeholder instance
    ph_data = {
        "id": "research_synth_001", "goal_type": "research_synthesis",
        "pipeline_depth_k": 3,
        "goal_text": "Summarize topic_1.", "inputs": {"topic": "topic_1"},
        "ground_truth": "", "rubric": [
            {"check_id": "c1", "check_type": "substring",
             "check_params": {"field": "answer", "must_contain": ["topic"]}}
        ],
        "corruption_pattern": None, "_placeholder": True,
    }
    _write_instance(goals, "research_synthesis", ph_data)

    # Genuine TBD instance (no _placeholder key)
    tbd_data = {
        "id": "finance_001", "goal_type": "financial_analysis",
        "pipeline_depth_k": 0,
        "goal_text": "", "inputs": {}, "ground_truth": "",
        "rubric": [], "corruption_pattern": None,
    }
    _write_instance(goals, "financial_analysis", tbd_data)

    report = validate_goal_suite(goals_dir=goals)

    assert report.placeholder_count == 1, f"Expected 1 placeholder, got {report.placeholder_count}"
    assert report.tbd_count         == 1, f"Expected 1 TBD, got {report.tbd_count}"
    assert report.total             == 2


def test_validate_warns_on_placeholders_by_default() -> None:
    """
    After full goal authoring, the goals/ suite has 0 placeholders.
    - Default to_text() must NOT warn about placeholders.
    - to_text(allow_placeholders=True) also produces no placeholder warning.
    - No hard violations.
    """
    report = validate_goal_suite(goals_dir=_GOALS_DIR)

    assert report.placeholder_count == 0
    assert report.hard_violation_count == 0

    default_text  = report.to_text(allow_placeholders=False)
    suppress_text = report.to_text(allow_placeholders=True)

    assert "[WARN] Placeholder instances:" not in default_text, (
        "No placeholders remain; default output must not warn"
    )
    assert "[WARN] Placeholder instances:" not in suppress_text, (
        "No placeholders remain; --allow-placeholders output must not warn"
    )
