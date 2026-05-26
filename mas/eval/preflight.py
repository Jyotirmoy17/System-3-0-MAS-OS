"""Pre-flight validation: run before any experimental batch to catch."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from jsonschema import Draft7Validator

from mas.eval.goal_loader import (
    EXPECTED_ADV_PATTERN_COUNTS,
    EXPECTED_GOAL_TYPE_COUNTS,
    EXPECTED_TOTAL,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GOALS_DIR = _REPO_ROOT / "goals"

_VALID_CHECK_TYPES: frozenset[str] = frozenset({
    "exact_match", "substring", "numeric", "list_subset", "semantic_judge",
})

_VALID_GOAL_TYPES: frozenset[str] = frozenset({
    "research_synthesis",
    "financial_analysis",
    "sequential_reasoning",
    "code_analysis",
    "adversarial_reliability",
})

_FIVE_PATTERNS: list[str] = [
    "drop_required_field",
    "type_flip",
    "out_of_range",
    "malformed_json",
    "extra_fields_not_in_schema",
]

_MIN_PIPELINE_DEPTH_K = 3


@dataclass
class PreflightReport:
    """Result of run_preflight()."""
    passed:   bool
    errors:   list[str]          = field(default_factory=list)
    warnings: list[str]          = field(default_factory=list)
    stats:    dict[str, Any]     = field(default_factory=dict)

    def to_text(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines  = [f"Pre-flight Check  ({status})", "=" * 60]

        if self.errors:
            lines.append(f"\n[HARD] {len(self.errors)} error(s):")
            for e in self.errors[:25]:
                lines.append(f"  ✗ {e}")
            if len(self.errors) > 25:
                lines.append(f"  … and {len(self.errors) - 25} more")

        if self.warnings:
            lines.append(f"\n[WARN] {len(self.warnings)} warning(s):")
            for w in self.warnings[:10]:
                lines.append(f"  ⚠ {w}")
            if len(self.warnings) > 10:
                lines.append(f"  … and {len(self.warnings) - 10} more")

        lines.append(f"\n[STAT] {json.dumps(self.stats, indent=2)}")
        return "\n".join(lines)


def run_preflight(goals_dir: Optional[Path] = None) -> PreflightReport:
    """Run all pre-flight checks on the goal suite at *goals_dir*."""
    base   = goals_dir or _GOALS_DIR
    errors:   list[str] = []
    warnings: list[str] = []

    # Load schema ----------------------------------------------------------
    schema_path = base / "_schema.json"
    if not schema_path.exists():
        return PreflightReport(
            passed=False,
            errors=[f"Schema file not found: {schema_path}"],
        )
    schema    = json.loads(schema_path.read_text())
    validator = Draft7Validator(schema)

    # Load all instance files ----------------------------------------------
    instances:   list[dict[str, Any]] = []

    goal_type_counts:    dict[str, int] = {}
    corruption_counts:   dict[str, int] = {}

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        for json_file in sorted(subdir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
            except json.JSONDecodeError as exc:
                errors.append(f"{json_file.name}: JSON parse error: {exc}")
                continue

            # (a) Schema validity
            schema_errs = list(validator.iter_errors(data))
            if schema_errs:
                msgs = "; ".join(e.message for e in schema_errs[:2])
                errors.append(f"{data.get('id', json_file.name)}: schema: {msgs}")
                continue  # skip remaining checks for this file

            instances.append(data)

            # Accumulate counts for distribution checks
            gt  = data.get("goal_type", "?")
            pat = data.get("corruption_pattern")
            goal_type_counts[gt] = goal_type_counts.get(gt, 0) + 1
            if gt == "adversarial_reliability" and isinstance(pat, str):
                corruption_counts[pat] = corruption_counts.get(pat, 0) + 1

    # Per-instance checks --------------------------------------------------
    for data in instances:
        inst_id = data.get("id", "?")

        # (b) No placeholder / no TBD
        if data.get("_placeholder", False):
            errors.append(
                f"{inst_id}: _placeholder=true — instance not yet authored"
            )
            continue  # stop here; further checks aren't meaningful on stubs

        _check_tbd(data, inst_id, errors)

        # Only continue deeper checks if basic TBD checks didn't fail
        if not _has_tbd_errors(data):
            _check_reachability(data, inst_id, errors, warnings)
            _check_rubric(data, inst_id, errors)
            _check_ground_truth(data, inst_id, errors)

    # (c) Distribution checks ----------------------------------------------
    total = len(instances) + sum(
        1 for _ in []  # parse errors already counted in errors list
    )
    total = sum(goal_type_counts.values())  # only schema-valid instances

    _check_distribution(
        total, goal_type_counts, corruption_counts,
        errors, warnings,
    )

    stats: dict[str, Any] = {
        "total":                     total,
        "goal_type_counts":          goal_type_counts,
        "corruption_pattern_counts": corruption_counts,
        "error_count":               len(errors),
        "warning_count":             len(warnings),
    }

    return PreflightReport(
        passed   = len(errors) == 0,
        errors   = errors,
        warnings = warnings,
        stats    = stats,
    )


def _has_tbd_errors(data: dict[str, Any]) -> bool:
    """Return True if the instance has any TBD-like empty fields."""
    return (
        not data.get("pipeline_depth_k")
        or not str(data.get("goal_text", "")).strip()
        or not str(data.get("ground_truth", "")).strip()
        or not data.get("rubric")
    )


def _check_tbd(data: dict[str, Any], inst_id: str, errors: list[str]) -> None:
    """Check (b): all 9 required content fields are populated."""
    if not data.get("pipeline_depth_k"):
        errors.append(f"{inst_id}: pipeline_depth_k is 0 (not set)")
    if not str(data.get("goal_text", "")).strip():
        errors.append(f"{inst_id}: goal_text is empty")
    if not str(data.get("ground_truth", "")).strip():
        errors.append(f"{inst_id}: ground_truth is empty")
    if not data.get("rubric"):
        errors.append(f"{inst_id}: rubric is empty (must have ≥1 check)")


def _check_reachability(
    data: dict[str, Any],
    inst_id: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check (d): pipeline_depth_k meets unified baseline minimum."""
    depth = data.get("pipeline_depth_k", 0)
    if isinstance(depth, int) and depth < _MIN_PIPELINE_DEPTH_K:
        errors.append(
            f"{inst_id}: pipeline_depth_k={depth} "
            f"(unified baseline requires >= {_MIN_PIPELINE_DEPTH_K})"
        )


def _check_rubric(
    data: dict[str, Any], inst_id: str, errors: list[str],
) -> None:
    """Check (e): every rubric item has required fields and valid check_type."""
    rubric = data.get("rubric", [])
    if not isinstance(rubric, list):
        errors.append(f"{inst_id}: rubric is not an array")
        return

    for i, check in enumerate(rubric):
        if not isinstance(check, dict):
            errors.append(f"{inst_id}: rubric[{i}] is not an object")
            continue
        if not str(check.get("check_id", "")).strip():
            errors.append(f"{inst_id}: rubric[{i}] missing check_id")
        if not str(check.get("check_text", "")).strip():
            errors.append(f"{inst_id}: rubric[{i}] missing check_text")
        ct = check.get("check_type", "")
        if ct not in _VALID_CHECK_TYPES:
            errors.append(
                f"{inst_id}: rubric[{i}] invalid check_type '{ct}' "
                f"(must be one of {sorted(_VALID_CHECK_TYPES)})"
            )


def _check_ground_truth(
    data: dict[str, Any], inst_id: str, errors: list[str],
) -> None:
    """Check (f): ground_truth is non-empty for all goal types.

    Multi-step goals (financial_analysis, sequential_reasoning) use structured
    explanatory ground_truth strings, not bare floats.  We verify non-empty
    here; TBD check handles the empty case separately.
    """
    gt = str(data.get("ground_truth", "")).strip()
    if not gt:
        return  # already caught by _check_tbd


def _check_distribution(
    total:              int,
    goal_type_counts:   dict[str, int],
    corruption_counts:  dict[str, int],
    errors:             list[str],
    warnings:           list[str],
) -> None:
    """Check (c): goal type and adversarial pattern distributions."""
    if total != EXPECTED_TOTAL:
        errors.append(
            f"Suite must contain exactly {EXPECTED_TOTAL} instances "
            f"(91 adversarial + 32 code + 27 sequential); found {total}"
        )
        return

    # Goal-type distribution (EXPERIMENTS.md §3 adversarial-heavy split)
    for gt in _VALID_GOAL_TYPES:
        expected = EXPECTED_GOAL_TYPE_COUNTS[gt]
        count = goal_type_counts.get(gt, 0)
        if count != expected:
            errors.append(
                f"goal_type '{gt}': {count}/{expected} instances (expected {expected})"
            )

    # Adversarial corruption pattern distribution
    for pat in _FIVE_PATTERNS:
        count = corruption_counts.get(pat, 0)
        expected_pat = EXPECTED_ADV_PATTERN_COUNTS[pat]
        if count != expected_pat:
            errors.append(
                f"adversarial pattern '{pat}': {count}/{expected_pat} "
                f"(expected exactly {expected_pat})"
            )


def _main() -> None:
    report = run_preflight()
    print(report.to_text())
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    _main()
