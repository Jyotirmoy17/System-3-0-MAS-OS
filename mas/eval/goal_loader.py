"""Load, validate, and report on goal instances from the goals/ directory."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jsonschema
from jsonschema import Draft7Validator

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GOALS_DIR = _REPO_ROOT / "goals"


@dataclass
class GoalSpec:
    """One goal instance, loaded from goals/<type>/<id>.json."""
    id:                  str
    goal_type:           str
    pipeline_depth_k:    int
    goal_text:           str
    inputs:              dict[str, Any]
    ground_truth:        str
    rubric:              list[dict[str, Any]]
    corruption_pattern:  Optional[str]
    placeholder:         bool = False   # True when _placeholder: true in the file

    def is_placeholder(self) -> bool:
        """Return True if this instance is a smoke-test placeholder (not real data)."""
        return self.placeholder

    def is_tbd(self) -> bool:
        """Return True if this instance still has un-authored placeholder (TBD) content.

        Placeholder instances (_placeholder: true) are NOT considered TBD — they
        have intentional content for smoke testing even though that content will be
        replaced by hand-authored material.
        """
        if self.placeholder:
            return False
        return (
            self.pipeline_depth_k == 0
            or not self.goal_text.strip()
            or not self.ground_truth.strip()
            or not self.rubric
        )


_MIN_PIPELINE_DEPTH_K = 3
_FIVE_PATTERNS    = [
    "drop_required_field",
    "type_flip",
    "out_of_range",
    "malformed_json",
    "extra_fields_not_in_schema",
]

# Canonical 2-condition A/B suite (thesis §6.2–6.3, DEV-022):
# exactly 150 goals — 91 adversarial, 32 code, 27 sequential (0 research / 0 financial).
EXPECTED_TOTAL: int = 150
EXPECTED_GOAL_TYPE_COUNTS: dict[str, int] = {
    "research_synthesis":        0,
    "financial_analysis":        0,
    "sequential_reasoning":     27,
    "code_analysis":            32,
    "adversarial_reliability":  91,
}
# 91 adversarial goals: cycle (i-1)%5 → 19 drop_required_field, 18 each for the rest.
EXPECTED_ADV_PATTERN_COUNTS: dict[str, int] = {
    "drop_required_field":          19,
    "type_flip":                    18,
    "out_of_range":                 18,
    "malformed_json":               18,
    "extra_fields_not_in_schema":   18,
}
# Back-compat alias (uniform 15×5 superseded at 75 goals).
EXPECTED_ADV_PATTERN_COUNT_EACH: int = 18


@dataclass
class ValidationReport:
    """Result of validate_goal_suite()."""
    total:                  int               = 0
    tbd_count:              int               = 0
    tbd_instances:          list[str]         = field(default_factory=list)
    placeholder_count:      int               = 0
    placeholder_instances:  list[str]         = field(default_factory=list)
    schema_errors:          list[str]         = field(default_factory=list)
    distribution_warnings:  list[str]         = field(default_factory=list)
    pattern_errors:         list[str]         = field(default_factory=list)
    goal_distribution_errors: list[str]       = field(default_factory=list)
    shallow_depth_count:    int               = 0
    shallow_depth_instances: list[str]        = field(default_factory=list)

    @property
    def hard_violation_count(self) -> int:
        """Schema violations, goal-type counts, and pattern distribution are hard."""
        return (
            len(self.schema_errors)
            + len(self.pattern_errors)
            + len(self.goal_distribution_errors)
        )

    def to_text(self, allow_placeholders: bool = False) -> str:
        """Human-readable report suitable for CLI stdout."""
        lines: list[str] = [
            f"Goal Suite Validation  ({self.total} instances)",
            "=" * 60,
        ]

        if self.schema_errors:
            lines.append(f"\n[HARD] Schema violations ({len(self.schema_errors)}):")
            for err in self.schema_errors[:10]:
                lines.append(f"  ✗ {err}")
            if len(self.schema_errors) > 10:
                lines.append(f"  … and {len(self.schema_errors) - 10} more")

        if self.pattern_errors:
            lines.append(f"\n[HARD] Adversarial pattern distribution ({len(self.pattern_errors)}):")
            for err in self.pattern_errors:
                lines.append(f"  ✗ {err}")

        if self.goal_distribution_errors:
            lines.append(
                f"\n[HARD] Goal-type distribution ({len(self.goal_distribution_errors)}):"
            )
            for err in self.goal_distribution_errors:
                lines.append(f"  ✗ {err}")

        if self.distribution_warnings:
            lines.append(f"\n[WARN] Suite distribution:")
            for w in self.distribution_warnings:
                lines.append(f"  ⚠ {w}")

        if self.shallow_depth_count > 0:
            lines.append(
                f"\n[WARN] pipeline_depth_k < {_MIN_PIPELINE_DEPTH_K}: "
                f"{self.shallow_depth_count}/{self.total}"
            )

        if self.placeholder_count > 0 and not allow_placeholders:
            lines.append(
                f"\n[WARN] Placeholder instances: {self.placeholder_count}/{self.total}"
            )
            lines.append(
                "       These are smoke-test stubs.  Hand-author before experimental runs."
            )
            lines.append(
                "       Use --allow-placeholders to suppress this warning."
            )
        elif self.placeholder_count > 0:
            lines.append(
                f"\n[INFO] Placeholder instances: {self.placeholder_count}/{self.total} (suppressed)"
            )

        if self.tbd_count > 0:
            lines.append(f"\n[INFO] TBD (empty content, not placeholder): {self.tbd_count}/{self.total}")
            lines.append(f"       {self.tbd_instances[:5]!r}{'…' if len(self.tbd_instances) > 5 else ''}")

        lines.append("")
        if self.hard_violation_count == 0 and self.tbd_count == 0 and self.placeholder_count == 0:
            lines.append("[OK]  Suite is complete and ready for experiment runs.")
        elif self.hard_violation_count == 0:
            lines.append("[OK]  No hard violations.")
        else:
            lines.append(f"[FAIL] {self.hard_violation_count} hard violation(s).")

        return "\n".join(lines)


_schema_cache: dict[Path, dict[str, Any]] = {}


def _get_schema(goals_dir: Path) -> dict[str, Any]:
    if goals_dir not in _schema_cache:
        schema_path = goals_dir / "_schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        _schema_cache[goals_dir] = json.loads(schema_path.read_text())
    return _schema_cache[goals_dir]


def _get_validator(goals_dir: Path) -> Draft7Validator:
    return Draft7Validator(_get_schema(goals_dir))


def _load_file(path: Path, goals_dir: Path) -> GoalSpec:
    """Parse, schema-validate, and return one GoalSpec. Raises on error."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error in {path.name}: {exc}") from exc

    validator = _get_validator(goals_dir)
    errors    = list(validator.iter_errors(data))
    if errors:
        msgs = "; ".join(e.message for e in errors[:3])
        raise jsonschema.ValidationError(
            f"Goal file '{path.name}' fails schema validation: {msgs}"
        )

    return GoalSpec(
        id                 = data["id"],
        goal_type          = data["goal_type"],
        pipeline_depth_k   = data["pipeline_depth_k"],
        goal_text          = data["goal_text"],
        inputs             = data["inputs"],
        ground_truth       = data["ground_truth"],
        rubric             = data["rubric"],
        corruption_pattern = data.get("corruption_pattern"),
        placeholder        = bool(data.get("_placeholder", False)),
    )


def load_goal(goal_id: str, goals_dir: Optional[Path] = None) -> GoalSpec:
    """Load and validate a single goal instance by its ID string."""
    base = goals_dir or _GOALS_DIR
    for subdir in base.iterdir():
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        candidate = subdir / f"{goal_id}.json"
        if candidate.exists():
            return _load_file(candidate, base)
    raise FileNotFoundError(f"Goal '{goal_id}' not found in {base}")


def load_all_goals(goals_dir: Optional[Path] = None) -> list[GoalSpec]:
    """Load and validate all goal instances in goals_dir."""
    base   = goals_dir or _GOALS_DIR
    goals: list[GoalSpec] = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        for json_file in sorted(subdir.glob("*.json")):
            goals.append(_load_file(json_file, base))
    return goals


def validate_goal_suite(goals_dir: Optional[Path] = None) -> ValidationReport:
    """Validate the full goal suite without raising on individual failures."""
    base   = goals_dir or _GOALS_DIR
    report = ValidationReport()

    try:
        validator = _get_validator(base)
    except FileNotFoundError as exc:
        report.schema_errors.append(str(exc))
        return report

    adversarial_patterns: dict[str, int] = {}
    goal_type_counts: dict[str, int] = {}

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        for json_file in sorted(subdir.glob("*.json")):
            report.total += 1

            # Parse
            try:
                data = json.loads(json_file.read_text())
            except json.JSONDecodeError as exc:
                report.schema_errors.append(f"{json_file.name}: {exc}")
                continue

            # Schema validation
            errors = list(validator.iter_errors(data))
            if errors:
                msgs = "; ".join(e.message for e in errors[:2])
                report.schema_errors.append(f"{json_file.name}: {msgs}")
                continue

            gt = data.get("goal_type")
            if isinstance(gt, str):
                goal_type_counts[gt] = goal_type_counts.get(gt, 0) + 1

            # Placeholder vs TBD detection (mutually exclusive)
            is_ph = bool(data.get("_placeholder", False))
            if is_ph:
                report.placeholder_count += 1
                report.placeholder_instances.append(data["id"])
            else:
                tbd = (
                    data.get("pipeline_depth_k") == 0
                    or not str(data.get("goal_text", "")).strip()
                    or not str(data.get("ground_truth", "")).strip()
                    or not data.get("rubric")
                )
                if tbd:
                    report.tbd_count += 1
                    report.tbd_instances.append(data["id"])

            depth = data.get("pipeline_depth_k", 0)
            if isinstance(depth, int) and depth < _MIN_PIPELINE_DEPTH_K:
                report.shallow_depth_count += 1
                report.shallow_depth_instances.append(data["id"])

            # Adversarial pattern counts
            if data.get("goal_type") == "adversarial_reliability":
                p = data.get("corruption_pattern")
                if isinstance(p, str):
                    adversarial_patterns[p] = adversarial_patterns.get(p, 0) + 1

    # Adversarial pattern distribution (hard violation if wrong — always checkable)
    for pattern in _FIVE_PATTERNS:
        count = adversarial_patterns.get(pattern, 0)
        expected_pat = EXPECTED_ADV_PATTERN_COUNTS[pattern]
        if count != expected_pat:
            report.pattern_errors.append(
                f"'{pattern}': {count}/{expected_pat} adversarial instances"
            )

    # Goal-type totals (hard when every file schema-validates)
    n_classified = sum(goal_type_counts.values())
    if not report.schema_errors:
        if report.total != EXPECTED_TOTAL:
            report.goal_distribution_errors.append(
                f"suite size: {report.total}/{EXPECTED_TOTAL} instances "
                f"(expected exactly {EXPECTED_TOTAL}: "
                f"91 adversarial + 32 code + 27 sequential)"
            )
        elif n_classified == EXPECTED_TOTAL:
            for gt, expected in EXPECTED_GOAL_TYPE_COUNTS.items():
                got = goal_type_counts.get(gt, 0)
                if got != expected:
                    report.goal_distribution_errors.append(
                        f"goal_type '{gt}': {got}/{expected} instances"
                    )

    return report


def _main() -> None:
    args = sys.argv[1:]
    if not args or args[0] != "validate":
        print(
            "Usage: python -m mas.eval.goal_loader validate [--allow-placeholders]",
            file=sys.stderr,
        )
        sys.exit(2)
    allow_ph = "--allow-placeholders" in args
    report   = validate_goal_suite()
    print(report.to_text(allow_placeholders=allow_ph))
    sys.exit(0 if report.hard_violation_count == 0 else 1)


if __name__ == "__main__":
    _main()
