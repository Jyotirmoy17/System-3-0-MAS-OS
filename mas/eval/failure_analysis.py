"""Post-hoc failure mode analysis for the System-3/0 evaluation."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mas.eval.cascade_detector import detect_cascade


#: Ordered list of all failure categories (priority high → low).
CATEGORIES: list[str] = [
    "agent_crash",
    "budget_exhausted",
    "infinite_loop",
    "schema_violation_uncaught",
    "cascade_failure",
    "rubric_failed_no_cascade",
    "schema_violation_caught",     # not a failure; completeness only
]

#: Human-readable labels for RESULTS.md tables.
CATEGORY_LABELS: dict[str, str] = {
    "agent_crash":               "Agent crash (unhandled exception)",
    "budget_exhausted":          "Budget exhausted (token limit)",
    "infinite_loop":             "Infinite loop / max-steps hit",
    "schema_violation_uncaught": "Schema violation — uncaught (cascade)",
    "cascade_failure":           "Cascade failure (structural, S0-enforced run)",
    "rubric_failed_no_cascade":  "Rubric failure — no cascade (semantic)",
    "schema_violation_caught":   "Schema violation — caught by S0 (not a failure)",
}

_ABLATION_CONDITIONS = frozenset({"BASELINE_LANGGRAPH"})

#: Keywords in crash_reason that indicate budget exhaustion.
_BUDGET_KEYWORDS = ("budget", "token limit", "max_token", "context length")

#: Keywords in crash_reason that indicate an infinite-loop / step-limit hit.
_LOOP_KEYWORDS   = ("max_step", "max step", "infinite", "loop", "step limit",
                    "recursion", "did not terminate")

_MAX_EXAMPLES = 5


def _load_runs(runs_path: str) -> list[dict[str, Any]]:
    """Load all non-empty JSON lines from *runs_path*."""
    path = Path(runs_path)
    if not path.exists():
        raise FileNotFoundError(f"runs.jsonl not found: {runs_path}")
    runs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping malformed JSON at line %d: %s", line_no, exc
                )
    return runs


def _matches_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    """Return True if *text* (case-insensitive) contains any of *keywords*."""
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _assign_category(run: dict[str, Any]) -> str | None:
    """Assign one failure category to *run* and return it.

Returns None for runs that are fully successful with no schema violations
(i.e. they belong in no category at all).

Priority order: crash types → structural cascades → semantic failure
→ caught violations (completeness only)."""
    success       = int(run.get("success", 0))
    crash_reason  = run.get("crash_reason") or ""
    gt_match      = run.get("ground_truth_match")
    condition     = run.get("condition", "")
    s0            = run.get("s0", {})
    s0_catches    = int(s0.get("catches", 0))

    # ── 1. Crash-derived categories ────────────────────────────────────────
    if crash_reason and success == 0:
        if _matches_keywords(crash_reason, _BUDGET_KEYWORDS):
            return "budget_exhausted"
        if _matches_keywords(crash_reason, _LOOP_KEYWORDS):
            return "infinite_loop"
        return "agent_crash"

    # ── 2 & 3. Cascade categories ──────────────────────────────────────────
    if detect_cascade(run):
        if condition in _ABLATION_CONDITIONS:
            return "schema_violation_uncaught"
        return "cascade_failure"

    # ── 4. Semantic failure (rubric failed, no structural cause) ───────────
    if gt_match is False:
        return "rubric_failed_no_cascade"

    # ── 5. S0 caught a violation but run still succeeded (completeness) ────
    if success == 1 and s0_catches > 0:
        return "schema_violation_caught"

    # Successful run with no schema events — exclude.
    return None


def _empty_bucket() -> dict[str, Any]:
    """Return a fresh, empty category bucket."""
    return {
        "count":        0,
        "by_condition": defaultdict(int),
        "by_goal_type": defaultdict(int),
        "examples":     [],
    }


def _freeze_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    """Convert defaultdicts to plain dicts for serialisation."""
    return {
        "count":        bucket["count"],
        "by_condition": dict(bucket["by_condition"]),
        "by_goal_type": dict(bucket["by_goal_type"]),
        "examples":     bucket["examples"],
    }


def categorize_failures(runs_path: str) -> dict[str, Any]:
    """Categorise all failure-mode runs in *runs_path* into seven categories.

Categories are mutually exclusive.  Successful runs with no schema
catches are excluded entirely.  ``schema_violation_caught`` is included
even though it is not a failure — it is logged for completeness."""
    runs = _load_runs(runs_path)
    buckets: dict[str, dict[str, Any]] = {cat: _empty_bucket() for cat in CATEGORIES}

    for run in runs:
        category = _assign_category(run)
        if category is None:
            continue

        b = buckets[category]
        b["count"] += 1
        b["by_condition"][run.get("condition", "unknown")] += 1
        b["by_goal_type"][run.get("goal_type",  "unknown")] += 1
        if len(b["examples"]) < _MAX_EXAMPLES:
            b["examples"].append(run.get("run_id", "unknown"))

    return {
        cat: _freeze_bucket(buckets[cat])
        for cat in CATEGORIES
        if buckets[cat]["count"] > 0
    }


def failure_section_markdown(runs_path: str) -> str:
    """Generate a markdown section describing all failure modes in *runs_path*.

Suitable for direct appending to RESULTS.md."""
    cats = categorize_failures(runs_path)
    runs = _load_runs(runs_path)
    total = len(runs)

    # Total failed runs (exclude schema_violation_caught from failure count).
    failure_cats = [c for c in CATEGORIES if c != "schema_violation_caught"]
    n_failures = sum(cats.get(c, {}).get("count", 0) for c in failure_cats)
    n_caught   = cats.get("schema_violation_caught", {}).get("count", 0)
    n_success  = total - n_failures - n_caught

    lines: list[str] = [
        "### Failure Mode Analysis\n",
        f"Total runs: {total}  ",
        f"Successful (no failure): {n_success}  ",
        f"Schema violations caught by S0: {n_caught}  ",
        f"Failures: {n_failures}\n",
    ]

    if not cats:
        lines.append("> No failures or schema catches found in the run log.\n")
        return "\n".join(lines)

    # ── Summary table ──────────────────────────────────────────────────────
    lines.append("#### Summary\n")
    lines.append("| Category | Count | % of runs |")
    lines.append("| --- | --- | --- |")

    for cat in CATEGORIES:
        if cat not in cats:
            continue
        d    = cats[cat]
        n    = d["count"]
        pct  = 100.0 * n / total if total else 0.0
        note = " *(not a failure)*" if cat == "schema_violation_caught" else ""
        label = CATEGORY_LABELS.get(cat, cat)
        lines.append(f"| {label}{note} | {n} | {pct:.1f}% |")

    lines.append("")

    # ── Per-category detail tables ─────────────────────────────────────────
    lines.append("#### Breakdown by Category\n")

    for cat in CATEGORIES:
        if cat not in cats:
            continue
        d     = cats[cat]
        label = CATEGORY_LABELS.get(cat, cat)
        lines.append(f"**{label}** — {d['count']} run(s)\n")

        # By condition
        if d["by_condition"]:
            lines.append("| Condition | Count |")
            lines.append("| --- | --- |")
            for cond, cnt in sorted(d["by_condition"].items()):
                lines.append(f"| {cond} | {cnt} |")
            lines.append("")

        # By goal type
        if d["by_goal_type"]:
            lines.append("| Goal type | Count |")
            lines.append("| --- | --- |")
            for gt, cnt in sorted(d["by_goal_type"].items()):
                lines.append(f"| {gt} | {cnt} |")
            lines.append("")

        # Example run IDs
        if d["examples"]:
            ids_str = ", ".join(f"`{rid}`" for rid in d["examples"])
            lines.append(f"Example run IDs (up to {_MAX_EXAMPLES}): {ids_str}\n")

    return "\n".join(lines)
