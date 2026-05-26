"""Post-run utilities for auditing, cleaning, filtering, and summarising."""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft7Validator

log = logging.getLogger(__name__)

RunLog = dict[str, Any]
Triple = tuple[str, str, int]   # (goal_id, condition, seed)


_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "run_id", "goal_id", "goal_type",
    "pipeline_depth_k", "condition", "routing_skipped", "seed",
    "model_slm", "model_llm", "started_at", "completed_at",
    "latency_s", "success", "cascade_failure",
    "tokens", "s0", "escalations", "per_step",
    "final_output", "ground_truth", "rubric_scores",
    "ground_truth_match", "voc_trace", "crash_reason",
    "grading_api_calls", "grading_cost_usd",
})

_RUNLOG_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": sorted(_REQUIRED_FIELDS),
    "additionalProperties": True,
    "properties": {
        "run_id":             {"type": "string", "minLength": 1},
        "goal_id":            {"type": "string", "minLength": 1},
        "goal_type":          {"type": "string", "minLength": 1},
        "pipeline_depth_k": {"type": "integer", "minimum": 0},
        "condition": {
            "type": "string",
            "enum": [
                "FULL_FRAMEWORK",
                "BASELINE_LANGGRAPH",
            ],
        },
        "routing_skipped": {"type": "boolean"},
        "seed":            {"type": "integer"},
        "model_slm":       {"type": "string"},
        "model_llm":       {"type": "string"},
        "started_at":      {"type": "string"},
        "completed_at":    {"type": "string"},
        "latency_s":       {"type": "number", "minimum": 0},
        "success":         {"type": "integer", "enum": [0, 1]},
        "cascade_failure": {"type": "integer", "enum": [0, 1]},
        "tokens": {
            "type": "object",
            "required": ["slm", "llm", "total"],
            "properties": {
                "slm":   {"type": "integer", "minimum": 0},
                "llm":   {"type": "integer", "minimum": 0},
                "total": {"type": "integer", "minimum": 0},
            },
        },
        "s0": {
            "type": "object",
            "required": ["catches", "repairs", "would_have_failed"],
        },
        "escalations": {
            "type": "object",
            "required": ["P1", "P2", "P3", "P4"],
        },
        "per_step":           {"type": "array"},
        "final_output":       {"type": "string"},
        "ground_truth":       {"type": "string"},
        "rubric_scores":      {"type": "object"},
        "ground_truth_match": {"type": "boolean"},
        "voc_trace":          {"type": "array"},
        "crash_reason": {
            "oneOf": [{"type": "null"}, {"type": "string"}],
        },
        "grading_api_calls": {"type": "integer", "minimum": 0},
        "grading_cost_usd":  {"type": "number",  "minimum": 0.0},
    },
}

_VALIDATOR = Draft7Validator(_RUNLOG_SCHEMA)

_VALID_STRATEGIES = frozenset({"latest", "earliest", "successful"})


def append_run_log_atomic(path: str | Path, entry: dict[str, Any]) -> None:
    """Append *entry* as one JSON line to *path* under an exclusive file lock.

Safe for concurrent callers from multiple processes or threads on Linux.
Uses fcntl.LOCK_EX (Unix only; document this if Windows support is needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, default=str) + "\n"
    with p.open("a", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@dataclass
class InvalidLine:
    """One problematic line in runs.jsonl."""
    line_no: int        # 1-based
    reason:  str        # "malformed_json" | "missing_fields" | "schema_violation"
    detail:  str        # human-readable error text
    snippet: str        # first 120 chars of the raw line


@dataclass
class DuplicateGroup:
    """A set of lines that share the same (goal_id, condition, seed) triple."""
    triple:   Triple
    count:    int
    line_nos: list[int]   # all 1-based line numbers for this triple


@dataclass
class ValidationReport:
    """Result of validate_runs_jsonl()."""
    path:              str
    total_lines:       int                    = 0
    valid_lines:       int                    = 0
    invalid_lines:     list[InvalidLine]      = field(default_factory=list)
    duplicate_groups:  list[DuplicateGroup]   = field(default_factory=list)

    @property
    def duplicate_count(self) -> int:
        """Number of extra (redundant) entries across all duplicate groups."""
        return sum(g.count - 1 for g in self.duplicate_groups)

    @property
    def ok(self) -> bool:
        """True iff the file has no invalid lines and no duplicates."""
        return not self.invalid_lines and not self.duplicate_groups

    def to_text(self) -> str:
        """Return a human-readable multi-line report."""
        lines: list[str] = [
            f"Run Log Validation — {self.path}",
            "=" * (len(f"Run Log Validation — {self.path}")),
            f"Total lines  : {self.total_lines}",
            f"Valid lines  : {self.valid_lines}",
            f"Invalid lines: {len(self.invalid_lines)}",
            f"Duplicates   : {len(self.duplicate_groups)} group(s)"
            f", {self.duplicate_count} extra entrie(s)",
        ]

        if self.invalid_lines:
            lines.append("")
            for inv in self.invalid_lines:
                lines.append(
                    f"[INVALID] Line {inv.line_no}: {inv.reason} — {inv.detail}"
                )
                if inv.snippet:
                    lines.append(f"          Snippet: {inv.snippet}")

        if self.duplicate_groups:
            lines.append("")
            for grp in self.duplicate_groups:
                gid, cond, seed = grp.triple
                lines.append(
                    f"[DUPLICATE] ({gid}, {cond}, seed={seed}): "
                    f"{grp.count} entries at lines {grp.line_nos}"
                )

        lines.append("")
        if self.ok:
            lines.append("[OK] File is clean — no errors, no duplicates.")
        else:
            n_errs = len(self.invalid_lines) + self.duplicate_count
            lines.append(f"[FAIL] {n_errs} issue(s) found.")

        return "\n".join(lines)


def validate_runs_jsonl(path: str | Path) -> ValidationReport:
    """Stream *path* line-by-line, validate JSON and schema, detect duplicates.

Does NOT modify the file."""
    p      = Path(path)
    report = ValidationReport(path=str(p))

    # triple → list of 1-based line numbers seen so far
    triple_lines: dict[Triple, list[int]] = defaultdict(list)

    with p.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            report.total_lines += 1
            line_no = report.total_lines
            stripped = raw_line.strip()
            if not stripped:
                continue    # skip blank lines without counting as invalid

            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                report.invalid_lines.append(InvalidLine(
                    line_no = line_no,
                    reason  = "malformed_json",
                    detail  = str(exc),
                    snippet = stripped[:120],
                ))
                continue

            if not isinstance(entry, dict):
                report.invalid_lines.append(InvalidLine(
                    line_no = line_no,
                    reason  = "malformed_json",
                    detail  = f"Expected JSON object, got {type(entry).__name__}",
                    snippet = stripped[:120],
                ))
                continue

            missing = _REQUIRED_FIELDS - entry.keys()
            if missing:
                report.invalid_lines.append(InvalidLine(
                    line_no = line_no,
                    reason  = "missing_fields",
                    detail  = f"Missing: {', '.join(sorted(missing))}",
                    snippet = stripped[:120],
                ))
                continue

            errors = list(_VALIDATOR.iter_errors(entry))
            if errors:
                msgs = "; ".join(e.message for e in errors[:3])
                report.invalid_lines.append(InvalidLine(
                    line_no = line_no,
                    reason  = "schema_violation",
                    detail  = msgs,
                    snippet = stripped[:120],
                ))
                continue

            report.valid_lines += 1
            try:
                triple: Triple = (
                    str(entry["goal_id"]),
                    str(entry["condition"]),
                    int(entry["seed"]),
                )
                triple_lines[triple].append(line_no)
            except (KeyError, ValueError):
                pass    # already validated above; shouldn't happen

    # Build duplicate groups (triples with more than one occurrence)
    for triple, lnos in triple_lines.items():
        if len(lnos) > 1:
            report.duplicate_groups.append(
                DuplicateGroup(triple=triple, count=len(lnos), line_nos=lnos)
            )

    return report


def deduplicate_runs(path: str | Path, strategy: str = "latest") -> int:
    """Remove duplicate (goal_id, condition, seed) entries from *path*.

Strategy options:
    "latest"     — keep the last-written entry for each triple.
    "earliest"   — keep the first-written entry for each triple.
    "successful" — prefer success=1 over success=0; ties → latest.

Uses an atomic temp-file + os.replace() swap so the file is never
in a partially-written state."""
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown deduplication strategy {strategy!r}. "
            f"Valid: {sorted(_VALID_STRATEGIES)}"
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"runs.jsonl not found: {p}")

    # Pass 1: collect (triple → winning entry) in memory.
    # Memory cost: one RunLog dict per unique triple (~300 dicts max).
    survivors: dict[Triple, RunLog] = {}
    total_valid = 0

    with p.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                # Preserve malformed lines unchanged (validate first if needed)
                _preserve_malformed(survivors, stripped)
                continue

            if not isinstance(entry, dict):
                continue

            total_valid += 1
            try:
                triple: Triple = (
                    str(entry["goal_id"]),
                    str(entry["condition"]),
                    int(entry["seed"]),
                )
            except (KeyError, ValueError):
                triple = (
                    str(entry.get("run_id", f"_unknown_{total_valid}")),
                    "__no_condition__",
                    -total_valid,
                )

            if triple not in survivors:
                survivors[triple] = entry
                continue

            # Apply strategy
            if strategy == "latest":
                survivors[triple] = entry

            elif strategy == "earliest":
                pass    # keep first; do nothing

            elif strategy == "successful":
                existing  = survivors[triple]
                new_succ  = int(entry.get("success", 0))
                old_succ  = int(existing.get("success", 0))
                if new_succ > old_succ:
                    survivors[triple] = entry
                elif new_succ == old_succ:
                    survivors[triple] = entry   # ties → latest

    kept  = len(survivors)
    total = total_valid
    removed = total - kept

    if removed == 0:
        log.info("No duplicates found — file unchanged.")
        return 0

    # Pass 2: write survivors to a temp file, then atomically replace.
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir    = p.parent,
        prefix = f".{p.name}.dedup.",
        suffix = ".tmp",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_fh:
            for entry in survivors.values():
                if isinstance(entry, dict):
                    tmp_fh.write(json.dumps(entry, default=str) + "\n")
                else:
                    tmp_fh.write(entry + "\n")    # malformed-line passthrough
        os.replace(tmp_name, p)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise

    log.info(
        "Deduplication (%s): kept %d / %d entries, removed %d.",
        strategy, kept, total, removed,
    )
    return removed


def _preserve_malformed(
    survivors: dict[Triple, Any],
    raw: str,
) -> None:
    """Store a malformed line under a synthetic key so it isn't silently dropped."""
    key: Triple = (f"__malformed_{len(survivors)}", "__malformed__", -1)
    survivors[key] = raw


def filter_runs(path: str | Path, **kwargs: Any) -> list[RunLog]:
    """Stream *path* and return entries where every kwarg matches.

Matching is exact equality on top-level scalar fields, e.g.::

    filter_runs("runs.jsonl",
                condition="FULL_FRAMEWORK",
                goal_type="adversarial_reliability",
                success=1)

Nested fields (tokens, s0, …) are not supported by this helper;
use a list comprehension on the return value for complex filters."""
    p       = Path(path)
    results: list[RunLog] = []

    if not p.exists():
        return results

    with p.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue

            if all(entry.get(k) == v for k, v in kwargs.items()):
                results.append(entry)

    return results


def runs_summary(path: str | Path) -> dict[str, Any]:
    """Stream *path* and compute aggregate statistics.

Returns a dict with keys:

    path, total_runs, schema_valid,
    by_condition  → {condition: {runs, successes, success_rate,
                                 cascade_failures, cfr,
                                 total_tokens, mean_tokens,
                                 mean_latency_s}},
    by_goal_type  → {goal_type: {runs, successes, success_rate}},
    by_pipeline_depth → {k: {runs, successes, success_rate}},"""
    p            = Path(path)
    total_runs   = 0
    schema_valid = 0

    # Accumulators — use defaultdict(lambda: {...}) pattern
    by_cond: dict[str, dict[str, int | float]] = defaultdict(lambda: {
        "runs": 0, "successes": 0, "cascade_failures": 0,
        "total_tokens": 0, "total_latency_s": 0.0,
    })
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {
        "runs": 0, "successes": 0,
    })
    by_depth: dict[str, dict[str, int]] = defaultdict(lambda: {
        "runs": 0, "successes": 0,
    })

    if p.exists():
        with p.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue

                total_runs += 1

                # Light schema check: all required fields present
                if not _REQUIRED_FIELDS.issubset(entry.keys()):
                    continue
                schema_valid += 1

                cond    = str(entry.get("condition", "unknown"))
                gtype   = str(entry.get("goal_type", "unknown"))
                depth_k = str(entry.get("pipeline_depth_k", "unknown"))
                succ    = int(entry.get("success", 0))
                casc    = int(entry.get("cascade_failure", 0))
                tokens  = int((entry.get("tokens") or {}).get("total", 0))
                latency = float(entry.get("latency_s", 0.0))

                # by_condition
                bc = by_cond[cond]
                bc["runs"]            = int(bc["runs"]) + 1
                bc["successes"]       = int(bc["successes"]) + succ
                bc["cascade_failures"] = int(bc["cascade_failures"]) + casc
                bc["total_tokens"]    = int(bc["total_tokens"]) + tokens
                bc["total_latency_s"] = float(bc["total_latency_s"]) + latency

                # by_goal_type
                bt = by_type[gtype]
                bt["runs"]      = int(bt["runs"]) + 1
                bt["successes"] = int(bt["successes"]) + succ

                # by_pipeline_depth
                bs = by_depth[depth_k]
                bs["runs"]      = int(bs["runs"]) + 1
                bs["successes"] = int(bs["successes"]) + succ

    # Compute derived rates
    def _finalise_cond(d: dict) -> dict:
        n = int(d["runs"])
        if n == 0:
            return {**d, "success_rate": 0.0, "cfr": 0.0,
                    "mean_tokens": 0.0, "mean_latency_s": 0.0}
        return {
            "runs":              n,
            "successes":         int(d["successes"]),
            "success_rate":      round(int(d["successes"]) / n, 4),
            "cascade_failures":  int(d["cascade_failures"]),
            "cfr":               round(int(d["cascade_failures"]) / n, 4),
            "total_tokens":      int(d["total_tokens"]),
            "mean_tokens":       round(int(d["total_tokens"]) / n, 1),
            "mean_latency_s":    round(float(d["total_latency_s"]) / n, 3),
        }

    def _finalise_simple(d: dict) -> dict:
        n = int(d["runs"])
        s = int(d["successes"])
        return {"runs": n, "successes": s,
                "success_rate": round(s / n, 4) if n else 0.0}

    return {
        "path":         str(p),
        "total_runs":   total_runs,
        "schema_valid": schema_valid,
        "by_condition": {c: _finalise_cond(v)  for c, v in by_cond.items()},
        "by_goal_type": {t: _finalise_simple(v) for t, v in by_type.items()},
        "by_pipeline_depth": {k: _finalise_simple(v) for k, v in by_depth.items()},
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Return a human-readable table from a runs_summary() dict."""
    lines: list[str] = [
        f"Run Log Summary — {summary['path']}",
        f"Total entries: {summary['total_runs']}  "
        f"(schema-valid: {summary['schema_valid']})",
        "",
    ]

    # Condition table
    by_cond = summary.get("by_condition", {})
    if by_cond:
        header = (
            f"{'Condition':<22} {'Runs':>5}  "
            f"{'Success%':>9}  {'CFR':>7}  {'Mean tokens':>11}  "
            f"{'Mean latency':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        _COND_ORDER = [
            "FULL_FRAMEWORK",
            "BASELINE_LANGGRAPH",
        ]
        ordered = [c for c in _COND_ORDER if c in by_cond]
        ordered += [c for c in by_cond if c not in _COND_ORDER]
        for cond in ordered:
            d = by_cond[cond]
            lines.append(
                f"{cond:<22} {d['runs']:>5}  "
                f"{100*d['success_rate']:>8.1f}%  "
                f"{100*d['cfr']:>6.1f}%  "
                f"{d['mean_tokens']:>11,.0f}  "
                f"{d['mean_latency_s']:>11.1f}s"
            )
        lines.append("")

    # Goal-type table
    by_type = summary.get("by_goal_type", {})
    if by_type:
        lines.append(f"{'Goal type':<30} {'Runs':>5}  {'Success%':>9}")
        lines.append("-" * 48)
        for gtype, d in sorted(by_type.items()):
            lines.append(
                f"{gtype:<30} {d['runs']:>5}  "
                f"{100*d['success_rate']:>8.1f}%"
            )
        lines.append("")

    # Pipeline depth table
    by_depth = summary.get("by_pipeline_depth", {})
    if by_depth:
        lines.append(f"{'pipeline_depth_k':<18} {'Runs':>5}  {'Success%':>9}")
        lines.append("-" * 36)
        for depth_k, d in sorted(by_depth.items(), key=lambda x: x[0]):
            lines.append(
                f"{depth_k:<18} {d['runs']:>5}  "
                f"{100*d['success_rate']:>8.1f}%"
            )

    return "\n".join(lines)
