"""Re-grade existing run logs against a real judge backend in place."""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from mas.eval.cascade_detector import detect_cascade
from mas.eval.goal_loader import GoalSpec, load_all_goals
from mas.eval.harness import _extract_from_output, score_rubric
from mas.eval.judge import JudgeProtocol

log = logging.getLogger(__name__)


@dataclass
class RegradeStats:
    """Summary counters for one regrade pass."""
    total_runs:         int = 0      # rows read
    eligible:           int = 0      # rows that matched the filters
    regraded:           int = 0      # rows whose rubric_scores were rewritten
    unchanged:          int = 0      # regraded but no field actually changed
    skipped_crashed:    int = 0      # run had crash_reason set
    skipped_no_output:  int = 0      # final_output empty / missing
    skipped_no_rubric:  int = 0      # goal has no rubric (or goal not found)
    skipped_filtered:   int = 0      # filtered out by --conditions / --goal-types
    skipped_already:    int = 0      # already graded by real backend, --force not set
    cascade_flips:      int = 0      # runs where cascade_failure changed value
    match_flips:        int = 0      # runs where ground_truth_match changed value
    judge_all_override: int = 0      # runs whose rubric was overridden by --judge-all


@dataclass
class RegradeResult:
    """Return value of regrade()."""
    stats:           RegradeStats
    backup_path:     Optional[Path] = None
    audit_log_path:  Optional[Path] = None
    runs_path:       Optional[Path] = None
    judge_backend:   str            = ""
    judge_model:     str            = ""
    elapsed_s:       float          = 0.0
    skipped_details: list[dict[str, str]] = field(default_factory=list)


def regrade(
    runs_path:       Path | str,
    judge_backend:   str,
    *,
    judge_model:     Optional[str] = None,
    ollama_url:      str           = "http://localhost:11434",
    goals_dir:       Path | str | None = None,
    filter_conditions: Optional[Iterable[str]]    = None,
    filter_goal_types: Optional[Iterable[str]]    = None,
    filter_goal_ids:   Optional[Iterable[str]]    = None,
    force:           bool          = False,
    backup:          bool          = True,
    dry_run:         bool          = False,
    audit_log_path:  Optional[Path | str] = None,
    judge_client:    Optional[JudgeProtocol] = None,
    judge_all:       bool          = False,
) -> RegradeResult:
    """Re-score every applicable run in *runs_path* with a real judge backend."""
    started_at = time.perf_counter()
    runs_path  = Path(runs_path)
    if not runs_path.exists():
        raise FileNotFoundError(f"runs.jsonl not found at {runs_path}")

    # Validate / build judge client
    if judge_client is None:
        judge_client = _build_judge(
            judge_backend = judge_backend,
            judge_model   = judge_model,
            ollama_url    = ollama_url,
        )

    # Load goals once and index by goal_id for O(1) rubric lookup.
    goals_index = _index_goals(goals_dir)

    # Normalise filters
    cond_filter = set(filter_conditions) if filter_conditions else None
    type_filter = set(filter_goal_types)  if filter_goal_types else None
    id_filter   = set(filter_goal_ids)    if filter_goal_ids   else None

    # Read all runs into memory.
    raw_lines: list[str]              = _read_lines(runs_path)
    runs:      list[dict[str, Any]]   = []
    parse_errors: list[tuple[int, str]] = []
    for idx, line in enumerate(raw_lines, start=1):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError as exc:
            parse_errors.append((idx, str(exc)))
    if parse_errors:
        log.warning(
            "regrade: %d malformed JSON line(s) skipped (first: line %d: %s)",
            len(parse_errors), parse_errors[0][0], parse_errors[0][1],
        )

    stats        = RegradeStats(total_runs=len(runs))
    audit_events: list[dict[str, Any]] = []
    skipped:     list[dict[str, str]] = []

    for run in runs:
        decision = _evaluate_run(
            run         = run,
            cond_filter = cond_filter,
            type_filter = type_filter,
            id_filter   = id_filter,
            goals_index = goals_index,
            force       = force,
        )
        if decision is _SKIP_FILTERED:
            stats.skipped_filtered += 1
            continue
        if decision is _SKIP_CRASHED:
            stats.skipped_crashed += 1
            skipped.append({"run_id": run.get("run_id", ""), "reason": "crashed"})
            continue
        if decision is _SKIP_NO_OUTPUT:
            stats.skipped_no_output += 1
            skipped.append({"run_id": run.get("run_id", ""), "reason": "no_output"})
            continue
        if decision is _SKIP_NO_RUBRIC:
            stats.skipped_no_rubric += 1
            skipped.append({"run_id": run.get("run_id", ""), "reason": "no_rubric"})
            continue
        if decision is _SKIP_ALREADY_GRADED:
            stats.skipped_already += 1
            skipped.append({"run_id": run.get("run_id", ""), "reason": "already_graded"})
            continue

        # decision is the GoalSpec for this run; proceed to regrade.
        goal: GoalSpec = decision  # type: ignore[assignment]
        stats.eligible += 1

        before = _snapshot(run)
        if judge_all:
            _apply_regrade_judge_all(run, goal, judge_client)
            stats.judge_all_override += 1
        else:
            _apply_regrade(run, goal, judge_client)
        after  = _snapshot(run)

        changed_match   = before["ground_truth_match"] != after["ground_truth_match"]
        changed_cascade = before["cascade_failure"]    != after["cascade_failure"]
        changed_scores  = before["rubric_scores"]      != after["rubric_scores"]
        any_change      = changed_match or changed_cascade or changed_scores

        stats.regraded  += 1
        if any_change:
            if changed_match:
                stats.match_flips += 1
            if changed_cascade:
                stats.cascade_flips += 1
        else:
            stats.unchanged += 1

        audit_events.append({
            "regrade_id":     uuid.uuid4().hex,
            "timestamp":      _iso_now(),
            "run_id":         run.get("run_id", ""),
            "goal_id":        run.get("goal_id", ""),
            "goal_type":      run.get("goal_type", ""),
            "condition":      run.get("condition", ""),
            "seed":           run.get("seed"),
            "judge_backend":  judge_backend,
            "judge_model":    _judge_model_string(judge_client, judge_model),
            "judge_all":      judge_all,
            "before":         before,
            "after":          after,
            "match_flipped":  changed_match,
            "cascade_flipped":changed_cascade,
        })

    # Persist results unless dry-run.
    backup_path:    Optional[Path] = None
    audit_log_path_resolved: Optional[Path] = None
    if not dry_run:
        if backup:
            backup_path = _write_backup(runs_path, raw_lines)
        _write_runs(runs_path, runs)
        audit_log_path_resolved = _resolve_audit_path(runs_path, audit_log_path)
        _append_audit(audit_log_path_resolved, audit_events)

    elapsed = time.perf_counter() - started_at
    log.info(
        "regrade complete: %d/%d eligible regraded "
        "(%d match flips, %d cascade flips) in %.2fs",
        stats.regraded, stats.total_runs,
        stats.match_flips, stats.cascade_flips, elapsed,
    )

    return RegradeResult(
        stats           = stats,
        backup_path     = backup_path,
        audit_log_path  = audit_log_path_resolved,
        runs_path       = runs_path,
        judge_backend   = judge_backend,
        judge_model     = _judge_model_string(judge_client, judge_model),
        elapsed_s       = elapsed,
        skipped_details = skipped,
    )


# Sentinels for _evaluate_run — avoids tuple/Optional gymnastics.
_SKIP_FILTERED       = object()
_SKIP_CRASHED        = object()
_SKIP_NO_OUTPUT      = object()
_SKIP_NO_RUBRIC      = object()
_SKIP_ALREADY_GRADED = object()


def _evaluate_run(
    *,
    run:         dict[str, Any],
    cond_filter: Optional[set[str]],
    type_filter: Optional[set[str]],
    id_filter:   Optional[set[str]],
    goals_index: dict[str, GoalSpec],
    force:       bool,
) -> GoalSpec | object:
    """Return GoalSpec to regrade against, or one of the _SKIP_* sentinels."""
    goal_id   = run.get("goal_id", "")
    goal_type = run.get("goal_type", "")
    condition = run.get("condition", "")

    if cond_filter and condition not in cond_filter:
        return _SKIP_FILTERED
    if type_filter and goal_type not in type_filter:
        return _SKIP_FILTERED
    if id_filter and goal_id not in id_filter:
        return _SKIP_FILTERED

    if run.get("crash_reason"):
        return _SKIP_CRASHED
    if not run.get("final_output"):
        return _SKIP_NO_OUTPUT

    goal = goals_index.get(goal_id)
    if goal is None or not goal.rubric:
        return _SKIP_NO_RUBRIC

    if not force and int(run.get("grading_api_calls", 0)) > 0:
        return _SKIP_ALREADY_GRADED

    return goal


def _apply_regrade(
    run:          dict[str, Any],
    goal:         GoalSpec,
    judge_client: JudgeProtocol,
) -> None:
    """Score the rubric, update the run dict in place, re-run cascade detection."""
    calls_before = int(getattr(judge_client, "total_api_calls", 0))
    cost_before  = float(getattr(judge_client, "total_cost_usd",  0.0))

    new_scores = score_rubric(
        goal.rubric,
        run.get("final_output", ""),
        judge_client,
    )

    run["rubric_scores"]      = new_scores
    run["ground_truth_match"] = bool(new_scores) and all(new_scores.values())
    run["grading_api_calls"]  = (
        int(getattr(judge_client, "total_api_calls", 0)) - calls_before
    )
    run["grading_cost_usd"]   = (
        float(getattr(judge_client, "total_cost_usd", 0.0)) - cost_before
    )

    run["cascade_failure"] = 1 if detect_cascade(run) else 0


# DEV-020: Synthetic criterion template used by --judge-all to build a
# rubric-independent verdict from the goal's natural-language fields.
# The judge must answer YES iff the agent's answer substantively addresses
# the goal as expressed in goal_text and ground_truth.  Kept short so the
# judge's input window is small and deterministic.
_JUDGE_ALL_CRITERION_TEMPLATE = (
    "The agent was asked the following goal:\n"
    "{goal_text}\n\n"
    "The expected answer (or key facts) is:\n"
    "{ground_truth}\n\n"
    "The agent's answer correctly and substantively satisfies the goal "
    "if and only if it engages with the actual question and produces "
    "content that matches the expected answer in meaning (not "
    "necessarily verbatim).  Stub answers, refusals, or single-step "
    "fragments that do not address the multi-step task fail this check."
)


def _build_judge_all_criterion(goal: GoalSpec) -> str:
    """Build the synthetic --judge-all criterion from a goal's natural fields."""
    return _JUDGE_ALL_CRITERION_TEMPLATE.format(
        goal_text    = (goal.goal_text    or "").strip(),
        ground_truth = (goal.ground_truth or "").strip(),
    )


def _apply_regrade_judge_all(
    run:          dict[str, Any],
    goal:         GoalSpec,
    judge_client: JudgeProtocol,
) -> None:
    """
    DEV-020 override: replace the authored rubric with a single judge verdict
    against a synthetic criterion built from goal_text + ground_truth.

    Behaviour:
      - The original rubric_scores are preserved under
        ``rubric_scores_pre_judge_all`` so reviewers can audit what was
        overridden.
      - rubric_scores becomes ``{"_judge_overall": <verdict>}``.
      - ground_truth_match becomes the judge verdict (bool).
      - grading_api_calls / grading_cost_usd are deltas for THIS run only.
      - cascade_failure is recomputed against the new
        ground_truth_match via detect_cascade.

    All judge_all-graded runs are EXPLORATORY ONLY per DEV-020.
    """
    calls_before = int(getattr(judge_client, "total_api_calls", 0))
    cost_before  = float(getattr(judge_client, "total_cost_usd",  0.0))

    answer    = _extract_from_output(run.get("final_output", "") or "", "answer")
    criterion = _build_judge_all_criterion(goal)
    try:
        verdict = bool(judge_client.judge(answer, criterion))
    except Exception:
        verdict = False  # fail-closed, mirroring OllamaJudge default

    # Preserve the pre-override rubric_scores once.  If a previous
    # judge_all pass already wrote this field, do NOT overwrite it
    # (we want to keep the *original*, pre-any-override scores).
    if "rubric_scores_pre_judge_all" not in run:
        run["rubric_scores_pre_judge_all"] = dict(run.get("rubric_scores") or {})

    run["rubric_scores"]      = {"_judge_overall": verdict}
    run["ground_truth_match"] = verdict
    run["grading_api_calls"]  = (
        int(getattr(judge_client, "total_api_calls", 0)) - calls_before
    )
    run["grading_cost_usd"]   = (
        float(getattr(judge_client, "total_cost_usd", 0.0)) - cost_before
    )

    run["cascade_failure"] = 1 if detect_cascade(run) else 0


def _snapshot(run: dict[str, Any]) -> dict[str, Any]:
    """Capture the four fields regrade may touch, for audit-log diffing."""
    return {
        "rubric_scores":      dict(run.get("rubric_scores") or {}),
        "ground_truth_match": run.get("ground_truth_match"),
        "cascade_failure":    run.get("cascade_failure"),
        "grading_api_calls":  run.get("grading_api_calls", 0),
    }


def _index_goals(goals_dir: Path | str | None) -> dict[str, GoalSpec]:
    """Load every goal once and index by id."""
    goals = load_all_goals(goals_dir=Path(goals_dir) if goals_dir else None)
    return {g.id: g for g in goals}


def _build_judge(
    *,
    judge_backend: str,
    judge_model:   Optional[str],
    ollama_url:    str,
) -> JudgeProtocol:
    """Instantiate a JudgeProtocol client based on the backend name."""
    if judge_backend == "ollama":
        from mas.eval.judge import OllamaJudge  # noqa: PLC0415
        return OllamaJudge(
            model      = judge_model or OllamaJudge.DEFAULT_MODEL,
            ollama_url = ollama_url,
        )
    if judge_backend == "anthropic":
        from mas.eval.judge import AnthropicJudge  # noqa: PLC0415
        return AnthropicJudge(
            model = judge_model or AnthropicJudge.DEFAULT_MODEL,
        )
    raise ValueError(
        f"Unsupported judge_backend {judge_backend!r}. "
        f"Must be one of: 'ollama', 'anthropic'."
    )


def _judge_model_string(
    judge_client: JudgeProtocol,
    fallback:     Optional[str],
) -> str:
    """Best-effort extraction of the judge's model identifier for the audit log."""
    if fallback:
        return fallback
    return str(getattr(judge_client, "_model", "") or "unknown")


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return fh.readlines()


def _write_backup(runs_path: Path, raw_lines: list[str]) -> Path:
    """Write the original lines to <runs_path>.bak.<UTC-timestamp>."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = runs_path.with_name(f"{runs_path.name}.bak.{ts}")
    with backup_path.open("w", encoding="utf-8") as fh:
        fh.writelines(raw_lines)
    log.info("regrade: backup written to %s", backup_path)
    return backup_path


def _write_runs(runs_path: Path, runs: list[dict[str, Any]]) -> None:
    """Write *runs* atomically: tmp file then os.replace onto runs_path."""
    tmp_path = runs_path.with_name(f"{runs_path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            for run in runs:
                fh.write(json.dumps(run) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, runs_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _resolve_audit_path(
    runs_path:      Path,
    explicit_path:  Optional[Path | str],
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    return runs_path.with_name(f"{runs_path.name}.regrade-audit.jsonl")


def _append_audit(path: Path, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    with path.open("a", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    log.info("regrade: %d audit event(s) appended to %s", len(events), path)


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_summary(result: RegradeResult) -> str:
    """Human-readable regrade summary suitable for stdout."""
    s = result.stats
    lines = [
        "Regrade summary",
        f"  runs file:        {result.runs_path}",
        f"  judge backend:    {result.judge_backend}",
        f"  judge model:      {result.judge_model}",
        f"  elapsed:          {result.elapsed_s:.2f}s",
        "",
        f"  total runs read:  {s.total_runs}",
        f"  eligible:         {s.eligible}",
        f"  regraded:         {s.regraded}",
        f"    unchanged:      {s.unchanged}",
        f"    match flipped:  {s.match_flips}",
        f"    cascade flipped:{s.cascade_flips}",
        f"    --judge-all:    {s.judge_all_override}",
        "",
        f"  skipped (filter): {s.skipped_filtered}",
        f"  skipped (crash):  {s.skipped_crashed}",
        f"  skipped (no out): {s.skipped_no_output}",
        f"  skipped (rubric): {s.skipped_no_rubric}",
        f"  skipped (graded): {s.skipped_already}",
    ]
    if result.backup_path:
        lines.append("")
        lines.append(f"  backup:           {result.backup_path}")
    if result.audit_log_path:
        lines.append(f"  audit log:        {result.audit_log_path}")
    if s.judge_all_override > 0:
        lines.append("")
        lines.append(
            "  WARNING: --judge-all override applied.  Per DEV-020, JPI / "
            "Accuracy values derived from this run log are EXPLORATORY ONLY."
        )
    return "\n".join(lines)


def result_to_dict(result: RegradeResult) -> dict[str, Any]:
    """Serialise a RegradeResult to a JSON-friendly dict (for tests / CLI)."""
    return {
        "stats":           asdict(result.stats),
        "backup_path":     str(result.backup_path) if result.backup_path else None,
        "audit_log_path":  str(result.audit_log_path) if result.audit_log_path else None,
        "runs_path":       str(result.runs_path) if result.runs_path else None,
        "judge_backend":   result.judge_backend,
        "judge_model":     result.judge_model,
        "elapsed_s":       result.elapsed_s,
        "skipped_details": result.skipped_details,
    }
