"""Interactive judge-calibration harness (DEV-019)."""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import textwrap
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from mas.eval.goal_loader import GoalSpec, load_all_goals
from mas.eval.harness import _extract_from_output  # noqa: PLC0415  (intentional)

log = logging.getLogger(__name__)


@dataclass
class CalibrationItem:
    """One (run, semantic_check) pair to grade."""
    item_idx:        int
    run_id:          str
    goal_id:         str
    goal_type:       str
    condition:       str
    seed:            int
    check_id:        str
    check_text:      str
    criteria:        str
    field:           str
    extracted_text:  str
    goal_text:       str
    ground_truth:    str
    # Verdict fields populated during grading.
    human_verdict:   Optional[bool] = None
    ai_verdict:      Optional[bool] = None
    ai_error:        Optional[str]  = None


@dataclass
class CalibrationSummary:
    n_total:                int
    n_graded:               int
    n_skipped:              int
    n_ai_errors:            int
    raw_agreement:          float
    cohens_kappa:           float
    kappa_interpretation:   str
    confusion_matrix:       dict[str, int]
    passes_threshold:       bool
    threshold:              float
    by_goal_type:           dict[str, dict[str, float | int]]
    by_condition:           dict[str, dict[str, float | int]]


@dataclass
class CalibrationResult:
    metadata: dict[str, Any]
    items:    list[CalibrationItem]
    summary:  CalibrationSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "items":    [asdict(it) for it in self.items],
            "summary":  asdict(self.summary),
        }


def build_candidate_items(
    runs:        Iterable[dict[str, Any]],
    goals_index: dict[str, GoalSpec],
) -> list[CalibrationItem]:
    """
    Expand runs.jsonl into one CalibrationItem per (run, semantic_judge_check).

    Skips runs that:
      - have no final_output, or
      - have crash_reason populated, or
      - point to an unknown goal_id, or
      - have a goal whose rubric contains zero semantic_judge checks.
    """
    items: list[CalibrationItem] = []
    for run in runs:
        if run.get("crash_reason"):
            continue
        final_output = run.get("final_output") or ""
        if not final_output:
            continue
        goal = goals_index.get(run.get("goal_id", ""))
        if goal is None:
            continue
        for check in goal.rubric:
            if check.get("check_type") != "semantic_judge":
                continue
            params      = check.get("check_params", {}) or {}
            field_name  = str(params.get("field", "answer"))
            criteria    = str(params.get("criteria", ""))
            extracted   = _extract_from_output(final_output, field_name)
            items.append(CalibrationItem(
                item_idx       = 0,   # filled after sampling
                run_id         = str(run.get("run_id", "")),
                goal_id        = goal.id,
                goal_type      = goal.goal_type,
                condition      = str(run.get("condition", "")),
                seed           = int(run.get("seed", 0)),
                check_id       = str(check.get("check_id", "")),
                check_text     = str(check.get("check_text", "")),
                criteria       = criteria,
                field          = field_name,
                extracted_text = extracted,
                goal_text      = goal.goal_text,
                ground_truth   = goal.ground_truth,
            ))
    return items


def sample_items(
    candidates: list[CalibrationItem],
    n:          int,
    seed:       int,
    *,
    prioritise_goal_type: Optional[str] = None,
) -> list[CalibrationItem]:
    """Sample n calibration items with round-robin diversity across goal_id."""
    if not candidates:
        return []

    rng = random.Random(seed)

    def _round_robin_by_goal(pool: list[CalibrationItem]) -> list[CalibrationItem]:
        groups: dict[str, list[CalibrationItem]] = defaultdict(list)
        for it in pool:
            groups[it.goal_id].append(it)
        # Shuffle each group, and shuffle the order in which goals contribute.
        for g in groups.values():
            rng.shuffle(g)
        order = list(groups.keys())
        rng.shuffle(order)
        out: list[CalibrationItem] = []
        # Round-robin pop one item per goal until pool is drained.
        while any(groups[g] for g in order):
            for g in order:
                if groups[g]:
                    out.append(groups[g].pop())
        return out

    if prioritise_goal_type:
        primary   = [c for c in candidates if c.goal_type == prioritise_goal_type]
        secondary = [c for c in candidates if c.goal_type != prioritise_goal_type]
        ordered   = _round_robin_by_goal(primary) + _round_robin_by_goal(secondary)
    else:
        ordered = _round_robin_by_goal(candidates)

    chosen = ordered[: n]
    for i, it in enumerate(chosen, start=1):
        it.item_idx = i
    return chosen


def cohens_kappa(graded: list[CalibrationItem]) -> float:
    """
    Compute Cohen's κ on the human-vs-AI 2×2 confusion matrix.

    Edge cases:
      - Empty input ⇒ NaN.
      - Total agreement on a single class ⇒ NaN (κ undefined; perfect
        agreement on one class is not informative without variance).
      - Otherwise κ = (p_o − p_e) / (1 − p_e).
    """
    pairs = [
        (it.human_verdict, it.ai_verdict)
        for it in graded
        if it.human_verdict is not None and it.ai_verdict is not None
    ]
    n = len(pairs)
    if n == 0:
        return float("nan")

    a = sum(1 for h, a_ in pairs if h is True  and a_ is True)
    b = sum(1 for h, a_ in pairs if h is True  and a_ is False)
    c = sum(1 for h, a_ in pairs if h is False and a_ is True)
    d = sum(1 for h, a_ in pairs if h is False and a_ is False)

    p_o = (a + d) / n
    p_e = (
        ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    )
    if math.isclose(p_e, 1.0):
        # Both raters agree on the same class for every item;
        # κ is mathematically undefined.
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def interpret_kappa(kappa: float) -> str:
    """
    Landis & Koch (1977) qualitative interpretation of κ.
    Returns 'undefined' if κ is NaN.
    """
    if math.isnan(kappa):
        return "undefined"
    if kappa < 0.0:
        return "poor (worse than chance)"
    if kappa < 0.2:
        return "slight"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "almost perfect"


def confusion_matrix(graded: list[CalibrationItem]) -> dict[str, int]:
    a = sum(1 for it in graded
            if it.human_verdict is True  and it.ai_verdict is True)
    b = sum(1 for it in graded
            if it.human_verdict is True  and it.ai_verdict is False)
    c = sum(1 for it in graded
            if it.human_verdict is False and it.ai_verdict is True)
    d = sum(1 for it in graded
            if it.human_verdict is False and it.ai_verdict is False)
    return {
        "human_pass_ai_pass": a,
        "human_pass_ai_fail": b,
        "human_fail_ai_pass": c,
        "human_fail_ai_fail": d,
    }


def _stratum_breakdown(
    graded: list[CalibrationItem],
    key:    str,   # "goal_type" or "condition"
) -> dict[str, dict[str, float | int]]:
    by_key: dict[str, list[CalibrationItem]] = defaultdict(list)
    for it in graded:
        by_key[getattr(it, key)].append(it)
    out: dict[str, dict[str, float | int]] = {}
    for k, sub in by_key.items():
        n = len(sub)
        agree = sum(
            1 for it in sub
            if it.human_verdict is not None
            and it.ai_verdict is not None
            and it.human_verdict == it.ai_verdict
        )
        kappa = cohens_kappa(sub)
        out[k] = {
            "n":             n,
            "agreement":     (agree / n) if n else 0.0,
            "cohens_kappa":  kappa,
        }
    return out


def summarise(
    items:     list[CalibrationItem],
    threshold: float,
) -> CalibrationSummary:
    """Compute aggregate statistics from a (possibly partial) set of items."""
    skipped     = sum(1 for it in items if it.human_verdict is None)
    ai_errors   = sum(1 for it in items if it.ai_error is not None)
    graded      = [it for it in items if it.human_verdict is not None
                   and it.ai_verdict is not None]
    n_graded    = len(graded)

    if n_graded == 0:
        agreement = 0.0
        kappa     = float("nan")
    else:
        agreement = sum(1 for it in graded
                        if it.human_verdict == it.ai_verdict) / n_graded
        kappa     = cohens_kappa(graded)

    return CalibrationSummary(
        n_total              = len(items),
        n_graded             = n_graded,
        n_skipped            = skipped,
        n_ai_errors          = ai_errors,
        raw_agreement        = agreement,
        cohens_kappa         = kappa,
        kappa_interpretation = interpret_kappa(kappa),
        confusion_matrix     = confusion_matrix(graded),
        passes_threshold     = (not math.isnan(kappa)) and kappa > threshold,
        threshold            = threshold,
        by_goal_type         = _stratum_breakdown(graded, "goal_type"),
        by_condition         = _stratum_breakdown(graded, "condition"),
    )


_BAR = "=" * 76
_RULE = "-" * 76


def _truncate(s: str, n: int) -> str:
    s = s if isinstance(s, str) else str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"\n  […truncated; original length = {len(s)} chars]"


def _wrap(s: str, indent: str = "  ") -> str:
    return "\n".join(textwrap.fill(line, width=72, initial_indent=indent,
                                   subsequent_indent=indent)
                     for line in s.splitlines())


def _render_item(it: CalibrationItem, n_total: int) -> str:
    return "\n".join([
        "",
        _BAR,
        f"ITEM {it.item_idx}/{n_total}    "
        f"goal={it.goal_id}  type={it.goal_type}  "
        f"cond={it.condition}  seed={it.seed}",
        f"check_id={it.check_id}  field={it.field}",
        _BAR,
        "",
        "GOAL:",
        _wrap(_truncate(it.goal_text, 800)),
        "",
        "GROUND TRUTH (reference):",
        _wrap(_truncate(it.ground_truth, 800)),
        "",
        "RUBRIC CRITERION (this is what you must judge against):",
        _wrap(_truncate(it.criteria, 600)),
        "",
        f"AGENT'S ANSWER (extracted '{it.field}' field):",
        _wrap(_truncate(it.extracted_text or "[empty]", 1500)),
        "",
        _RULE,
    ])


def _prompt_human(prompt_writer=sys.stdout, prompt_reader=input) -> Optional[bool]:
    """
    Prompt the operator for y / n / s (skip) / q (quit).

    Returns True (pass), False (fail), or None (skip).
    Quit raises KeyboardInterrupt so the outer loop can save and exit.
    """
    while True:
        prompt_writer.write(
            "Your verdict — does the answer satisfy the criterion? "
            "[y=PASS / n=FAIL / s=skip / q=quit]: "
        )
        prompt_writer.flush()
        try:
            raw = prompt_reader().strip().lower()
        except EOFError:
            print()
            raise KeyboardInterrupt
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        if raw in ("s", "skip"):
            return None
        if raw in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        prompt_writer.write("  invalid — please enter y, n, s, or q.\n")


def _ai_verdict(judge: Any, item: CalibrationItem) -> tuple[Optional[bool], Optional[str]]:
    """Call the judge and return (verdict, error). Errors are captured, not raised."""
    try:
        verdict = judge.judge(item.extracted_text, item.criteria)
        return bool(verdict), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def grade_interactively(
    items:           list[CalibrationItem],
    judge:           Any,
    output_path:     Path,
    *,
    incremental_save: bool = True,
    prompt_reader:    Any  = input,
    prompt_writer:    Any  = sys.stdout,
) -> list[CalibrationItem]:
    """
    Walk through *items*, grade each one, save partial state after every item.

    Returns the (possibly partially-graded) list.  If KeyboardInterrupt
    is raised mid-loop, the partial state is still on disk and the
    function returns what's been graded so far.
    """
    n = len(items)
    for idx, item in enumerate(items, start=1):
        prompt_writer.write(_render_item(item, n_total=n))
        prompt_writer.flush()

        # AI verdict first — independent of human, so its result is not
        # influenced by what the human says.  Render it AFTER the human
        # has decided (below) to avoid biasing the human grader.
        ai, ai_err = _ai_verdict(judge, item)
        item.ai_verdict = ai
        item.ai_error   = ai_err

        try:
            human = _prompt_human(prompt_writer=prompt_writer,
                                  prompt_reader=prompt_reader)
        except KeyboardInterrupt:
            prompt_writer.write(
                "\n[interrupted — partial results saved to "
                f"{output_path}]\n"
            )
            if incremental_save:
                _save_partial(items, output_path, judge)
            return items
        item.human_verdict = human

        # Inline progress feedback.
        if human is None:
            tag = "SKIPPED"
        else:
            agree = (ai is not None) and (human == ai)
            tag   = (
                f"Human={'PASS' if human else 'FAIL'}  "
                f"AI={'PASS' if ai else ('FAIL' if ai is False else 'ERROR')}  "
                f"{'(agree)' if agree else '(DISAGREE)' if ai is not None else '(ai error)'}"
            )
        prompt_writer.write(f"\n  → {tag}\n")
        if ai_err:
            prompt_writer.write(f"  AI error: {ai_err}\n")

        # Running agreement (only on graded so far).
        graded_so_far = [it for it in items[:idx]
                         if it.human_verdict is not None and it.ai_verdict is not None]
        if graded_so_far:
            agreed = sum(1 for it in graded_so_far if it.human_verdict == it.ai_verdict)
            prompt_writer.write(
                f"  Running agreement: {agreed}/{len(graded_so_far)} "
                f"({agreed / len(graded_so_far):.1%})\n"
            )
        prompt_writer.flush()

        if incremental_save:
            _save_partial(items, output_path, judge)
    return items


def _now_iso_z() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_metadata(
    runs_path:     Path,
    judge:         Any,
    judge_backend: str,
    n_requested:   int,
    seed:          int,
    threshold:     float,
) -> dict[str, Any]:
    return {
        "calibration_id":      uuid.uuid4().hex,
        "started_at":          _now_iso_z(),
        "runs_path":           str(runs_path),
        "judge_backend":       judge_backend,
        "judge_model":         str(getattr(judge, "_model", "unknown")),
        "n_requested":         n_requested,
        "sample_seed":         seed,
        "kappa_threshold":     threshold,
        "host":                os.uname().nodename if hasattr(os, "uname") else "",
        "python_version":      sys.version.split()[0],
    }


def _save_partial(items: list[CalibrationItem], path: Path, judge: Any) -> None:
    """Write a transient JSON snapshot of progress (overwritten on every call)."""
    snapshot = {
        "items":   [asdict(it) for it in items],
        "summary": asdict(summarise(items, threshold=0.6)),
        "_partial": True,
        "judge_total_api_calls": int(getattr(judge, "total_api_calls", 0)),
        "judge_total_cost_usd":  float(getattr(judge, "total_cost_usd",  0.0)),
        "saved_at": _now_iso_z(),
    }
    path.write_text(json.dumps(snapshot, indent=2))


def _save_final(
    result: CalibrationResult,
    judge:  Any,
    path:   Path,
) -> None:
    payload = result.to_dict()
    payload["judge_total_api_calls"] = int(getattr(judge, "total_api_calls", 0))
    payload["judge_total_cost_usd"]  = float(getattr(judge, "total_cost_usd",  0.0))
    payload["_partial"] = False
    payload["saved_at"] = _now_iso_z()
    path.write_text(json.dumps(payload, indent=2))


def render_summary(result: CalibrationResult) -> str:
    s = result.summary
    lines = [
        "",
        _BAR,
        "  JUDGE CALIBRATION SUMMARY",
        _BAR,
        f"  judge backend:     {result.metadata['judge_backend']}",
        f"  judge model:       {result.metadata['judge_model']}",
        f"  runs file:         {result.metadata['runs_path']}",
        f"  sample seed:       {result.metadata['sample_seed']}",
        f"  N requested:       {result.metadata['n_requested']}",
        "",
        f"  total items:       {s.n_total}",
        f"  graded (human+AI): {s.n_graded}",
        f"  skipped by human:  {s.n_skipped}",
        f"  AI errors:         {s.n_ai_errors}",
        "",
        f"  raw agreement:     {s.raw_agreement:.1%}",
        f"  Cohen's kappa:     {s.cohens_kappa:.4f}  ({s.kappa_interpretation})",
        f"  threshold (DEV-019): kappa > {s.threshold}  →  "
        f"{'PASS' if s.passes_threshold else 'FAIL'}",
        "",
        "  Confusion matrix (Human vs AI):",
        "                      AI=Pass    AI=Fail",
        f"      Human=Pass  {s.confusion_matrix['human_pass_ai_pass']:>6}     "
        f"{s.confusion_matrix['human_pass_ai_fail']:>6}",
        f"      Human=Fail  {s.confusion_matrix['human_fail_ai_pass']:>6}     "
        f"{s.confusion_matrix['human_fail_ai_fail']:>6}",
        "",
    ]
    if s.by_goal_type:
        lines.append("  By goal_type:")
        for k, v in sorted(s.by_goal_type.items()):
            kappa_str = "n/a" if math.isnan(v["cohens_kappa"]) else f"{v['cohens_kappa']:.3f}"
            lines.append(
                f"      {k:<28} n={v['n']:>3}   agree={v['agreement']:.1%}   κ={kappa_str}"
            )
        lines.append("")
    if s.by_condition:
        lines.append("  By condition:")
        for k, v in sorted(s.by_condition.items()):
            kappa_str = "n/a" if math.isnan(v["cohens_kappa"]) else f"{v['cohens_kappa']:.3f}"
            lines.append(
                f"      {k:<28} n={v['n']:>3}   agree={v['agreement']:.1%}   κ={kappa_str}"
            )
        lines.append("")
    lines.append(_BAR)
    return "\n".join(lines)


def render_per_item_table(items: list[CalibrationItem]) -> str:
    rows = [
        ("idx", "goal_id", "cond", "seed", "human", "ai", "agree"),
        ("---", "---",     "---",  "---",  "---",   "---","---"),
    ]
    for it in items:
        h_val = "—" if it.human_verdict is None else ("PASS" if it.human_verdict else "FAIL")
        a_val = (
            "ERR" if it.ai_error else
            ("—" if it.ai_verdict is None else
             ("PASS" if it.ai_verdict else "FAIL"))
        )
        if it.human_verdict is None or it.ai_verdict is None:
            agree = "—"
        else:
            agree = "✓" if it.human_verdict == it.ai_verdict else "✗"
        rows.append((
            str(it.item_idx),
            it.goal_id[:20],
            it.condition[:18],
            str(it.seed),
            h_val,
            a_val,
            agree,
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        out.append("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "python -m mas.eval.calibrate_judge",
        description = (
            "Interactive judge calibration harness (DEV-019).  "
            "Samples N (run, semantic_judge_check) pairs and computes "
            "Cohen's kappa between human and OllamaJudge verdicts."
        ),
    )
    p.add_argument(
        "--runs", "-r",
        dest    = "runs_path",
        default = None,
        help    = "Path to runs.jsonl (default: $MAS_RUNS_LOG or ./runs.jsonl).",
    )
    p.add_argument(
        "--n",
        dest    = "n",
        type    = int,
        default = 30,
        help    = "Number of items to grade (default: 30, per DEV-019).",
    )
    p.add_argument(
        "--judge-backend",
        dest    = "judge_backend",
        default = "ollama",
        choices = ["ollama", "anthropic"],
        help    = "Judge backend (default: ollama).",
    )
    p.add_argument(
        "--judge-model",
        dest    = "judge_model",
        default = None,
        help    = (
            "Override the judge model (default: 'qwen2.5:14b' for ollama; "
            "see DEV-019 for the same-model-bias rationale)."
        ),
    )
    p.add_argument(
        "--ollama-url",
        dest    = "ollama_url",
        default = "http://localhost:11434",
        help    = "Ollama base URL (default: http://localhost:11434).",
    )
    p.add_argument(
        "--output", "-o",
        dest    = "output_path",
        default = None,
        help    = (
            "Output JSON path (default: "
            "calibration_results_<UTC-timestamp>.json)."
        ),
    )
    p.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Sampling seed (default: 42).",
    )
    p.add_argument(
        "--threshold",
        type    = float,
        default = 0.6,
        help    = "DEV-019 acceptance threshold for κ (default: 0.6).",
    )
    p.add_argument(
        "--prioritise",
        dest    = "prioritise",
        default = "adversarial_reliability",
        help    = (
            "Goal type to prioritise during sampling (default: "
            "adversarial_reliability).  Pass 'none' to disable."
        ),
    )
    p.add_argument(
        "--no-incremental-save",
        dest    = "incremental_save",
        action  = "store_false",
        default = True,
        help    = (
            "Disable per-item save.  Default behaviour writes the "
            "output JSON after every grade so Ctrl+C does not lose "
            "progress."
        ),
    )
    return p


def _resolve_runs_path(arg_value: Optional[str]) -> Path:
    raw = (
        arg_value
        or os.environ.get("MAS_RUNS_LOG")
        or "./runs.jsonl"
    )
    return Path(raw)


def _read_runs_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"runs.jsonl not found at {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("calibrate_judge: line %d malformed (%s) — skipped", line_no, exc)
    return rows


def _build_judge(
    *,
    judge_backend: str,
    judge_model:   Optional[str],
    ollama_url:    str,
):
    if judge_backend == "ollama":
        from mas.eval.judge import OllamaJudge  # noqa: PLC0415
        return OllamaJudge(
            model      = judge_model or OllamaJudge.DEFAULT_MODEL,
            ollama_url = ollama_url,
        )
    if judge_backend == "anthropic":
        from mas.eval.judge import AnthropicJudge  # noqa: PLC0415
        return AnthropicJudge(model=judge_model or AnthropicJudge.DEFAULT_MODEL)
    raise ValueError(f"Unsupported judge_backend: {judge_backend!r}")


def _resolve_output_path(arg_value: Optional[str]) -> Path:
    if arg_value:
        return Path(arg_value)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"calibration_results_{ts}.json")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)

    runs_path   = _resolve_runs_path(args.runs_path)
    output_path = _resolve_output_path(args.output_path)
    threshold   = float(args.threshold)
    prioritise  = None if args.prioritise.lower() == "none" else args.prioritise

    print(f"\nReading runs from {runs_path} ...")
    try:
        runs = _read_runs_jsonl(runs_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  loaded {len(runs)} run(s)")

    print("Loading goal suite ...")
    try:
        goals = load_all_goals()
    except Exception as exc:
        print(f"ERROR: failed to load goal suite: {exc}", file=sys.stderr)
        return 1
    goals_index = {g.id: g for g in goals}
    print(f"  loaded {len(goals_index)} goal(s)")

    print("Building candidate (run, semantic_check) pairs ...")
    candidates = build_candidate_items(runs, goals_index)
    print(f"  {len(candidates)} eligible item(s) found")
    if not candidates:
        print(
            "ERROR: no semantic_judge rubric checks present in any run.\n"
            "       Either runs.jsonl is empty/has only crashed runs, or your "
            "rubric design contains no semantic_judge checks (current goals "
            "have 14 such checks across 14 goal files; see DEV-019).",
            file=sys.stderr,
        )
        return 1

    items = sample_items(
        candidates,
        n=args.n,
        seed=args.seed,
        prioritise_goal_type=prioritise,
    )
    if len(items) < args.n:
        print(
            f"WARNING: only {len(items)} candidate item(s) available "
            f"(requested {args.n}).  Proceeding with the available items."
        )

    print("Building judge client ...")
    try:
        judge = _build_judge(
            judge_backend = args.judge_backend,
            judge_model   = args.judge_model,
            ollama_url    = args.ollama_url,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metadata = _build_metadata(
        runs_path     = runs_path,
        judge         = judge,
        judge_backend = args.judge_backend,
        n_requested   = args.n,
        seed          = args.seed,
        threshold     = threshold,
    )
    metadata["n_actual"] = len(items)

    print(
        f"\nReady to grade {len(items)} item(s).  "
        "y=PASS  n=FAIL  s=skip  q=quit-and-save\n"
    )

    items = grade_interactively(
        items,
        judge            = judge,
        output_path      = output_path,
        incremental_save = args.incremental_save,
    )

    summary = summarise(items, threshold=threshold)
    metadata["finished_at"] = _now_iso_z()
    result = CalibrationResult(
        metadata = metadata,
        items    = items,
        summary  = summary,
    )
    _save_final(result, judge, output_path)

    print(render_per_item_table(items))
    print(render_summary(result))
    print(f"  full results saved to: {output_path}")

    if not summary.passes_threshold:
        print(
            "\nNOTE: Cohen's kappa below the DEV-019 threshold.  "
            "OllamaJudge results should be reported as exploratory "
            "(not confirmatory) in RESULTS.md."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
