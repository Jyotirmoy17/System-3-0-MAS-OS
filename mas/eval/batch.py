"""Batch orchestrator for the full 300-run experimental matrix (150 goals × 2 conditions)."""
from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, NamedTuple

log = logging.getLogger(__name__)


RunLog = dict[str, Any]


class RunSpec(NamedTuple):
    """One atomic unit of work: a (goal_id, condition, seed) triple."""
    goal_id:   str
    condition: str
    seed:      int


_ALL_CONDITIONS: list[str] = [
    "FULL_FRAMEWORK",
    "BASELINE_LANGGRAPH",
]

_ALL_GOAL_TYPES: list[str] = [
    "adversarial_reliability",
    "code_analysis",
    "financial_analysis",
    "research_synthesis",
    "sequential_reasoning",
]

# Ollama concurrency limit on a 24 GB Ryzen (no GPU)
_MAX_PARALLEL: int = 2


def build_run_queue(
    conditions:         list[str],
    goal_types:         list[str] | None = None,
    instance_ids:       list[str] | None = None,
    seeds_per_instance: int               = 1,
    goals_dir:          Path | None       = None,
) -> list[RunSpec]:
    """Build the ordered list of RunSpecs for a batch run.

Ordering: goal_id (as discovered) × condition (as provided) × seed."""
    if instance_ids is not None:
        effective_ids: list[str] = list(instance_ids)
    else:
        from mas.eval.goal_loader import load_all_goals  # noqa: PLC0415
        goals = load_all_goals(goals_dir)
        if goal_types:
            type_set = set(goal_types)
            goals    = [g for g in goals if g.goal_type in type_set]
        effective_ids = [g.id for g in goals]

    seeds = list(range(seeds_per_instance))
    queue: list[RunSpec] = [
        RunSpec(gid, cond, seed)
        for gid   in effective_ids
        for cond  in conditions
        for seed  in seeds
    ]
    return queue


def load_completed(completed_path: Path) -> set[tuple[str, str, int]]:
    """Load the set of finished (goal_id, condition, seed) triples."""
    done: set[tuple[str, str, int]] = set()
    if not completed_path.exists():
        return done
    for raw_line in completed_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            try:
                done.add((parts[0], parts[1], int(parts[2])))
            except ValueError:
                log.warning("Malformed line in %s: %r", completed_path.name, line)
    return done


def append_completed(completed_path: Path, spec: RunSpec) -> None:
    """Atomically append one completed triple to runs_completed.txt.

Uses fcntl.LOCK_EX so parallel workers don't interleave writes."""
    completed_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{spec.goal_id}\t{spec.condition}\t{spec.seed}\n"
    with completed_path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(line)
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def append_run_log(runs_jsonl: Path, run_log: RunLog) -> None:
    """Append one RunLog JSON line to runs.jsonl under an exclusive file lock."""
    runs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(run_log, default=str) + "\n"
    with runs_jsonl.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(line)
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def format_progress_line(n: int, total: int, run_log: RunLog) -> str:
    """Format a single progress line for stdout."""
    return (
        f"[{n}/{total}] {run_log['goal_id']} {run_log['condition']} "
        f"success={run_log['success']} cascade={run_log['cascade_failure']} "
        f"latency={run_log.get('latency_s', 0.0):.1f}s"
    )


def print_summary(results: list[RunLog]) -> None:
    """Print the end-of-batch summary table to stdout."""
    from collections import defaultdict  # noqa: PLC0415

    by_cond: dict[str, list[RunLog]] = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    col_w = 22
    header = (
        f"{'Condition':<{col_w}} {'Runs':>5}  "
        f"{'Success%':>9}  {'CFR':>7}  {'Mean tokens':>11}"
    )
    print("\n" + header)
    print("-" * len(header))

    for cond in _ALL_CONDITIONS:
        logs = by_cond.get(cond)
        if not logs:
            continue
        n    = len(logs)
        succ = sum(r["success"] for r in logs)
        cfr  = sum(r["cascade_failure"] for r in logs)
        toks = sum(r["tokens"]["total"] for r in logs)
        print(
            f"{cond:<{col_w}} {n:>5}  "
            f"{100 * succ / n:>8.1f}%  "
            f"{100 * cfr  / n:>6.1f}%  "
            f"{toks // n:>11,}"
        )


def _default_run_fn(spec: RunSpec, overrides: dict[str, Any]) -> RunLog:
    """Sequential-mode adapter: call run_one() with no pre-warming."""
    from mas.eval.harness import run_one  # noqa: PLC0415
    return run_one(
        spec.goal_id, spec.condition, spec.seed, overrides,
        _skip_warmup=True,
    )


def _default_preflight(goals_dir: Path | None) -> Any:
    """Default preflight adapter."""
    from mas.eval.preflight import run_preflight  # noqa: PLC0415
    return run_preflight(goals_dir)


def _worker_fn(
    spec:               RunSpec,
    overrides:          dict[str, Any],
    runs_jsonl_str:     str,
    completed_path_str: str,
) -> RunLog:
    """Process-pool worker: execute one RunSpec and persist results.

Pre-warms models on the first call in each worker process.
Writes to runs.jsonl and runs_completed.txt under fcntl locks
so concurrent workers don't corrupt either file."""
    from mas.eval.harness import run_one, prewarm  # noqa: PLC0415

    # Warm once per worker process.
    if not getattr(_worker_fn, "_warmed", False):
        try:
            prewarm(
                overrides.get("slm",        "phi3:mini"),
                overrides.get("llm",        "llama3.1:8b"),
                overrides.get("ollama_url", "http://localhost:11434"),
            )
        except Exception as exc:
            log.warning("Worker pre-warm failed: %s", exc)
        _worker_fn._warmed = True  # type: ignore[attr-defined]

    run_log = run_one(
        spec.goal_id, spec.condition, spec.seed, overrides,
        _skip_warmup=True,
    )
    append_run_log(Path(runs_jsonl_str), run_log)
    append_completed(Path(completed_path_str), spec)
    return run_log


_interrupted: bool = False


def _handle_sigint(sig: int, frame: Any) -> None:
    global _interrupted
    _interrupted = True
    print(
        "\n[SIGINT] Interrupt received — finishing current run …",
        file=sys.stderr,
        flush=True,
    )


def _run_sequential(
    queue:          list[RunSpec],
    runs_jsonl:     Path,
    completed_path: Path,
    overrides:      dict[str, Any],
    run_fn:         Callable[[RunSpec, dict[str, Any]], RunLog],
    total:          int,
) -> list[RunLog]:
    """Execute *queue* sequentially, respecting _interrupted flag."""
    results: list[RunLog] = []
    for i, spec in enumerate(queue, 1):
        if _interrupted:
            break
        try:
            run_log = run_fn(spec, overrides)
            append_run_log(runs_jsonl, run_log)
            append_completed(completed_path, spec)
            results.append(run_log)
            print(format_progress_line(i, total, run_log), flush=True)
        except Exception as exc:
            log.error(
                "Run %d/%d failed for %s/%s: %s",
                i, total, spec.goal_id, spec.condition, exc,
            )
    return results


def _run_parallel(
    queue:          list[RunSpec],
    runs_jsonl:     Path,
    completed_path: Path,
    overrides:      dict[str, Any],
    parallel:       int,
    total:          int,
) -> list[RunLog]:
    """Execute *queue* using ProcessPoolExecutor."""
    results: list[RunLog] = []
    runs_jsonl_str     = str(runs_jsonl)
    completed_path_str = str(completed_path)

    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _worker_fn, spec, overrides,
                runs_jsonl_str, completed_path_str,
            ): (idx, spec)
            for idx, spec in enumerate(queue, 1)
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            idx, spec = futures[future]
            if _interrupted:
                pool.shutdown(wait=False, cancel_futures=True)
                break
            try:
                run_log = future.result()
                results.append(run_log)
                print(format_progress_line(done_count, total, run_log), flush=True)
            except Exception as exc:
                log.error(
                    "Worker failed for %s/%s: %s",
                    spec.goal_id, spec.condition, exc,
                )
    return results


def run_batch(
    queue:           list[RunSpec],
    runs_jsonl:      Path,
    completed_path:  Path,
    model_overrides: dict[str, Any] | None = None,
    parallel:        int                   = 1,
    max_runs:        int | None            = None,
    dry_run:         bool                  = False,
    resume:          bool                  = False,
    skip_warmup:     bool                  = False,
    skip_preflight:  bool                  = False,
    *,
    _run_one_fn:    Callable[[RunSpec, dict[str, Any]], RunLog] | None = None,
    _preflight_fn:  Callable[[Path | None], Any] | None               = None,
    _goals_dir:     Path | None                                        = None,
) -> list[RunLog]:
    """Orchestrate the batch run queue."""
    global _interrupted
    _interrupted = False

    overrides = model_overrides or {}
    parallel  = min(max(parallel, 1), _MAX_PARALLEL)

    if resume:
        done  = load_completed(completed_path)
        queue = [s for s in queue if (s.goal_id, s.condition, s.seed) not in done]
        log.info("Resume: %d triples completed, %d remaining.", len(done), len(queue))

    if dry_run:
        effective = queue[:max_runs] if max_runs is not None else queue
        m = len(effective)
        print(f"DRY RUN — {m} run(s) queued (not executing):")
        for i, spec in enumerate(effective, 1):
            print(f"  [{i}/{m}] {spec.goal_id} | {spec.condition} | seed={spec.seed}")
        return []

    if not skip_preflight:
        pf_fn  = _preflight_fn or _default_preflight
        report = pf_fn(_goals_dir)
        if not report.passed:
            n_err = len(report.errors)
            print(
                f"PRE-FLIGHT FAILED ({n_err} error(s)) — aborting batch.",
                file=sys.stderr,
            )
            for err in report.errors[:10]:
                print(f"  ✗ {err}", file=sys.stderr)
            if n_err > 10:
                print(f"  … and {n_err - 10} more", file=sys.stderr)
            raise SystemExit(1)
        for w in report.warnings:
            log.warning("Pre-flight: %s", w)

    if not skip_warmup and _run_one_fn is None:
        from mas.eval.harness import prewarm  # noqa: PLC0415
        try:
            prewarm(
                overrides.get("slm",        "phi3:mini"),
                overrides.get("llm",        "llama3.1:8b"),
                overrides.get("ollama_url", "http://localhost:11434"),
            )
        except Exception as exc:
            log.warning("Pre-warm failed: %s", exc)

    effective_queue = queue[:max_runs] if max_runs is not None else queue
    total = len(effective_queue)

    original_handler = signal.signal(signal.SIGINT, _handle_sigint)
    results: list[RunLog] = []

    try:
        if parallel >= 2 and _run_one_fn is None:
            # Process pool (real Ollama runs only; cannot pickle arbitrary fns)
            results = _run_parallel(
                effective_queue, runs_jsonl, completed_path,
                overrides, parallel, total,
            )
        else:
            # Sequential (default, and always used in tests)
            run_fn = _run_one_fn or _default_run_fn
            results = _run_sequential(
                effective_queue, runs_jsonl, completed_path,
                overrides, run_fn, total,
            )
    finally:
        signal.signal(signal.SIGINT, original_handler)

    if _interrupted:
        print(
            "\nBatch interrupted. To resume: add --resume to your command.",
            file=sys.stderr, flush=True,
        )

    if results:
        print_summary(results)

    return results
