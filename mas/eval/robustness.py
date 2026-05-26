"""Pre-registered robustness checks (EXPERIMENTS.md §7.5)."""
from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable

from mas.eval.cascade_detector import detect_cascade

log = logging.getLogger(__name__)


_ALL_GOAL_TYPES: list[str] = [
    "research_synthesis",
    "financial_analysis",
    "sequential_reasoning",
    "code_analysis",
    "adversarial_reliability",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GOALS_DIR  = _REPO_ROOT / "goals"

RunLog = dict[str, Any]


def _load_runs_from(path: str | Path) -> list[RunLog]:
    """Load all non-empty JSON lines from *path*.  Returns [] if file absent."""
    p = Path(path)
    if not p.exists():
        return []
    runs: list[RunLog] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Skipping malformed line in %s", path)
    return runs


def _append_run_log(path: str | Path, run_log: RunLog) -> None:
    """Atomically append one JSON run-log line to *path*.

Uses fcntl.LOCK_EX to avoid corruption from concurrent writers."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(run_log) + "\n")
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _compute_cfr(runs: list[RunLog]) -> tuple[float, int]:
    """Return (cfr, n) for *runs*.

CFR is computed from detect_cascade() (re-evaluates the three-condition
definition from EXPERIMENTS.md §5) rather than from the stored
``cascade_failure`` field, so it is correct even for robustness log
entries that may omit that field."""
    if not runs:
        return (0.0, 0)
    n = len(runs)
    k = sum(1 for r in runs if detect_cascade(r))
    return (k / n, n)


def _escalations_p1(runs: list[RunLog]) -> float:
    """Return the mean P1 escalation count across *runs*.  0.0 if empty."""
    if not runs:
        return 0.0
    total = sum(r.get("escalations", {}).get("P1", 0) for r in runs)
    return total / len(runs)


def _select_stratified(
    n_instances: int,
    goals_dir: Path | None = None,
    *,
    rng_seed: int = 42,
) -> list[str]:
    """Sample *n_instances* goal IDs stratified across the five goal types.

Takes ``n_instances // 5`` from each type (extras go to the first
types in order).  If a type has fewer instances than requested, all
available IDs for that type are used."""
    root = goals_dir or _GOALS_DIR
    rng  = random.Random(rng_seed)

    per_type = n_instances // len(_ALL_GOAL_TYPES)
    extras   = n_instances % len(_ALL_GOAL_TYPES)

    selected: list[str] = []
    for idx, goal_type in enumerate(_ALL_GOAL_TYPES):
        want = per_type + (1 if idx < extras else 0)
        type_dir = root / goal_type
        if not type_dir.exists():
            log.warning("Goal type directory missing: %s", type_dir)
            continue
        files = sorted(type_dir.glob("*.json"))
        ids   = [f.stem for f in files]
        chosen = rng.sample(ids, min(want, len(ids)))
        selected.extend(chosen)

    return selected


def _completed_set(runs: list[RunLog]) -> set[tuple[str, int]]:
    """Return a set of (goal_id, seed) pairs already in *runs*."""
    return {(r["goal_id"], int(r["seed"])) for r in runs}


def _default_run_one(
    goal_id: str,
    condition: str,
    seed: int,
    model_overrides: dict[str, Any] | None = None,
    *,
    _skip_warmup: bool = True,
    _goals_dir: Path | None = None,
) -> RunLog:
    """Thin wrapper that imports and calls the real run_one()."""
    from mas.eval.harness import run_one  # noqa: PLC0415
    return run_one(
        goal_id         = goal_id,
        condition       = condition,
        seed            = seed,
        model_overrides = model_overrides,
        _skip_warmup    = _skip_warmup,
        _goals_dir      = _goals_dir,
    )


def temperature_sweep(
    n_instances: int = 50,
    *,
    temperatures: list[float] | None = None,
    log_dir: str | Path = ".",
    goals_dir: Path | None = None,
    rng_seed: int = 42,
    _run_one_fn: Callable[..., RunLog] | None = None,
) -> dict:
    """Re-run 50 random instances at temperature ∈ {0.0, 0.7} (EXPERIMENTS.md §7.5).

Runs under FULL_FRAMEWORK for each temperature.  Results are written to
separate files so the check is independently resumable:

    {log_dir}/runs_robustness_temperature_t0.0.jsonl
    {log_dir}/runs_robustness_temperature_t0.7.jsonl

Note: temperature is passed via model_overrides but is not yet threaded
through to the kernel (see DEV-007).  The structural logic is complete."""
    if temperatures is None:
        temperatures = [0.0, 0.7]

    run_fn = _run_one_fn or _default_run_one
    log_dir_p = Path(log_dir)
    instance_ids = _select_stratified(n_instances, goals_dir, rng_seed=rng_seed)

    results: dict[str, dict] = {}

    for temp in temperatures:
        tag       = f"t{temp}"
        log_path  = log_dir_p / f"runs_robustness_temperature_{tag}.jsonl"
        existing  = _load_runs_from(log_path)
        done      = _completed_set(existing)

        for goal_id in instance_ids:
            key = (goal_id, 0)   # seed=0 for temperature sweep
            if key in done:
                log.debug("temperature_sweep: skip %s (already done)", goal_id)
                continue
            try:
                result = run_fn(
                    goal_id,
                    "FULL_FRAMEWORK",
                    0,
                    {"temperature": temp},
                    _skip_warmup=True,
                    _goals_dir=goals_dir,
                )
                _append_run_log(log_path, result)
            except Exception as exc:
                log.error("temperature_sweep: run_one failed for %s @t=%s: %s",
                          goal_id, temp, exc)

        final_runs = _load_runs_from(log_path)
        cfr, n     = _compute_cfr(final_runs)
        results[f"t={temp}"] = {"cfr": cfr, "n": n}

    # Delta direction: higher temperature generally raises CFR.
    # "preserved" means CFR is monotonically non-decreasing with temperature.
    sorted_temps  = sorted(temperatures)
    sorted_cfrs   = [results.get(f"t={t}", {}).get("cfr", 0.0) for t in sorted_temps]
    delta_preserved = all(
        sorted_cfrs[i] <= sorted_cfrs[i + 1]
        for i in range(len(sorted_cfrs) - 1)
    )
    results["delta_preserved"] = delta_preserved   # type: ignore[assignment]

    return results


def seed_sweep(
    n_instances: int = 50,
    *,
    base_seed: int = 0,
    seed_delta: int = 1000,
    log_dir: str | Path = ".",
    goals_dir: Path | None = None,
    rng_seed: int = 42,
    _run_one_fn: Callable[..., RunLog] | None = None,
) -> dict:
    """Re-run 50 random instances at seed=0 and seed=1000 (EXPERIMENTS.md §7.5).

Writes to:
    {log_dir}/runs_robustness_seed_s{base_seed}.jsonl
    {log_dir}/runs_robustness_seed_s{base_seed + seed_delta}.jsonl"""
    run_fn = _run_one_fn or _default_run_one
    log_dir_p = Path(log_dir)
    instance_ids = _select_stratified(n_instances, goals_dir, rng_seed=rng_seed)
    seeds = [base_seed, base_seed + seed_delta]

    results: dict[str, dict] = {}

    for seed_val in seeds:
        log_path  = log_dir_p / f"runs_robustness_seed_s{seed_val}.jsonl"
        existing  = _load_runs_from(log_path)
        done      = _completed_set(existing)

        for goal_id in instance_ids:
            key = (goal_id, seed_val)
            if key in done:
                log.debug("seed_sweep: skip %s seed=%d (already done)", goal_id, seed_val)
                continue
            try:
                result = run_fn(
                    goal_id,
                    "FULL_FRAMEWORK",
                    seed_val,
                    None,
                    _skip_warmup=True,
                    _goals_dir=goals_dir,
                )
                _append_run_log(log_path, result)
            except Exception as exc:
                log.error("seed_sweep: run_one failed for %s seed=%d: %s",
                          goal_id, seed_val, exc)

        final_runs = _load_runs_from(log_path)
        cfr, n     = _compute_cfr(final_runs)
        results[f"seed={seed_val}"] = {"cfr": cfr, "n": n}

    cfr_base  = results.get(f"seed={seeds[0]}",    {}).get("cfr", 0.0)
    cfr_delta = results.get(f"seed={seeds[1]}", {}).get("cfr", 0.0)
    delta_preserved = abs(cfr_base - cfr_delta) < 0.20
    results["delta_preserved"] = delta_preserved   # type: ignore[assignment]

    return results


def drop_adversarial_sensitivity(runs_path: str) -> dict:
    """Recompute primary analysis excluding adversarial_reliability runs.

This is a fully post-hoc check — no new runs are executed."""
    from mas.eval.statistics import primary_analysis  # noqa: PLC0415

    all_runs = _load_runs_from(runs_path)
    if not all_runs:
        raise FileNotFoundError(f"No runs found in {runs_path}")

    non_adv = [
        r for r in all_runs
        if r.get("goal_type") != "adversarial_reliability"
    ]
    n_excluded  = len(all_runs) - len(non_adv)
    n_remaining = len(non_adv)

    if not non_adv:
        return {
            "note":        "excluding adversarial",
            "n_excluded":  n_excluded,
            "n_remaining": 0,
            "cfr_by_condition": {},
            "alpha":            0.05,
            "tests":       {},
        }

    # Write filtered runs to a temp file and call the canonical primary_analysis().
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        for run in non_adv:
            tmp.write(json.dumps(run) + "\n")
        tmp_path = tmp.name

    try:
        result = primary_analysis(tmp_path)
    except Exception as exc:
        log.warning("drop_adversarial: primary_analysis raised: %s", exc)
        from mas.eval.statistics import (  # noqa: PLC0415
            ALPHA,
            compute_cfr_by_condition,
        )
        result = {
            "cfr_by_condition": compute_cfr_by_condition(non_adv),
            "alpha":            ALPHA,
            "tests":            {},
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    result["note"]        = "excluding adversarial"
    result["n_excluded"]  = n_excluded
    result["n_remaining"] = n_remaining
    return result


def threshold_sensitivity(
    taus: list[float] | None = None,
    *,
    n_instances: int = 50,
    log_dir: str | Path = ".",
    goals_dir: Path | None = None,
    rng_seed: int = 42,
    _run_one_fn: Callable[..., RunLog] | None = None,
) -> dict:
    """Re-run 50 random instances at τ_sys1 ∈ {0.5, 0.7, 0.9} (EXPERIMENTS.md §7.5).

Each τ writes to its own JSONL file so the check is independently resumable:
    {log_dir}/runs_robustness_threshold_tau0.5.jsonl"""
    if taus is None:
        taus = [0.5, 0.7, 0.9]

    run_fn = _run_one_fn or _default_run_one
    log_dir_p = Path(log_dir)
    instance_ids = _select_stratified(n_instances, goals_dir, rng_seed=rng_seed)

    results: dict[str, dict] = {}

    for tau in taus:
        tag      = f"tau{tau}"
        log_path = log_dir_p / f"runs_robustness_threshold_{tag}.jsonl"
        existing = _load_runs_from(log_path)
        done     = _completed_set(existing)

        for goal_id in instance_ids:
            key = (goal_id, 0)   # seed=0 for threshold sweep
            if key in done:
                log.debug("threshold_sensitivity: skip %s tau=%s (done)", goal_id, tau)
                continue
            try:
                result = run_fn(
                    goal_id,
                    "FULL_FRAMEWORK",
                    0,
                    {"tau": tau},
                    _skip_warmup=True,
                    _goals_dir=goals_dir,
                )
                _append_run_log(log_path, result)
            except Exception as exc:
                log.error("threshold_sensitivity: run_one failed %s tau=%s: %s",
                          goal_id, tau, exc)

        final_runs  = _load_runs_from(log_path)
        cfr, n      = _compute_cfr(final_runs)
        esc_p1      = _escalations_p1(final_runs)
        results[f"tau={tau}"] = {"cfr": cfr, "n": n, "escalations_p1": esc_p1}

    return results


def append_robustness_section(
    results_md_path: str | Path,
    check_name: str,
    result: dict,
) -> None:
    """Append (or update) a named robustness sub-section inside RESULTS.md.

Looks for a ``### Robustness Checks`` heading; if found, appends the
new sub-section under it.  If the heading is absent, appends the full
section at the end of the file.

Each call is idempotent for the same *check_name*: if a
``#### {check_name}`` sub-heading already exists it is replaced."""
    path = Path(results_md_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""

    sub_heading = f"#### {check_name}"
    new_block   = _format_robustness_block(check_name, result)

    if sub_heading in existing:
        lines   = existing.splitlines(keepends=True)
        out     = []
        skip    = False
        replaced = False
        for line in lines:
            if line.rstrip() == sub_heading and not replaced:
                # Consume until next #### or ### heading
                skip = True
                out.append(new_block + "\n")
                replaced = True
                continue
            if skip:
                stripped = line.rstrip()
                if stripped.startswith("####") or stripped.startswith("###"):
                    skip = False
                else:
                    continue
            out.append(line)
        existing = "".join(out)
    else:
        # Append under the Robustness Checks section heading
        robustness_heading = "### Robustness Checks"
        if robustness_heading in existing:
            existing = existing.replace(
                robustness_heading,
                robustness_heading + "\n\n" + new_block,
                1,
            )
        else:
            existing = existing.rstrip() + "\n\n## Robustness Checks\n\n" + new_block + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing, encoding="utf-8")
    log.info("Appended robustness section '%s' to %s", check_name, path)


def _format_robustness_block(check_name: str, result: dict) -> str:
    """Format a robustness result dict as a Markdown sub-section."""
    lines = [f"#### {check_name}\n"]

    if "note" in result:
        lines.append(f"*Note: {result['note']}*\n")

    if "n_excluded" in result:
        lines.append(
            f"Runs excluded: {result['n_excluded']}  "
            f"Runs used: {result['n_remaining']}\n"
        )

    # CFR table for temperature / seed sweeps (keys like "t=0.0" or "seed=0")
    cfr_keys = [k for k in result if k not in (
        "delta_preserved", "note", "n_excluded", "n_remaining",
        "cfr_by_condition", "alpha", "tests",
    ) and isinstance(result[k], dict) and "cfr" in result[k]]

    if cfr_keys:
        lines.append("| Setting | CFR | n |")
        lines.append("| --- | --- | --- |")
        for key in cfr_keys:
            d = result[key]
            lines.append(f"| {key} | {d['cfr']:.3f} | {d['n']} |")
        lines.append("")

    if "delta_preserved" in result:
        dp = result["delta_preserved"]
        lines.append(f"Delta direction preserved: **{'Yes' if dp else 'No'}**\n")

    # CFR by condition table (drop_adversarial)
    if "cfr_by_condition" in result and result["cfr_by_condition"]:
        lines.append("**CFR by condition (excluding adversarial):**\n")
        lines.append("| Condition | CFR | n |")
        lines.append("| --- | --- | --- |")
        for cond, d in sorted(result["cfr_by_condition"].items()):
            lines.append(f"| {cond} | {d['cfr']:.3f} | {d['n']} |")
        lines.append("")

    # Escalations column for threshold sensitivity
    esc_keys = [k for k in cfr_keys if "escalations_p1" in result.get(k, {})]
    if esc_keys:
        lines.append("| τ | CFR | n | Mean P1 escalations |")
        lines.append("| --- | --- | --- | --- |")
        for key in esc_keys:
            d = result[key]
            lines.append(
                f"| {key} | {d['cfr']:.3f} | {d['n']} "
                f"| {d.get('escalations_p1', 0.0):.3f} |"
            )
        lines.append("")

    return "\n".join(lines)
