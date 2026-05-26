"""Composite descriptive metrics for the System-3/0 evaluation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


#: PoC-default JPI weights (w1 Accuracy, w2 V_s, w3 log(E_b)).  These are
#: the values in `poc/harness.py:74-81` and `poc/visualizer.py:234`.  The
#: thesis (Eq. 3.22 / 6.4) does not fix specific weights.
DEFAULT_JPI_WEIGHTS: tuple[float, float, float] = (0.50, 0.30, 0.20)

#: Lower clamp for E_b before taking log to avoid log(0) → -inf.
#: Matches `poc/harness.py:79`.
_E_B_FLOOR: float = 1e-9


def _is_completed(run: Mapping[str, Any]) -> bool:
    """A run is 'completed' iff crash_reason is None/empty.  Crashed runs
    are excluded from accuracy / V_s / E_b — they would otherwise inflate
    failure counts in ways the thesis metrics weren't designed for."""
    cr = run.get("crash_reason")
    return not cr  # None or "" → True


def _success(run: Mapping[str, Any]) -> int:
    """1 iff `success` field is 1, else 0.  No crash filtering — this is
    the raw harness success indicator used inside E_b's numerator."""
    return 1 if int(run.get("success", 0)) == 1 else 0


def _ground_truth_match(run: Mapping[str, Any]) -> Optional[bool]:
    """`ground_truth_match` may be True / False / None (ungraded).
    Returns the raw value; callers decide how to handle None."""
    v = run.get("ground_truth_match")
    if v is None:
        return None
    return bool(v)


def _all_steps_schema_valid(run: Mapping[str, Any]) -> bool:
    """
    True iff every per_step entry has schema_valid=True (i.e. the entire
    trajectory was schema-clean).  This implements V_s with the audit
    semantics: y_i is a full trajectory, and y_i ∈ L(G) iff no per-step
    violation occurred.

    The thesis defines V_s on a single output y_i; for multi-step runs
    we apply the conjunction across steps because L(G) is the per-step
    schema and a violation at any step is a trajectory-level violation.
    Document this choice in DEV-019.
    """
    per_step = run.get("per_step") or []
    if not per_step:
        return False
    return all(bool(s.get("schema_valid", False)) for s in per_step)


def _total_tokens(run: Mapping[str, Any]) -> int:
    tok = run.get("tokens") or {}
    if isinstance(tok, Mapping):
        return int(tok.get("total", 0))
    return 0


def accuracy(runs: Iterable[Mapping[str, Any]]) -> float:
    """
    Accuracy = mean of `ground_truth_match` over completed, graded runs.

    Runs with `ground_truth_match is None` (ungraded) are excluded.
    Runs with `crash_reason` populated are also excluded — a crashed run
    with `ground_truth_match=False` is a deployment failure, not an
    accuracy datapoint.

    Returns 0.0 on empty input.
    """
    vals: list[int] = []
    for r in runs:
        if not _is_completed(r):
            continue
        m = _ground_truth_match(r)
        if m is None:
            continue
        vals.append(1 if m else 0)
    return sum(vals) / len(vals) if vals else 0.0


def schema_compliance(runs: Iterable[Mapping[str, Any]]) -> float:
    """
    V_s = (1/N) Σ I(y_i ∈ L(G))  — thesis Eq. 3.19 / 6.1.

    Implemented as: fraction of runs whose entire per_step trajectory
    is schema_valid=True.  Crashed runs are counted as non-compliant
    (their trajectories cannot be validated by definition).

    Returns 0.0 on empty input.
    """
    runs_list = list(runs)
    n = len(runs_list)
    if n == 0:
        return 0.0
    compliant = sum(1 for r in runs_list if _all_steps_schema_valid(r))
    return compliant / n


def budget_efficiency(runs: Iterable[Mapping[str, Any]]) -> float:
    """
    E_b = (Σ I(success_i) / Σ Tokens_i) × 1000  — thesis Eq. 3.20 / 6.2.

    "Successes per 1,000 tokens spent."  Returns 0.0 if total tokens is
    zero (no inference happened) — by convention rather than infinity.
    """
    runs_list = list(runs)
    successes  = sum(_success(r) for r in runs_list)
    total_toks = sum(_total_tokens(r) for r in runs_list)
    if total_toks <= 0:
        return 0.0
    return (successes / total_toks) * 1000.0


def jpi(
    runs:    Iterable[Mapping[str, Any]],
    weights: tuple[float, float, float] = DEFAULT_JPI_WEIGHTS,
) -> float:
    """
    Joint Performance Index — thesis Eq. 3.22 / 6.4:

        JPI = w_1 · Accuracy  +  w_2 · V_s  +  w_3 · log(E_b)

    Uses `max(E_b, 1e-9)` inside log() to match the PoC harness clamp
    (`poc/harness.py:79`).  The default weights (0.5, 0.3, 0.2) are the
    PoC values; pass a different tuple for sensitivity analysis.
    """
    runs_list = list(runs)
    w1, w2, w3 = weights
    a  = accuracy(runs_list)
    vs = schema_compliance(runs_list)
    eb = budget_efficiency(runs_list)
    return w1 * a + w2 * vs + w3 * math.log(max(eb, _E_B_FLOOR))


def cfr(runs: Iterable[Mapping[str, Any]]) -> float:
    """
    Cascade Failure Rate = (1/N) Σ I(cascade_failure_i = 1).

    Reads the per-run `cascade_failure` field already populated by
    `mas/eval/cascade_detector.detect_cascade`.  Returns 0.0 on empty input.
    """
    runs_list = list(runs)
    n = len(runs_list)
    if n == 0:
        return 0.0
    return sum(1 for r in runs_list if int(r.get("cascade_failure") or 0) == 1) / n


def success_rate(runs: Iterable[Mapping[str, Any]]) -> float:
    """Fraction of runs with `success == 1`.  Counts crashed runs as failures."""
    runs_list = list(runs)
    n = len(runs_list)
    if n == 0:
        return 0.0
    return sum(_success(r) for r in runs_list) / n


MetricFn = Callable[[Sequence[Mapping[str, Any]]], float]


def bootstrap_ci(
    runs:  Sequence[Mapping[str, Any]],
    fn:    MetricFn,
    *,
    n:     int   = 10_000,
    alpha: float = 0.05,
    seed:  int   = 0,
) -> tuple[float, float]:
    """
    Non-parametric percentile bootstrap CI for *fn* applied to *runs*.

    Resamples *runs* with replacement *n* times, computes *fn* on each
    resample, returns the (alpha/2, 1-alpha/2) percentiles.  Returns
    (point, point) on empty input.
    """
    if not runs:
        point = fn(())
        return (point, point)

    rng = random.Random(seed)
    indices = list(range(len(runs)))
    samples: list[float] = []
    for _ in range(n):
        boot = [runs[rng.choice(indices)] for _ in indices]
        samples.append(fn(boot))
    samples.sort()
    lo_idx = int((alpha / 2.0) * n)
    hi_idx = int((1.0 - alpha / 2.0) * n) - 1
    lo_idx = max(0, min(lo_idx, n - 1))
    hi_idx = max(0, min(hi_idx, n - 1))
    return (samples[lo_idx], samples[hi_idx])


@dataclass(frozen=True)
class ConditionMetrics:
    """Aggregated descriptive metrics for one experimental condition."""
    condition:        str
    n_runs:           int
    accuracy:         float
    schema_v_s:       float
    budget_e_b:       float
    log_e_b:          float
    jpi:              float
    cfr:              float
    success_rate:     float
    accuracy_ci:      tuple[float, float]
    schema_v_s_ci:    tuple[float, float]
    budget_e_b_ci:    tuple[float, float]
    jpi_ci:           tuple[float, float]
    cfr_ci:           tuple[float, float]


def metrics_for_condition(
    condition: str,
    runs:      Sequence[Mapping[str, Any]],
    *,
    weights:        tuple[float, float, float] = DEFAULT_JPI_WEIGHTS,
    bootstrap_n:    int   = 10_000,
    bootstrap_seed: int   = 0,
) -> ConditionMetrics:
    """Compute the full descriptive panel for one condition with CIs."""
    runs_seq = list(runs)
    a  = accuracy(runs_seq)
    vs = schema_compliance(runs_seq)
    eb = budget_efficiency(runs_seq)
    j  = jpi(runs_seq, weights=weights)
    c  = cfr(runs_seq)
    s  = success_rate(runs_seq)

    a_ci  = bootstrap_ci(runs_seq, accuracy,
                         n=bootstrap_n, seed=bootstrap_seed)
    vs_ci = bootstrap_ci(runs_seq, schema_compliance,
                         n=bootstrap_n, seed=bootstrap_seed)
    eb_ci = bootstrap_ci(runs_seq, budget_efficiency,
                         n=bootstrap_n, seed=bootstrap_seed)
    j_ci  = bootstrap_ci(runs_seq, lambda rs: jpi(rs, weights=weights),
                         n=bootstrap_n, seed=bootstrap_seed)
    c_ci  = bootstrap_ci(runs_seq, cfr,
                         n=bootstrap_n, seed=bootstrap_seed)

    return ConditionMetrics(
        condition     = condition,
        n_runs        = len(runs_seq),
        accuracy      = a,
        schema_v_s    = vs,
        budget_e_b    = eb,
        log_e_b       = math.log(max(eb, _E_B_FLOOR)),
        jpi           = j,
        cfr           = c,
        success_rate  = s,
        accuracy_ci   = a_ci,
        schema_v_s_ci = vs_ci,
        budget_e_b_ci = eb_ci,
        jpi_ci        = j_ci,
        cfr_ci        = c_ci,
    )


def metrics_table(
    runs_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights:        tuple[float, float, float] = DEFAULT_JPI_WEIGHTS,
    bootstrap_n:    int   = 10_000,
    bootstrap_seed: int   = 0,
) -> dict[str, ConditionMetrics]:
    """Compute ConditionMetrics for every condition in *runs_by_condition*."""
    return {
        cond: metrics_for_condition(
            cond, runs,
            weights        = weights,
            bootstrap_n    = bootstrap_n,
            bootstrap_seed = bootstrap_seed,
        )
        for cond, runs in runs_by_condition.items()
    }
