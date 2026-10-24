"""Primary statistical analysis for the System-3/0 evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, norm
from scipy.stats import wilcoxon as scipy_wilcoxon

from mas.eval.cascade_detector import detect_cascade


#: Significance threshold for the 2-condition protocol (thesis §6.4).
ALPHA: float = 0.05

#: Conditions retained in the simplified protocol.
_KEEP_CONDITIONS: tuple[str, ...] = ("FULL_FRAMEWORK", "BASELINE_LANGGRAPH")

_ABLATIONS: tuple[str, ...] = ("BASELINE_LANGGRAPH",)


def _get_field(run: dict[str, Any], field: str) -> Any:
    """Retrieve a value from a run dict, supporting dot-notation for nested fields."""
    parts = field.split(".")
    val: Any = run
    for part in parts:
        if not isinstance(val, dict):
            raise KeyError(
                f"Field path '{field}' invalid: segment '{part}' reached a non-dict"
            )
        val = val[part]
    return val


def _pair_runs(
    runs_a: list[dict], runs_b: list[dict]
) -> list[tuple[dict, dict]]:
    """Pair runs by (goal_id, seed)."""
    index_b: dict[tuple[str, int], dict] = {}
    for run in runs_b:
        key: tuple[str, int] = (run["goal_id"], int(run["seed"]))
        if key in index_b:
            raise ValueError(
                f"Duplicate (goal_id, seed) in runs_b: "
                f"goal_id={key[0]!r}, seed={key[1]}"
            )
        index_b[key] = run

    pairs: list[tuple[dict, dict]] = []
    for run in runs_a:
        key = (run["goal_id"], int(run["seed"]))
        if key not in index_b:
            raise ValueError(
                f"No matching run in runs_b for "
                f"goal_id={key[0]!r}, seed={key[1]}"
            )
        pairs.append((run, index_b[key]))

    if len(pairs) != len(index_b):
        raise ValueError(
            f"runs_b has {len(index_b)} unique (goal_id, seed) pairs but "
            f"only {len(pairs)} could be matched from runs_a. "
            "Both lists must cover the same set of instances."
        )

    return pairs


def _wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p_hat = k / n
    z2_n = z * z / n
    denom = 1.0 + z2_n
    center = (p_hat + z2_n / 2.0) / denom
    margin = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2_n / (4.0 * n)) / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_cfr_by_condition(runs: list[dict]) -> dict[str, dict]:
    """Compute CFR for each condition found in *runs*.

Uses detect_cascade() to determine per-run cascade failure status,
then computes the proportion per condition with Wilson score 95% CIs."""
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["condition"], []).append(run)

    result: dict[str, dict] = {}
    for cond, cond_runs in grouped.items():
        n = len(cond_runs)
        k = sum(1 for r in cond_runs if detect_cascade(r))
        cfr = k / n if n > 0 else 0.0
        ci_low, ci_high = _wilson_ci(k, n)
        result[cond] = {
            "cfr": cfr,
            "n": n,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return result


def mcnemar_paired(
    runs_a: list[dict],
    runs_b: list[dict],
    outcome: str,
) -> dict:
    """McNemar's exact test on a binary *outcome* field for paired runs.

Pairs runs by (goal_id, seed). The outcome field must be 0/1 or bool.
Uses scipy.stats.binomtest for exact two-sided inference."""
    pairs = _pair_runs(runs_a, runs_b)

    n_A1_B0 = 0  # A positive, B negative
    n_A0_B1 = 0  # A negative, B positive

    for run_a, run_b in pairs:
        val_a = int(bool(_get_field(run_a, outcome)))
        val_b = int(bool(_get_field(run_b, outcome)))
        if val_a == 1 and val_b == 0:
            n_A1_B0 += 1
        elif val_a == 0 and val_b == 1:
            n_A0_B1 += 1

    n_discordant = n_A1_B0 + n_A0_B1
    n_total = len(pairs)

    if n_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n": n_total,
            "discordant_pairs": (0, 0),
        }

    # Exact two-sided McNemar via Binomial test.
    # Under H0, each discordant pair is equally likely to be (A+,B-) or (A-,B+).
    bt = binomtest(
        min(n_A1_B0, n_A0_B1), n_discordant, 0.5, alternative="two-sided"
    )

    # Chi-squared statistic with continuity correction (Edwards, 1948).
    statistic = (abs(n_A1_B0 - n_A0_B1) - 1.0) ** 2 / n_discordant

    return {
        "statistic": float(statistic),
        "p_value": float(bt.pvalue),
        "n": n_total,
        "discordant_pairs": (n_A1_B0, n_A0_B1),
    }


def wilcoxon_paired(
    runs_a: list[dict],
    runs_b: list[dict],
    outcome: str,
) -> dict:
    """Wilcoxon signed-rank test on a continuous *outcome* field for paired runs.

Pairs runs by (goal_id, seed). Supports dot notation for nested fields
(e.g. "tokens.total", "latency_s")."""
    pairs = _pair_runs(runs_a, runs_b)

    diffs = [
        float(_get_field(run_a, outcome)) - float(_get_field(run_b, outcome))
        for run_a, run_b in pairs
    ]

    median_diff = float(np.median(diffs))

    # scipy.stats.wilcoxon ignores zero differences by default (zero_method='wilcox').
    non_zero = [d for d in diffs if d != 0.0]

    if len(non_zero) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n": len(pairs),
            "median_diff": median_diff,
        }

    stat, p_val = scipy_wilcoxon(non_zero, alternative="two-sided")

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n": len(pairs),
        "median_diff": median_diff,
    }


def bootstrap_ci_paired(
    runs_a: list[dict],
    runs_b: list[dict],
    outcome: str,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap 95% CI for the mean difference (A − B) on *outcome*.

Resamples pairs with replacement and computes the empirical 2.5th and
97.5th percentiles of the bootstrap distribution of mean differences."""
    pairs = _pair_runs(runs_a, runs_b)

    diffs = np.array([
        float(_get_field(ra, outcome)) - float(_get_field(rb, outcome))
        for ra, rb in pairs
    ])

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))

    return (lower, upper)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

h = 2·arcsin(√p1) − 2·arcsin(√p2)

Conventionally: |h| < 0.2 small, 0.2–0.5 medium, > 0.8 large."""
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def cliffs_delta(values_a: list, values_b: list) -> float:
    """Cliff's delta non-parametric effect size for two distributions.

delta = (#{a > b} − #{a < b}) / (|A| · |B|)

Conventionally: |δ| < 0.147 negligible, 0.147–0.33 small,
0.33–0.474 medium, > 0.474 large."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    dominance = np.sign(a[:, None] - b[None, :])
    return float(dominance.sum() / (len(a) * len(b)))


def _load_runs(runs_path: str) -> list[dict]:
    """
    Load all non-empty JSON lines from *runs_path*, filtering to the
    simplified 2-condition protocol.

    Only conditions in {FULL_FRAMEWORK, BASELINE_LANGGRAPH} are retained.
    """
    runs: list[dict] = []
    with Path(runs_path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                run = json.loads(line)
                if run.get("condition") in _KEEP_CONDITIONS:
                    runs.append(run)
    return runs


def _group_by_condition(runs: list[dict]) -> dict[str, list[dict]]:
    """Group *runs* by their 'condition' field."""
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["condition"], []).append(run)
    return groups


def _run_primary_tests(
    full_runs: list[dict],
    by_condition: dict[str, list[dict]],
) -> dict:
    """McNemar on cascade_failure and success; Wilcoxon on total tokens vs ablation."""
    tests: dict[str, dict] = {}

    for ablation in _ABLATIONS:
        ablation_runs = by_condition.get(ablation, [])
        if not ablation_runs or not full_runs:
            continue

        cfr_mc = mcnemar_paired(full_runs, ablation_runs, "cascade_failure")
        cfr_ci = bootstrap_ci_paired(full_runs, ablation_runs, "cascade_failure")
        p1_cfr = sum(int(bool(r["cascade_failure"])) for r in full_runs) / len(full_runs)
        p2_cfr = sum(int(bool(r["cascade_failure"])) for r in ablation_runs) / len(ablation_runs)
        tests[f"{ablation}_cfr"] = {
            **cfr_mc,
            "significant": cfr_mc["p_value"] < ALPHA,
            "effect_size": cohens_h(p1_cfr, p2_cfr),
            "ci": cfr_ci,
        }

        suc_mc = mcnemar_paired(full_runs, ablation_runs, "success")
        suc_ci = bootstrap_ci_paired(full_runs, ablation_runs, "success")
        p1_suc = sum(int(bool(r["success"])) for r in full_runs) / len(full_runs)
        p2_suc = sum(int(bool(r["success"])) for r in ablation_runs) / len(ablation_runs)
        tests[f"{ablation}_success"] = {
            **suc_mc,
            "significant": suc_mc["p_value"] < ALPHA,
            "effect_size": cohens_h(p1_suc, p2_suc),
            "ci": suc_ci,
        }

        tok_wc = wilcoxon_paired(full_runs, ablation_runs, "tokens.total")
        tok_ci = bootstrap_ci_paired(full_runs, ablation_runs, "tokens.total")
        vals_a = [float(_get_field(r, "tokens.total")) for r in full_runs]
        vals_b = [float(_get_field(r, "tokens.total")) for r in ablation_runs]
        tests[f"{ablation}_tokens"] = {
            **tok_wc,
            "significant": tok_wc["p_value"] < ALPHA,
            "effect_size": cliffs_delta(vals_a, vals_b),
            "ci": tok_ci,
        }

    return tests


def primary_analysis(runs_path: str) -> dict:
    """Load runs.jsonl and return CFR by condition plus paired primary tests."""
    runs = _load_runs(runs_path)
    by_condition = _group_by_condition(runs)
    full_runs = by_condition.get("FULL_FRAMEWORK", [])

    return {
        "cfr_by_condition": compute_cfr_by_condition(runs),
        "alpha": ALPHA,
        "tests": _run_primary_tests(full_runs, by_condition),
    }
