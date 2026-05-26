"""Figure 3: Per-step pipeline survival curve (Theorem 3.2 visualization)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mas.eval.figures import (
    FIXED_CREATION_DATE,
    STYLE,
)
from mas.eval.statistics import _load_runs


def _wilson(k: int, n: int) -> tuple[float, float]:
    """Return (lo, hi) Wilson score 95% CI for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compute_survival(
    runs: list[dict],
    condition: str,
    use_schema_valid: bool = False,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Compute cumulative survival S(k) for k = 1 … max_steps.

S(k) = fraction of runs (with ≥ k steps) where all steps 1 … k are
schema-valid.

For FULL_FRAMEWORK (use_schema_valid=True) we check
``per_step[i]["schema_valid"]``.  For BASELINE we check that
``per_step[i]["would_have_been_invalid"]`` is False (raw-output
validity)."""
    cond_runs = [r for r in runs if r["condition"] == condition]

    max_k = max(
        (len(r.get("per_step", [])) for r in cond_runs),
        default=0,
    )

    steps:       list[int]   = []
    survival:    list[float] = []
    ci_lo:       list[float] = []
    ci_hi:       list[float] = []
    n_eligible:  list[int]   = []

    for k in range(1, max_k + 1):
        eligible = [r for r in cond_runs if len(r.get("per_step", [])) >= k]
        n = len(eligible)
        if n == 0:
            break

        n_survived = 0
        for r in eligible:
            ps = r["per_step"][:k]
            if use_schema_valid:
                ok = all(s.get("schema_valid", True) for s in ps)
            else:
                ok = all(not s.get("would_have_been_invalid", False) for s in ps)
            if ok:
                n_survived += 1

        lo, hi = _wilson(n_survived, n)
        steps.append(k)
        survival.append(n_survived / n)
        ci_lo.append(lo)
        ci_hi.append(hi)
        n_eligible.append(n)

    return steps, survival, ci_lo, ci_hi, n_eligible


def fit_epsilon(
    steps: list[int],
    survival: list[float],
) -> Optional[float]:
    """Fit per-step error rate ε from survival S(k) ≈ (1-ε)^k.

Uses log-linear regression: log(S(k)) = k * log(1-ε).
Drops points where S(k) ≤ 0 (can't take log)."""
    xs = []
    ys = []
    for k, s in zip(steps, survival):
        if s > 0:
            xs.append(k)
            ys.append(math.log(s))

    if len(xs) < 2:
        return None

    # Check for zero variance in y (all survival = 1.0)
    if len(set(round(y, 12) for y in ys)) == 1:
        return None

    result = stats.linregress(xs, ys)
    slope = float(result.slope)  # type: ignore[attr-defined]
    # slope = log(1-ε) → ε = 1 - exp(slope)
    eps = 1.0 - math.exp(slope)
    return float(np.clip(eps, 0.0, 1.0))


# Hardcoded x for System 0/3 k=1 marker (left of baseline at x=1.0).
_SYSTEM03_K1_X: float = 0.85

def generate(
    runs_path: str,
    output_path: str = "figures/per_step_survival.pdf",
) -> Optional[dict]:
    """Generate the per-step survival figure (Theorem 3.2 visualization).

Saves PDF and PNG.  Returns a dict with the fitted epsilon value so
tests can assert on it without parsing the figure text."""
    runs = _load_runs(runs_path)

    # ── Survival curves ───────────────────────────────────────────────────────
    full_steps, full_surv, full_lo, full_hi, full_n = compute_survival(
        runs, "FULL_FRAMEWORK", use_schema_valid=True
    )
    bl_steps, bl_surv, bl_lo, bl_hi, bl_n = compute_survival(
        runs, "BASELINE_LANGGRAPH", use_schema_valid=False
    )
    # ── Fit ε from BASELINE ────────────────────────────────────────────────────
    eps: Optional[float] = None
    if bl_steps:
        eps = fit_epsilon(bl_steps, bl_surv)

    # ── Figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # ── 1. Baseline (background) ────────────────────────────────────────────────
    baseline_x: list[float] = []
    if bl_steps:
        baseline_x = [float(k) for k in bl_steps]
        baseline_y = bl_surv
        ax.fill_between(
            baseline_x,
            [max(0.0, v) for v in bl_lo],
            [min(1.0, v) for v in bl_hi],
            alpha=0.18,
            color="tab:blue",
            zorder=1,
        )
        ax.plot(
            baseline_x,
            baseline_y,
            color="tab:blue",
            marker="o",
            markersize=8,
            linewidth=2,
            zorder=5,
            label="Baseline",
        )

    # ── 2. System 0/3 — blue star at x=0.85 (baseline stays at x=1.0) ───────
    if full_steps:
        full_y0 = full_surv[0]
        ax.scatter(
            [_SYSTEM03_K1_X],
            [full_y0],
            color="blue",
            marker="*",
            s=300,
            zorder=10,
            label="System 0/3",
        )
        xmax = float(max(baseline_x) if baseline_x else full_steps[-1])
        ax.hlines(
            y=full_y0,
            xmin=_SYSTEM03_K1_X,
            xmax=xmax,
            colors="blue",
            linestyles="dotted",
            alpha=0.5,
            zorder=4,
        )
        # Synthetic / multi-step FULL: steps k≥2 at integer k (k=1 only at 0.85).
        if len(full_steps) > 1:
            ax.fill_between(
                full_steps[1:],
                [max(0.0, v) for v in full_lo[1:]],
                [min(1.0, v) for v in full_hi[1:]],
                alpha=0.18,
                color="blue",
                zorder=1,
            )
            ax.plot(
                full_steps[1:],
                full_surv[1:],
                color="blue",
                linewidth=2,
                marker="*",
                markersize=10,
                zorder=6,
            )

    # Theoretical (1-ε)^k decay overlay (legend only; no text boxes)
    if eps is not None and bl_steps:
        k_max = bl_steps[-1]
        k_dense = np.linspace(1, k_max, 200)
        theory = (1 - eps) ** k_dense
        ax.plot(
            k_dense, theory,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Predicted $(1-\\varepsilon)^k$  (fitted $\\varepsilon={eps:.3f}$)",
            zorder=2,
        )

    # ── Axes ─────────────────────────────────────────────────────────────────
    all_steps = sorted(set(full_steps) | set(bl_steps))
    ax.set_xlim(0.5, 5.5)
    if all_steps:
        ax.set_xticks(all_steps)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Pipeline step  $k$", fontsize=11)
    ax.set_ylabel("$P$(pipeline valid through step $k$)", fontsize=11)
    ax.set_title(
        "Per-Step Pipeline Survival Curve\n"
        "Cumulative schema validity through step $k$ (among runs that reach $k$)",
        fontsize=12,
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}" if 0 <= v <= 1 else "")
    )
    ax.legend(fontsize=9.5, loc="lower left")

    # ── Save ─────────────────────────────────────────────────────────────────
    pdf_path = Path(output_path)
    png_path = pdf_path.with_suffix(".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight",
                metadata={"CreationDate": FIXED_CREATION_DATE})
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"fitted_epsilon": eps}
