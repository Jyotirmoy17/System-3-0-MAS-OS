"""Headline figure: Cascade Failure Rate by condition with 95% Wilson score."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from mas.eval.statistics import (
    _group_by_condition,
    _load_runs,
    compute_cfr_by_condition,
    mcnemar_paired,
)


#: Display order for conditions on the x-axis (simplified 2-condition protocol).
_CONDITION_ORDER: tuple[str, ...] = (
    "FULL_FRAMEWORK",
    "BASELINE_LANGGRAPH",
)

#: Short x-axis labels.
_LABELS: dict[str, str] = {
    "FULL_FRAMEWORK":     "System\n0/3",
    "BASELINE_LANGGRAPH": "Baseline",
}

#: Wong (2011) color-blind safe 8-color palette (hex).
_PALETTE: dict[str, str] = {
    "FULL_FRAMEWORK":     "#0072B2",  # blue
    "BASELINE_LANGGRAPH": "#CC79A7",  # reddish-purple
}

#: B&W-distinguishable hatch patterns (one per condition, same order).
_HATCHES: dict[str, str] = {
    "FULL_FRAMEWORK":     "",
    "BASELINE_LANGGRAPH": "xxxx",
}

#: Expected number of goal instances per condition.
_EXPECTED_N: int = 150


def _sig_stars(p_value: float) -> str:
    """Return significance stars for a p-value.

Uses unconventional thresholds because the correction already
accounts for the family-wise error rate:
  p < 0.001  → "***"
  p < 0.01   → "**"
  p < 0.05   → "*"
  otherwise  → ""  (no annotation)"""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _draw_significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y_top: float,
    label: str,
    bracket_height: float = 0.012,
    fontsize: float = 11.0,
) -> None:
    """Draw a square bracket between x1 and x2 at height y_top, with *label* centred."""
    y_arm = y_top - bracket_height
    ax.plot(
        [x1, x1, x2, x2],
        [y_arm, y_top, y_top, y_arm],
        lw=1.0,
        color="black",
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y_top + bracket_height * 0.3,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


def generate(
    runs_path: str,
    output_path: str = "figures/cascade_bar.pdf",
) -> None:
    """Generate the headline CFR-by-condition bar chart.

Reads runs.jsonl, computes per-condition CFR with Wilson score 95% CIs,
runs a McNemar test (FULL vs baseline), and
renders the figure.  Saves both PDF (vector, lossless) and PNG (300 dpi
raster) at *output_path* and the same path with .png extension."""
    # ── Load + compute ───────────────────────────────────────────────────────
    runs = _load_runs(runs_path)
    by_condition = _group_by_condition(runs)
    cfr_data     = compute_cfr_by_condition(runs)

    full_runs = by_condition.get("FULL_FRAMEWORK", [])
    if not full_runs:
        raise ValueError("runs_path contains no FULL_FRAMEWORK runs.")

    # Run paired McNemar on cascade_failure: FULL vs BASELINE.
    p_values: dict[str, float] = {}
    baseline_runs = by_condition.get("BASELINE_LANGGRAPH", [])
    if baseline_runs:
        result = mcnemar_paired(full_runs, baseline_runs, "cascade_failure")
        p_values["BASELINE_LANGGRAPH"] = float(result["p_value"])

    # Determine n (number of runs per condition; use FULL as reference).
    n_per_condition = len(full_runs)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.subplots_adjust(bottom=0.18, top=0.88, left=0.12, right=0.97)

    x_positions = np.arange(len(_CONDITION_ORDER), dtype=float)
    bar_width   = 0.55

    cfr_vals = []
    ci_lows  = []
    ci_highs = []

    for cond in _CONDITION_ORDER:
        v = cfr_data.get(cond, {"cfr": 0.0, "ci_low": 0.0, "ci_high": 0.0})
        cfr_vals.append(v["cfr"] * 100.0)
        # Clamp to ≥ 0 — floating-point subtraction can yield tiny negatives
        # (e.g. -7e-16) when CFR=0, which matplotlib 3.10+ rejects.
        ci_lows.append(max(0.0, (v["cfr"] - v["ci_low"]) * 100.0))
        ci_highs.append(max(0.0, (v["ci_high"] - v["cfr"]) * 100.0))

    # ── Bars ─────────────────────────────────────────────────────────────────
    bars = ax.bar(
        x_positions,
        cfr_vals,
        width=bar_width,
        color=[_PALETTE[c] for c in _CONDITION_ORDER],
        hatch=[_HATCHES[c] for c in _CONDITION_ORDER],
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    # ── Error bars ────────────────────────────────────────────────────────────
    ax.errorbar(
        x_positions,
        cfr_vals,
        yerr=[ci_lows, ci_highs],
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=1.2,
        elinewidth=1.2,
        zorder=4,
    )

    # ── Bar value labels ─────────────────────────────────────────────────────
    for bar, v in zip(bars, cfr_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(ci_highs) * 0.15 + 0.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    # ── Significance bracket (FULL vs BASELINE) ──────────────────────────────
    full_idx = 0  # FULL_FRAMEWORK is always index 0
    y_max_data = max(v + e for v, e in zip(cfr_vals, ci_highs))
    # Reserve vertical space above the tallest bar.
    bracket_base = y_max_data + 4.0
    bracket_step = 8.0  # vertical spacing between stacked brackets

    ablation_indices = {"BASELINE_LANGGRAPH": 1}
    bracket_level = 0
    for abl, abl_idx in ablation_indices.items():
        p = p_values.get(abl)
        if p is None:
            continue
        stars = _sig_stars(p)
        if not stars:
            # Still annotate "ns" so the reader knows the test was run.
            label = f"ns  (p={p:.3f})"
        else:
            label = f"{stars}  (p={p:.2e})"
        y_bracket = bracket_base + bracket_level * bracket_step
        _draw_significance_bracket(
            ax,
            x_positions[full_idx],
            x_positions[abl_idx],
            y_bracket / 100.0 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else y_bracket,
            label,
            bracket_height=0.6,
            fontsize=8.5,
        )
        bracket_level += 1

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_LABELS[c] for c in _CONDITION_ORDER],
        fontsize=10,
    )
    ax.set_ylabel("Cascade Failure Rate (%)", fontsize=11)
    ax.set_ylim(0, bracket_base + bracket_level * bracket_step + 10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="both", labelsize=10)

    # Remove top and right spines (no chart-junk).
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── Title + caption ───────────────────────────────────────────────────────
    ax.set_title(
        "Cascade Failure Rate by Experimental Condition",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    fig.text(
        0.5,
        0.01,
        f"Error bars: Wilson score 95% CI.  "
        "Significance: McNemar exact test (α=0.05).  "
        f"n={n_per_condition} per condition.  "
        f"* p<0.05  ** p<0.01  *** p<0.001.",
        ha="center",
        fontsize=7.5,
        color="#444444",
        wrap=True,
    )

    # ── Legend (hatch key for B&W readers) ───────────────────────────────────
    legend_patches = [
        mpatches.Patch(
            facecolor=_PALETTE[c],
            hatch=_HATCHES[c],
            edgecolor="black",
            linewidth=0.8,
            label=_LABELS.get(c, c.replace("_", " ").title()).replace("\n", " "),
        )
        for c in _CONDITION_ORDER
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=8,
        loc="upper right",
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    pdf_path = Path(output_path)
    png_path = pdf_path.with_suffix(".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Fix the PDF creation date so that identical inputs produce byte-identical
    # output regardless of wall-clock time (reproducibility requirement).
    _fixed_date = datetime.datetime(2026, 1, 1, 0, 0, 0,
                                    tzinfo=datetime.timezone.utc)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight",
                metadata={"CreationDate": _fixed_date})
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
