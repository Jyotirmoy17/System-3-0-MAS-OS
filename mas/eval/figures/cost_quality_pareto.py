"""Figure 2: Cost-Quality Pareto scatter."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from mas.eval.figures import (
    CONDITION_COLORS,
    CONDITION_SHORT,
    FIXED_CREATION_DATE,
    GOAL_TYPE_MARKERS,
    GOAL_TYPE_SHORT,
    STYLE,
)
from mas.eval.statistics import _load_runs


def _compute_pareto_frontier(
    points: list[dict],
) -> list[dict]:
    """Compute the upper-left Pareto frontier (minimize tokens, maximize success).

A point (x, y) is non-dominated iff no other point has both x' ≤ x
and y' ≥ y (strictly better in at least one dimension)."""
    sorted_pts = sorted(points, key=lambda p: p["mean_tokens"])
    frontier: list[dict] = []
    best_success = -math.inf
    for pt in sorted_pts:
        if pt["success_rate"] > best_success:
            best_success = pt["success_rate"]
            frontier.append(pt)
    return frontier


def generate(
    runs_path: str,
    output_path: str = "figures/cost_quality_pareto.pdf",
) -> None:
    """Generate the cost-quality Pareto scatter figure.

One marker per (condition, goal_type) cell (up to 20 points).
Color encodes condition; marker shape encodes goal type.  The Pareto
frontier is overlaid as a dashed black line connecting non-dominated
cells.  The x-axis uses a log scale for tokens."""
    runs = _load_runs(runs_path)

    # ── Aggregate per (condition, goal_type) cell ────────────────────────────
    cells: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"tokens": [], "success": []}
    )
    for run in runs:
        key = (run["condition"], run["goal_type"])
        cells[key]["tokens"].append(float(run["tokens"]["total"]))
        cells[key]["success"].append(int(bool(run["success"])))

    points: list[dict[str, Any]] = []
    for (cond, gtype), data in cells.items():
        if not data["tokens"]:
            continue
        points.append({
            "condition":    cond,
            "goal_type":    gtype,
            "mean_tokens":  float(np.mean(data["tokens"])),
            "success_rate": float(np.mean(data["success"])),
        })

    pareto = _compute_pareto_frontier(points)

    # ── Figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    fig.subplots_adjust(left=0.10, right=0.76, bottom=0.12, top=0.90)

    # One scatter call per (condition, goal_type) → one PathCollection each.
    for pt in points:
        color  = CONDITION_COLORS.get(pt["condition"], "#555555")
        marker = GOAL_TYPE_MARKERS.get(pt["goal_type"], "o")
        ax.scatter(
            pt["mean_tokens"],
            pt["success_rate"],
            color=color,
            marker=marker,
            s=110,
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
            label=f'{pt["condition"]}|{pt["goal_type"]}',
        )

    # ── Pareto frontier step-line ─────────────────────────────────────────────
    if len(pareto) >= 2:
        px = [p["mean_tokens"] for p in pareto]
        py = [p["success_rate"] for p in pareto]
        ax.step(px, py, where="post", color="black", linestyle="--",
                linewidth=1.5, zorder=3, label="_nolegend_")
        # Label end of frontier
        ax.annotate(
            "Pareto\nfrontier",
            xy=(px[-1], py[-1]),
            xytext=(px[-1] * 1.05, py[-1] - 0.06),
            fontsize=8,
            color="black",
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black"},
        )

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlabel("Mean Total Tokens per Run  (log scale)", fontsize=11)
    ax.set_ylabel("Execution Completion Rate (CR)", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}" if 0 <= v <= 1 else "")
    )
    ax.set_title(
        "Cost-Quality Pareto  —  mean tokens vs execution completion\n"
        "per (condition × goal type) cell",
        fontsize=12,
    )

    # ── Dual legend (right of axes) ───────────────────────────────────────────
    present_conditions = sorted(
        {p["condition"] for p in points},
        key=lambda c: list(CONDITION_COLORS).index(c)
        if c in CONDITION_COLORS else 99,
    )
    present_goal_types = sorted(
        {p["goal_type"] for p in points},
        key=lambda g: list(GOAL_TYPE_MARKERS).index(g)
        if g in GOAL_TYPE_MARKERS else 99,
    )

    cond_handles = [
        mlines.Line2D(
            [], [],
            marker="o",
            color="w",
            markerfacecolor=CONDITION_COLORS.get(c, "#555"),
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=10,
            label=CONDITION_SHORT.get(c, c),
        )
        for c in present_conditions
    ]
    gtype_handles = [
        mlines.Line2D(
            [], [],
            marker=GOAL_TYPE_MARKERS.get(g, "o"),
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=10,
            linestyle="None",
            label=GOAL_TYPE_SHORT.get(g, g),
        )
        for g in present_goal_types
    ]
    pareto_handle = mlines.Line2D(
        [], [], color="black", linestyle="--", linewidth=1.5,
        label="Pareto frontier",
    )

    leg1 = ax.legend(
        handles=cond_handles,
        title="Condition",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8.5,
        title_fontsize=9,
        framealpha=0.9,
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=gtype_handles + [pareto_handle],
        title="Goal Type",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.55),
        fontsize=8.5,
        title_fontsize=9,
        framealpha=0.9,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    pdf_path = Path(output_path)
    png_path = pdf_path.with_suffix(".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight",
                metadata={"CreationDate": FIXED_CREATION_DATE})
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
