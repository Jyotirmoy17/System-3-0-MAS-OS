"""Figure 4: Frontier-model vs local-model comparison (FULL_FRAMEWORK only)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mas.eval.figures import (
    FIXED_CREATION_DATE,
    STYLE,
)
from mas.eval.statistics import _load_runs
from mas.eval.cascade_detector import detect_cascade


#: Keywords (case-insensitive) in model_llm that identify frontier models.
_FRONTIER_KEYWORDS: tuple[str, ...] = ("claude", "gpt-4", "gpt4", "gemini")

_METRICS: list[str] = ["CFR", "Success", "Tokens\n(norm.)", "Latency\n(norm.)"]

_LOCAL_COLOR    = "#0072B2"   # blue (Wong 2011)
_FRONTIER_COLOR = "#E69F00"   # orange (Wong 2011)


def _is_frontier(run: dict) -> bool:
    """Return True if the run used a frontier (cloud) model."""
    model_llm: str = run.get("model_llm", "")
    return any(kw in model_llm.lower() for kw in _FRONTIER_KEYWORDS)


def _compute_cfr(runs: list[dict]) -> float:
    """Compute CFR = fraction of runs that are cascade failures."""
    if not runs:
        return 0.0
    return sum(1 for r in runs if detect_cascade(r)) / len(runs)


def _mean_metric(runs: list[dict], field_path: list[str]) -> float:
    """Return the mean of a nested field across runs."""
    if not runs:
        return 0.0
    vals = []
    for r in runs:
        node: object = r
        for key in field_path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if node is not None:
            try:
                vals.append(float(node))
            except (TypeError, ValueError):
                pass
    return float(np.mean(vals)) if vals else 0.0


def _gather_metrics(runs: list[dict]) -> dict[str, float]:
    """Compute the four comparison metrics for a set of runs."""
    return {
        "cfr":          _compute_cfr(runs),
        "success":      float(np.mean([r["success"] for r in runs])) if runs else 0.0,
        "mean_tokens":  _mean_metric(runs, ["tokens", "total"]),
        "mean_latency": _mean_metric(runs, ["latency_s"]),
    }


def _placeholder_figure(
    fig: "plt.Figure",
    ax: "plt.Axes",
    message: str = "Frontier replication not available.",
) -> None:
    """Replace axes content with a centred placeholder message."""
    ax.clear()
    ax.set_axis_off()
    ax.text(
        0.5, 0.5,
        message,
        ha="center", va="center",
        fontsize=13,
        color="#555555",
        style="italic",
        transform=ax.transAxes,
        wrap=True,
    )
    ax.set_title(
        "Frontier vs Local  (FULL_FRAMEWORK only)",
        fontsize=12, fontweight="bold",
    )


def generate(
    runs_path: str,
    output_path: str = "figures/frontier_vs_local.pdf",
) -> None:
    """Generate the frontier-vs-local comparison figure.

Filters runs to FULL_FRAMEWORK condition, then splits by model backend.
If no frontier runs exist, saves a placeholder figure noting that the
frontier replication data are not yet available.

Tokens and latency bars are normalized to the local baseline (local = 1.0).
CFR and success are shown as raw proportions."""
    runs = _load_runs(runs_path)

    # ── Filter to FULL_FRAMEWORK ─────────────────────────────────────────────
    full_runs = [r for r in runs if r["condition"] == "FULL_FRAMEWORK"]

    local_runs    = [r for r in full_runs if not _is_frontier(r)]
    frontier_runs = [r for r in full_runs if _is_frontier(r)]

    # ── Figure scaffold ──────────────────────────────────────────────────────
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    # ── Placeholder if frontier data absent ───────────────────────────────────
    if not frontier_runs:
        _placeholder_figure(
            fig, ax,
            message=(
                "Frontier replication not available.\n"
                "Re-run with --backend frontier after obtaining\n"
                "Anthropic API access."
            ),
        )
        _save(fig, output_path)
        return

    # ── Metrics ──────────────────────────────────────────────────────────────
    loc = _gather_metrics(local_runs)
    fro = _gather_metrics(frontier_runs)

    # Normalize tokens and latency to local baseline
    norm_tokens  = (fro["mean_tokens"]  / loc["mean_tokens"])  if loc["mean_tokens"]  else 1.0
    norm_latency = (fro["mean_latency"] / loc["mean_latency"]) if loc["mean_latency"] else 1.0

    local_vals    = [loc["cfr"],     loc["success"], 1.0,         1.0]
    frontier_vals = [fro["cfr"],     fro["success"], norm_tokens, norm_latency]

    # ── Grouped bars ─────────────────────────────────────────────────────────
    n_metrics = len(_METRICS)
    x = np.arange(n_metrics)
    width = 0.32

    bars_local = ax.bar(
        x - width / 2,
        local_vals,
        width=width,
        color=_LOCAL_COLOR,
        edgecolor="black",
        linewidth=0.7,
        label=f"Local  (phi3+llama8b)  n={len(local_runs)}",
        zorder=3,
    )
    bars_frontier = ax.bar(
        x + width / 2,
        frontier_vals,
        width=width,
        color=_FRONTIER_COLOR,
        edgecolor="black",
        linewidth=0.7,
        label=f"Frontier  (Claude Sonnet)  n={len(frontier_runs)}",
        zorder=3,
    )

    # Reference line at 1.0 for normalised metrics (last two bars)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.9,
               alpha=0.6, zorder=2)

    # Value labels
    for bar in list(bars_local) + list(bars_frontier):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            zorder=5,
        )

    # Shade last two groups to indicate "normalised" metrics
    for xi in [2, 3]:
        ax.axvspan(xi - 0.5, xi + 0.5, color="#f0f0f0", zorder=0)

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(_METRICS, fontsize=10)
    ax.set_ylabel("Proportion / Normalised ratio", fontsize=10)
    ax.set_title(
        "Frontier vs Local  —  FULL_FRAMEWORK condition\n"
        "Tokens and latency normalised to local baseline (local = 1.0)",
        fontsize=11,
    )
    ax.set_ylim(0, max(max(local_vals), max(frontier_vals)) * 1.18 + 0.05)
    ax.legend(fontsize=9, loc="upper right")

    # Annotation distinguishing normalised columns
    ax.text(
        2.5, ax.get_ylim()[1] * 0.95,
        "← proportions   |   normalised ratios →",
        ha="center",
        fontsize=8,
        color="#666666",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    _save(fig, output_path)


def _save(fig: "plt.Figure", output_path: str) -> None:
    """Save figure as PDF and PNG, then close it."""
    pdf_path = Path(output_path)
    png_path = pdf_path.with_suffix(".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight",
                metadata={"CreationDate": FIXED_CREATION_DATE})
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
