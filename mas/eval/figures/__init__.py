"""Figure generation modules for the System-3/0 evaluation."""
from __future__ import annotations

import datetime


#: Apply with: plt.rcParams.update(STYLE)
STYLE: dict = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    "axes.axisbelow": True,
}


CONDITION_COLORS: dict[str, str] = {
    "FULL_FRAMEWORK":     "#0072B2",   # blue
    "BASELINE_LANGGRAPH": "#CC79A7",   # reddish-purple
}

#: B&W-distinguishable hatch patterns (one per condition).
CONDITION_HATCHES: dict[str, str] = {
    "FULL_FRAMEWORK":     "",
    "BASELINE_LANGGRAPH": "xxxx",
}

#: Short display names for axis labels and legends.
CONDITION_SHORT: dict[str, str] = {
    "FULL_FRAMEWORK":     "System 0/3",
    "BASELINE_LANGGRAPH": "Baseline",
}


GOAL_TYPE_MARKERS: dict[str, str] = {
    "research_synthesis":      "o",   # circle
    "financial_analysis":      "s",   # square
    "sequential_reasoning":    "^",   # triangle-up
    "code_analysis":           "D",   # diamond
    "adversarial_reliability": "P",   # plus (filled)
}

GOAL_TYPE_SHORT: dict[str, str] = {
    "research_synthesis":      "Research",
    "financial_analysis":      "Financial",
    "sequential_reasoning":    "Sequential",
    "code_analysis":           "Code",
    "adversarial_reliability": "Adversarial",
}


FIXED_CREATION_DATE: datetime.datetime = datetime.datetime(
    2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
)
