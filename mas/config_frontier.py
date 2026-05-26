"""Configuration overrides for frontier (Claude Sonnet) replication runs."""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


#: Use claude-sonnet-4-5 for both SLM and LLM tiers.
#: Claude Sonnet is strong enough for both tiers; no model split required.
FRONTIER_MODEL_SLM = "claude-sonnet-4-5"
FRONTIER_MODEL_LLM = "claude-sonnet-4-5"


_INPUT_COST_PER_TOKEN  = 3.0  / 1_000_000   # $3.00 / MTok
_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000   # $15.00 / MTok

# Conservative per-run estimate (prompt + response, 4-step pipeline)
_ESTIMATED_TOKENS_PER_RUN = 4_000
_ESTIMATED_COST_PER_RUN_USD = (
    _ESTIMATED_TOKENS_PER_RUN * 0.6 * _INPUT_COST_PER_TOKEN   # ~60% input
    + _ESTIMATED_TOKENS_PER_RUN * 0.4 * _OUTPUT_COST_PER_TOKEN  # ~40% output
)

#: Maximum recommended frontier runs before cost warning escalates.
MAX_FRONTIER_RUNS = 150


FRONTIER_MODEL_OVERRIDES: dict = {
    "backend":  "anthropic",
    "slm":      FRONTIER_MODEL_SLM,
    "llm":      FRONTIER_MODEL_LLM,
    "tau":      0.72,
}


def warn_cost(n_instances: int = 150) -> None:
    """
    Print a cost warning before starting a frontier batch.

    Args:
        n_instances: Number of goal instances to run.
    """
    estimated = n_instances * _ESTIMATED_COST_PER_RUN_USD
    key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))

    log.warning(
        "FRONTIER REPLICATION — %d instances × ~$%.3f/run ≈ $%.2f USD total",
        n_instances, _ESTIMATED_COST_PER_RUN_USD, estimated,
    )
    if not key_present:
        log.error(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Frontier runs will fail immediately."
        )
    print(
        f"\n{'='*60}\n"
        f"  FRONTIER REPLICATION COST WARNING\n"
        f"  Model:     {FRONTIER_MODEL_LLM}\n"
        f"  Instances: {n_instances}\n"
        f"  Estimated: ~${estimated:.2f} USD ({n_instances} runs × "
        f"~${_ESTIMATED_COST_PER_RUN_USD:.3f}/run)\n"
        f"  API key:   {'SET' if key_present else 'NOT SET — will fail'}\n"
        f"{'='*60}\n"
    )


def estimated_cost(n_instances: int) -> float:
    """
    Return the estimated USD cost for *n_instances* frontier runs.

    Args:
        n_instances: Number of goal instances.

    Returns:
        Estimated total cost in USD.
    """
    return n_instances * _ESTIMATED_COST_PER_RUN_USD
