"""Post-hoc Cascade Failure Rate (CFR) detection from run logs."""
from __future__ import annotations

from typing import Any


def detect_cascade(run_log: dict[str, Any]) -> bool:
    """Return True iff *run_log* represents a cascade failure."""
    ground_truth_match = run_log.get("ground_truth_match")

    if ground_truth_match is True or ground_truth_match is None:
        return False

    # Adversarial injection was scheduled but never reached step 2 (DEV-014).
    # Corruption never propagated, so this cannot be a cascade failure.
    if run_log.get("adversarial_injection_status") == "scheduled_but_not_reached":
        return False

    per_step = run_log.get("per_step", [])
    return any(
        step.get("would_have_been_invalid") is True
        and step.get("downstream_consumed") is True
        for step in per_step
    )
