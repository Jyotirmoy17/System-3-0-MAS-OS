"""Unit tests for mas/eval/cascade_detector.py::detect_cascade()."""
from __future__ import annotations

from mas.eval.cascade_detector import detect_cascade


def _step(
    step_id: int = 1,
    would_have_been_invalid: bool = False,
    downstream_consumed: bool = True,
) -> dict:
    return {
        "step_id":               step_id,
        "agent_id":              f"agent_{step_id}",
        "tier":                  2,
        "schema_valid":          not would_have_been_invalid,
        "would_have_been_invalid": would_have_been_invalid,
        "downstream_consumed":   downstream_consumed,
        "tokens_used":           100,
        "latency_ms":            500,
    }


def test_cascade_positive():
    """Step 2 has invalid output that is consumed downstream; final output fails rubric."""
    run_log = {
        "per_step": [
            _step(1, would_have_been_invalid=False, downstream_consumed=True),
            _step(2, would_have_been_invalid=True,  downstream_consumed=True),
            _step(3, would_have_been_invalid=False, downstream_consumed=False),
        ],
        "ground_truth_match": False,
    }
    assert detect_cascade(run_log) is True


def test_cascade_no_invalid():
    """All outputs are schema-valid.  No cascade regardless of rubric outcome."""
    run_log = {
        "per_step": [
            _step(1, would_have_been_invalid=False, downstream_consumed=True),
            _step(2, would_have_been_invalid=False, downstream_consumed=False),
        ],
        "ground_truth_match": False,
    }
    assert detect_cascade(run_log) is False


def test_cascade_not_consumed():
    """Invalid output is the final step; no downstream consumer → no cascade."""
    run_log = {
        "per_step": [
            _step(1, would_have_been_invalid=True, downstream_consumed=False),
        ],
        "ground_truth_match": False,
    }
    assert detect_cascade(run_log) is False


def test_cascade_correct_output():
    """Schema violation was consumed by downstream, but the final answer is correct."""
    run_log = {
        "per_step": [
            _step(1, would_have_been_invalid=True, downstream_consumed=True),
            _step(2, would_have_been_invalid=False, downstream_consumed=False),
        ],
        "ground_truth_match": True,
    }
    assert detect_cascade(run_log) is False


def test_empty_run_log():
    """Empty step list must return False without raising."""
    run_log = {"per_step": [], "ground_truth_match": False}
    assert detect_cascade(run_log) is False


def test_unscored_run_not_cascade():
    """When the eval harness has not yet scored the run, cascade is False."""
    run_log = {
        "per_step": [
            _step(1, would_have_been_invalid=True, downstream_consumed=True),
        ],
        "ground_truth_match": None,
    }
    assert detect_cascade(run_log) is False
