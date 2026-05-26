"""Unit tests for `mas/eval/composite_metrics.py`."""
from __future__ import annotations

import math
from typing import Any

from mas.eval.composite_metrics import (
    DEFAULT_JPI_WEIGHTS,
    accuracy,
    bootstrap_ci,
    budget_efficiency,
    cfr,
    jpi,
    metrics_for_condition,
    metrics_table,
    schema_compliance,
    success_rate,
)


# Fixture helpers — minimal RunLog dicts that match the schema fields
# composite_metrics.py reads.  Other fields (run_id, started_at, ...) are
# not required by the metric functions and are omitted for brevity.

def _run(
    *,
    success:           int  = 1,
    cascade_failure:   int  = 0,
    crash_reason:      str | None = None,
    ground_truth_match: bool | None = True,
    total_tokens:      int  = 1000,
    schema_valid_steps: tuple[bool, ...] = (True,),
) -> dict[str, Any]:
    """Build a minimal RunLog-shaped dict for metric tests."""
    return {
        "success":            success,
        "cascade_failure":    cascade_failure,
        "crash_reason":       crash_reason,
        "ground_truth_match": ground_truth_match,
        "tokens":             {"total": total_tokens, "slm": 0, "llm": total_tokens},
        "per_step":           [{"schema_valid": v} for v in schema_valid_steps],
    }


def test_accuracy_all_pass() -> None:
    runs = [_run(ground_truth_match=True) for _ in range(4)]
    assert accuracy(runs) == 1.0


def test_accuracy_all_fail() -> None:
    runs = [_run(ground_truth_match=False) for _ in range(4)]
    assert accuracy(runs) == 0.0


def test_accuracy_mixed() -> None:
    runs = [
        _run(ground_truth_match=True),
        _run(ground_truth_match=True),
        _run(ground_truth_match=False),
        _run(ground_truth_match=False),
    ]
    assert accuracy(runs) == 0.5


def test_accuracy_excludes_ungraded_none_values() -> None:
    """
    Runs with ground_truth_match=None (ungraded) are excluded from the
    accuracy denominator so a half-graded batch doesn't artificially
    deflate accuracy.
    """
    runs = [
        _run(ground_truth_match=True),
        _run(ground_truth_match=False),
        _run(ground_truth_match=None),  # excluded
        _run(ground_truth_match=None),  # excluded
    ]
    assert accuracy(runs) == 0.5


def test_accuracy_excludes_crashed_runs() -> None:
    """Crashed runs are excluded — they are deployment failures, not
    accuracy datapoints.  Without this, a single crash inflates failure
    rate even though the underlying agent never produced an answer."""
    runs = [
        _run(ground_truth_match=True),
        _run(ground_truth_match=False, crash_reason="RuntimeError: boom"),
    ]
    assert accuracy(runs) == 1.0


def test_accuracy_empty_returns_zero() -> None:
    assert accuracy([]) == 0.0


def test_v_s_all_steps_valid() -> None:
    runs = [_run(schema_valid_steps=(True, True, True)) for _ in range(3)]
    assert schema_compliance(runs) == 1.0


def test_v_s_any_invalid_step_drops_run() -> None:
    """V_s applies the conjunction across per_step: a single invalid
    step makes the whole run non-compliant."""
    runs = [
        _run(schema_valid_steps=(True, True, True)),       # compliant
        _run(schema_valid_steps=(True, False, True)),      # NOT compliant
        _run(schema_valid_steps=(True, True, True)),       # compliant
    ]
    assert schema_compliance(runs) == 2.0 / 3.0


def test_v_s_empty_per_step_is_noncompliant() -> None:
    """A run with no per_step (typically a crash before validate ran)
    cannot be schema-compliant by definition."""
    runs = [{"per_step": []}]
    assert schema_compliance(runs) == 0.0


def test_v_s_empty_runs_returns_zero() -> None:
    assert schema_compliance([]) == 0.0


def test_e_b_simple_calculation() -> None:
    """E_b = (successes / total_tokens) * 1000."""
    runs = [_run(success=1, total_tokens=1000) for _ in range(3)]
    # 3 successes / 3000 tokens * 1000 = 1.0
    assert budget_efficiency(runs) == 1.0


def test_e_b_with_failures() -> None:
    runs = [
        _run(success=1, total_tokens=500),
        _run(success=0, total_tokens=500),
        _run(success=1, total_tokens=500),
        _run(success=0, total_tokens=500),
    ]
    # 2 successes / 2000 tokens * 1000 = 1.0
    assert budget_efficiency(runs) == 1.0


def test_e_b_zero_tokens_returns_zero() -> None:
    """No tokens spent ⇒ E_b = 0 by convention (avoid division by zero
    and the unintuitive +infinity that 'infinity successes per token'
    would produce)."""
    runs = [_run(success=1, total_tokens=0)]
    assert budget_efficiency(runs) == 0.0


def test_e_b_zero_successes_returns_zero() -> None:
    runs = [_run(success=0, total_tokens=1000) for _ in range(5)]
    assert budget_efficiency(runs) == 0.0


def test_jpi_default_weights() -> None:
    """JPI = 0.5*Acc + 0.3*V_s + 0.2*log(E_b) with PoC defaults."""
    runs = [
        _run(success=1, ground_truth_match=True,  total_tokens=1000,
             schema_valid_steps=(True, True)),
        _run(success=1, ground_truth_match=True,  total_tokens=1000,
             schema_valid_steps=(True, True)),
    ]
    # Acc = 1.0, V_s = 1.0, E_b = 2/2000*1000 = 1.0, log(1.0) = 0
    # JPI = 0.5*1.0 + 0.3*1.0 + 0.2*0 = 0.8
    assert math.isclose(jpi(runs), 0.8, rel_tol=1e-9)


def test_jpi_zero_eb_uses_log_floor() -> None:
    """E_b = 0 ⇒ log clamp at log(1e-9) ≈ -20.72.  JPI must not be -inf."""
    runs = [_run(success=0, ground_truth_match=False, total_tokens=1000,
                 schema_valid_steps=(True,)) for _ in range(3)]
    j = jpi(runs)
    expected = 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * math.log(1e-9)
    assert math.isclose(j, expected, rel_tol=1e-9)
    assert math.isfinite(j)


def test_jpi_collapses_when_no_successes() -> None:
    """
    A condition with high V_s but zero accuracy and zero successes must
    rank below a condition with positive accuracy.  This is the
    formal structure that distinguishes low-accuracy early halts from
    successful multi-step completion (FULL_FRAMEWORK).
    """
    halts_at_step1 = [
        _run(success=0, ground_truth_match=False, total_tokens=200,
             schema_valid_steps=(True,))
        for _ in range(10)
    ]
    solves = [
        _run(success=1, ground_truth_match=True, total_tokens=2000,
             schema_valid_steps=(True, True, True))
        for _ in range(10)
    ]
    assert jpi(solves) > jpi(halts_at_step1)


def test_jpi_custom_weights() -> None:
    """User-specified weights override the defaults."""
    runs = [
        _run(success=1, ground_truth_match=True, total_tokens=1000,
             schema_valid_steps=(True,)),
    ]
    # All-Accuracy weighting
    assert math.isclose(jpi(runs, weights=(1.0, 0.0, 0.0)), 1.0, rel_tol=1e-9)
    # All-V_s weighting
    assert math.isclose(jpi(runs, weights=(0.0, 1.0, 0.0)), 1.0, rel_tol=1e-9)


def test_jpi_default_weights_constant() -> None:
    """The PoC default weights are (0.5, 0.3, 0.2) and must stay so to
    match `poc/harness.py:74-81`."""
    assert DEFAULT_JPI_WEIGHTS == (0.50, 0.30, 0.20)


def test_cfr_counts_cascade_failure_field() -> None:
    runs = [
        _run(cascade_failure=0),
        _run(cascade_failure=1),
        _run(cascade_failure=1),
        _run(cascade_failure=0),
    ]
    assert cfr(runs) == 0.5


def test_cfr_handles_missing_field_as_zero() -> None:
    runs = [{"per_step": []}, {"per_step": [], "cascade_failure": 1}]
    assert cfr(runs) == 0.5


def test_cfr_handles_none_value_as_zero() -> None:
    """`cascade_failure: None` means 'not yet computed' — treat as 0."""
    runs = [{"cascade_failure": None}, {"cascade_failure": 1}]
    assert cfr(runs) == 0.5


def test_cfr_empty_returns_zero() -> None:
    assert cfr([]) == 0.0


def test_success_rate_basic() -> None:
    runs = [_run(success=1), _run(success=0), _run(success=1)]
    assert math.isclose(success_rate(runs), 2.0 / 3.0, rel_tol=1e-9)


def test_bootstrap_ci_constant_metric_has_zero_width() -> None:
    """A metric that is constant across all runs has CI = (point, point)."""
    runs = [_run(ground_truth_match=True) for _ in range(20)]
    lo, hi = bootstrap_ci(runs, accuracy, n=200, seed=0)
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_includes_point_estimate() -> None:
    """Point estimate must lie within (or at) the CI bounds for a
    well-behaved metric on non-pathological data."""
    runs = [_run(ground_truth_match=True if i % 2 == 0 else False)
            for i in range(50)]
    point = accuracy(runs)
    lo, hi = bootstrap_ci(runs, accuracy, n=500, seed=42)
    assert lo <= point <= hi


def test_bootstrap_ci_seed_reproducibility() -> None:
    """Identical seeds yield identical CIs."""
    runs = [_run(ground_truth_match=(i < 7)) for i in range(20)]
    ci_a = bootstrap_ci(runs, accuracy, n=500, seed=123)
    ci_b = bootstrap_ci(runs, accuracy, n=500, seed=123)
    assert ci_a == ci_b


def test_bootstrap_ci_empty_input_returns_point_pair() -> None:
    lo, hi = bootstrap_ci([], accuracy, n=100, seed=0)
    assert lo == 0.0 and hi == 0.0


def test_metrics_for_condition_populates_all_fields() -> None:
    runs = [
        _run(success=1, ground_truth_match=True,  total_tokens=1000,
             schema_valid_steps=(True, True), cascade_failure=0),
        _run(success=1, ground_truth_match=True,  total_tokens=1000,
             schema_valid_steps=(True, True), cascade_failure=0),
    ]
    m = metrics_for_condition("FULL_FRAMEWORK", runs, bootstrap_n=200)
    assert m.condition  == "FULL_FRAMEWORK"
    assert m.n_runs     == 2
    assert m.accuracy   == 1.0
    assert m.schema_v_s == 1.0
    assert m.cfr        == 0.0
    assert math.isclose(m.budget_e_b, 1.0, rel_tol=1e-9)
    # log(1.0) = 0
    assert math.isclose(m.log_e_b, 0.0, abs_tol=1e-9)


def test_metrics_table_preserves_condition_order() -> None:
    """metrics_table must preserve the dict insertion order of conditions."""
    runs_by_cond = {
        "FULL_FRAMEWORK":     [_run() for _ in range(2)],
        "BASELINE_LANGGRAPH": [_run() for _ in range(2)],
    }
    table = metrics_table(runs_by_cond, bootstrap_n=100)
    assert list(table.keys()) == [
        "FULL_FRAMEWORK", "BASELINE_LANGGRAPH",
    ]


def test_metrics_table_empty_condition() -> None:
    """An empty condition produces a ConditionMetrics with n_runs=0 and
    all metrics at their empty-input defaults."""
    table = metrics_table({"FULL_FRAMEWORK": []}, bootstrap_n=100)
    m = table["FULL_FRAMEWORK"]
    assert m.n_runs    == 0
    assert m.accuracy  == 0.0
    assert m.schema_v_s == 0.0
    assert m.cfr       == 0.0
    # log_e_b clamps at log(1e-9) ≈ -20.72
    assert m.log_e_b < -20.0
