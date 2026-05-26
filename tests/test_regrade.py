"""Unit tests for `mas/eval/regrade.py`."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from mas.eval.regrade import (
    RegradeStats,
    format_summary,
    regrade,
    result_to_dict,
)


# Minimal valid goal-suite schema: copy from goals/_schema.json so
# load_all_goals validates against the same rules production uses.
_REPO_ROOT       = Path(__file__).resolve().parent.parent
_REAL_SCHEMA     = _REPO_ROOT / "goals" / "_schema.json"


@pytest.fixture
def tmp_goals_dir(tmp_path: Path) -> Path:
    """
    Build a minimal goals/ tree with two adversarial_reliability goals,
    each with one semantic_judge check.  Returns the path.
    """
    goals_root = tmp_path / "goals"
    (goals_root / "adversarial_reliability").mkdir(parents=True)
    shutil.copy(_REAL_SCHEMA, goals_root / "_schema.json")

    def _write_goal(goal_id: str, expected_substring: str) -> None:
        spec: dict[str, Any] = {
            "id":                  goal_id,
            "goal_type":           "adversarial_reliability",
            "pipeline_depth_k":    3,
            "goal_text":           "Test goal for regrade.",
            "inputs":              {},
            "ground_truth":        f"The answer should mention {expected_substring}.",
            "rubric": [
                {
                    "check_id":     "sem_main",
                    "check_text":   f"Answer must mention {expected_substring}.",
                    "check_type":   "semantic_judge",
                    "check_params": {
                        "field":    "answer",
                        "criteria": f"Identifies {expected_substring}.",
                    },
                },
            ],
            "corruption_pattern":  "drop_required_field",
        }
        path = goals_root / "adversarial_reliability" / f"{goal_id}.json"
        path.write_text(json.dumps(spec))

    _write_goal("adversarial_001", "Paris")
    _write_goal("adversarial_002", "London")
    return goals_root


@pytest.fixture
def tmp_runs_path(tmp_path: Path) -> Path:
    """Return a not-yet-created path to runs.jsonl in tmp_path."""
    return tmp_path / "runs.jsonl"


def _write_runs(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for run in runs:
            fh.write(json.dumps(run) + "\n")


def _stub_run(
    *,
    run_id:     str  = "r1",
    goal_id:    str  = "adversarial_001",
    condition:  str  = "FULL_FRAMEWORK",
    final_output: str = '{"answer": "Paris is the capital of France."}',
    crash_reason: str | None = None,
    grading_api_calls: int  = 0,
    rubric_scores: dict[str, bool] | None = None,
    ground_truth_match: bool | None = True,
    cascade_failure:  int   = 0,
    per_step:    list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct a minimal run dict with the fields regrade reads/writes."""
    return {
        "run_id":              run_id,
        "goal_id":             goal_id,
        "goal_type":           "adversarial_reliability",
        "condition":           condition,
        "seed":                42,
        "success":             1,
        "final_output":        final_output,
        "rubric_scores":       rubric_scores if rubric_scores is not None else {"sem_main": True},
        "ground_truth_match":  ground_truth_match,
        "cascade_failure":     cascade_failure,
        "crash_reason":        crash_reason,
        "grading_api_calls":   grading_api_calls,
        "grading_cost_usd":    0.0,
        "tokens":              {"slm": 0, "llm": 100, "total": 100},
        "per_step":            per_step or [
            {
                "step_id":                 1,
                "schema_valid":            True,
                "would_have_been_invalid": False,
                "downstream_consumed":     False,
            },
            {
                "step_id":                 2,
                "schema_valid":            False,
                "would_have_been_invalid": True,
                "downstream_consumed":     True,
            },
        ],
    }


class _StubJudge:
    """
    Toy JudgeProtocol implementation that returns True iff *answer* contains
    *trigger* (case-insensitive).  Tracks calls and (always-zero) cost so the
    regrade module's accounting can be verified.
    """
    DEFAULT_MODEL = "stub-model-v0"

    def __init__(self, trigger: str = "Paris") -> None:
        self._trigger          = trigger.lower()
        self._model            = "stub-model-v0"
        self.total_api_calls: int   = 0
        self.total_cost_usd:  float = 0.0
        self.calls:           list[tuple[str, str]] = []

    def judge(self, answer: str, criteria: str) -> bool:
        self.calls.append((answer, criteria))
        self.total_api_calls += 1
        return self._trigger in (answer or "").lower()


def test_regrade_updates_rubric_scores_and_ground_truth_match(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    """A run whose final_output mentions the expected substring is regraded
    to ground_truth_match=True; one that doesn't, to False."""
    runs = [
        _stub_run(run_id="r-paris",  final_output='{"answer": "Paris is the capital."}'),
        _stub_run(run_id="r-berlin", final_output='{"answer": "Berlin is in Germany."}'),
    ]
    _write_runs(tmp_runs_path, runs)

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",  # ignored — explicit judge_client wins
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(trigger="Paris"),
    )

    out = [json.loads(l) for l in tmp_runs_path.read_text().splitlines() if l.strip()]
    by_id = {r["run_id"]: r for r in out}

    assert by_id["r-paris" ]["ground_truth_match"] is True
    assert by_id["r-berlin"]["ground_truth_match"] is False
    assert by_id["r-paris" ]["rubric_scores"]["sem_main"] is True
    assert by_id["r-berlin"]["rubric_scores"]["sem_main"] is False
    assert result.stats.regraded == 2


def test_regrade_per_run_grading_call_count_is_delta_not_total(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    """Each run's grading_api_calls must be the delta for THAT run, not the
    cumulative judge total.  Otherwise downstream cost/budget accounting
    double-counts across runs."""
    runs = [_stub_run(run_id=f"r{i}", goal_id="adversarial_001") for i in range(3)]
    _write_runs(tmp_runs_path, runs)

    judge = _StubJudge(trigger="Paris")
    regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = judge,
    )

    out = [json.loads(l) for l in tmp_runs_path.read_text().splitlines() if l.strip()]
    for r in out:
        assert r["grading_api_calls"] == 1
    # Judge total across all runs == sum of per-run deltas.
    assert judge.total_api_calls == sum(r["grading_api_calls"] for r in out)


def test_regrade_recomputes_cascade_failure(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    """
    A run with would_have_been_invalid=True + downstream_consumed=True
    (a propagated schema violation) becomes a cascade ONLY when the new
    ground_truth_match is False.  Verify both directions.
    """
    propagating_step_log = [
        {"step_id": 1, "schema_valid": True,  "would_have_been_invalid": False, "downstream_consumed": False},
        {"step_id": 2, "schema_valid": False, "would_have_been_invalid": True,  "downstream_consumed": True},
    ]
    runs = [
        _stub_run(run_id="r-pass", final_output='{"answer": "Paris."}',
                  per_step=propagating_step_log, cascade_failure=0),
        _stub_run(run_id="r-fail", final_output='{"answer": "Berlin."}',
                  per_step=propagating_step_log, cascade_failure=0),
    ]
    _write_runs(tmp_runs_path, runs)

    regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(trigger="Paris"),
    )

    out = {json.loads(l)["run_id"]: json.loads(l)
           for l in tmp_runs_path.read_text().splitlines() if l.strip()}

    assert out["r-pass"]["cascade_failure"] == 0
    assert out["r-fail"]["cascade_failure"] == 1


def test_regrade_condition_filter(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    """Runs whose condition is NOT in the filter set are skipped."""
    runs = [
        _stub_run(run_id="full",   condition="FULL_FRAMEWORK"),
        _stub_run(run_id="baseline", condition="BASELINE_LANGGRAPH"),
        _stub_run(run_id="basel",  condition="BASELINE_LANGGRAPH"),
    ]
    _write_runs(tmp_runs_path, runs)

    judge = _StubJudge(trigger="Paris")
    result = regrade(
        runs_path         = tmp_runs_path,
        judge_backend     = "ollama",
        goals_dir         = tmp_goals_dir,
        filter_conditions = ["BASELINE_LANGGRAPH"],
        judge_client      = judge,
    )

    assert result.stats.eligible          == 2
    assert result.stats.regraded          == 2
    assert result.stats.skipped_filtered  == 1
    # Judge was only invoked for the two BASELINE runs.
    assert judge.total_api_calls          == 2


def test_regrade_goal_id_filter(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    runs = [
        _stub_run(run_id="r1", goal_id="adversarial_001"),
        _stub_run(run_id="r2", goal_id="adversarial_002"),
    ]
    _write_runs(tmp_runs_path, runs)

    result = regrade(
        runs_path       = tmp_runs_path,
        judge_backend   = "ollama",
        goals_dir       = tmp_goals_dir,
        filter_goal_ids = ["adversarial_002"],
        judge_client    = _StubJudge(trigger="Paris"),
    )

    assert result.stats.regraded == 1
    assert result.stats.skipped_filtered == 1


def test_regrade_skips_already_graded_unless_force(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    """Without --force, runs with grading_api_calls > 0 are not retouched."""
    runs = [
        _stub_run(run_id="ungraded", grading_api_calls=0),
        _stub_run(run_id="graded",   grading_api_calls=1),
    ]
    _write_runs(tmp_runs_path, runs)

    r1 = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(trigger="Paris"),
    )
    assert r1.stats.regraded         == 1
    assert r1.stats.skipped_already  == 1

    # With --force, both are regraded.
    _write_runs(tmp_runs_path, runs)  # reset
    r2 = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        force         = True,
        judge_client  = _StubJudge(trigger="Paris"),
    )
    assert r2.stats.regraded == 2
    assert r2.stats.skipped_already == 0


def test_regrade_skips_crashed_runs(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    runs = [_stub_run(run_id="r-crash", crash_reason="RuntimeError: boom")]
    _write_runs(tmp_runs_path, runs)

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(),
    )
    assert result.stats.skipped_crashed == 1
    assert result.stats.regraded        == 0


def test_regrade_skips_runs_with_missing_final_output(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    runs = [_stub_run(run_id="r-empty", final_output="")]
    _write_runs(tmp_runs_path, runs)

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(),
    )
    assert result.stats.skipped_no_output == 1


def test_regrade_skips_runs_with_unknown_goal_id(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    runs = [_stub_run(run_id="r-orphan", goal_id="goal_does_not_exist")]
    _write_runs(tmp_runs_path, runs)

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(),
    )
    assert result.stats.skipped_no_rubric == 1


def test_regrade_writes_backup_by_default(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    runs = [_stub_run(run_id="r1")]
    _write_runs(tmp_runs_path, runs)
    original_text = tmp_runs_path.read_text()

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(trigger="Paris"),
    )

    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert result.backup_path.read_text() == original_text


def test_regrade_dry_run_does_not_write_runs_or_backup_or_audit(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    runs = [_stub_run(run_id="r1", final_output='{"answer": "Berlin."}')]
    _write_runs(tmp_runs_path, runs)
    original = tmp_runs_path.read_text()

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        dry_run       = True,
        judge_client  = _StubJudge(trigger="Paris"),
    )

    # File untouched
    assert tmp_runs_path.read_text() == original
    # No backup written
    assert result.backup_path is None
    # No audit log written
    assert result.audit_log_path is None
    # But stats reflect the in-memory regrade
    assert result.stats.regraded == 1
    assert result.stats.match_flips == 1   # True → False


def test_regrade_skip_backup_flag(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    runs = [_stub_run(run_id="r1")]
    _write_runs(tmp_runs_path, runs)

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        backup        = False,
        judge_client  = _StubJudge(trigger="Paris"),
    )
    assert result.backup_path is None


def test_regrade_writes_audit_log_with_before_after_diff(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    runs = [
        _stub_run(run_id="r1", final_output='{"answer": "Berlin."}',
                  ground_truth_match=True),  # stub default — about to flip
    ]
    _write_runs(tmp_runs_path, runs)

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(trigger="Paris"),
    )

    assert result.audit_log_path is not None
    audit_lines = [json.loads(l)
                   for l in result.audit_log_path.read_text().splitlines()
                   if l.strip()]
    assert len(audit_lines) == 1
    ev = audit_lines[0]
    assert ev["run_id"]          == "r1"
    assert ev["judge_backend"]   == "ollama"
    assert ev["before"]["ground_truth_match"] is True
    assert ev["after"] ["ground_truth_match"] is False
    assert ev["match_flipped"]   is True


def test_regrade_idempotent_on_second_run_with_force(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    """Running regrade twice with the same judge produces identical content
    on the second pass — i.e. regrade has no hidden state that drifts."""
    runs = [_stub_run(run_id="r1", final_output='{"answer": "Paris."}')]
    _write_runs(tmp_runs_path, runs)

    regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(trigger="Paris"),
    )
    after_first = tmp_runs_path.read_text()

    regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        force         = True,
        judge_client  = _StubJudge(trigger="Paris"),
    )
    after_second = tmp_runs_path.read_text()

    assert after_first == after_second


def test_regrade_rejects_invalid_backend_name(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    runs = [_stub_run()]
    _write_runs(tmp_runs_path, runs)

    with pytest.raises(ValueError):
        regrade(
            runs_path     = tmp_runs_path,
            judge_backend = "stub",   # not 'ollama' or 'anthropic'
            goals_dir     = tmp_goals_dir,
        )


def test_regrade_raises_when_runs_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-file.jsonl"
    with pytest.raises(FileNotFoundError):
        regrade(
            runs_path     = missing,
            judge_backend = "ollama",
        )


def test_regrade_handles_malformed_jsonl_lines(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    """A malformed line is skipped with a WARNING; the rest is regraded."""
    valid = json.dumps(_stub_run(run_id="r-good"))
    tmp_runs_path.write_text(valid + "\n" + "{not valid json\n")

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(trigger="Paris"),
    )

    # Only the valid run was processed.
    assert result.stats.total_runs == 1
    assert result.stats.regraded   == 1


def test_format_summary_includes_judge_metadata(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    _write_runs(tmp_runs_path, [_stub_run(run_id="r1")])
    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        judge_model   = "stub-model-v0",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(),
    )
    out = format_summary(result)
    assert "ollama"          in out
    assert "stub-model-v0"   in out
    assert "regraded:"       in out


def test_result_to_dict_is_json_serialisable(
    tmp_goals_dir: Path, tmp_runs_path: Path,
) -> None:
    _write_runs(tmp_runs_path, [_stub_run(run_id="r1")])
    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir,
        judge_client  = _StubJudge(),
    )
    payload = result_to_dict(result)
    json.dumps(payload)           # must not raise
    assert payload["judge_backend"] == "ollama"
    assert payload["stats"]["regraded"] == 1


def test_regrade_stats_dataclass_starts_at_zero() -> None:
    s = RegradeStats()
    assert s.total_runs         == 0
    assert s.eligible           == 0
    assert s.regraded           == 0
    assert s.match_flips        == 0
    assert s.cascade_flips      == 0
    assert s.skipped_filtered   == 0
    assert s.judge_all_override == 0


@pytest.fixture
def tmp_goals_dir_substring_only(tmp_path: Path) -> Path:
    """
    Build a goals/ tree where the rubric is *purely* substring — no
    semantic_judge.  This is the case --judge-all is designed to
    affect: under default regrade the judge is never called; under
    --judge-all the judge replaces the rubric entirely.
    """
    goals_root = tmp_path / "goals"
    (goals_root / "adversarial_reliability").mkdir(parents=True)
    shutil.copy(_REAL_SCHEMA, goals_root / "_schema.json")

    spec: dict[str, Any] = {
        "id":                  "adversarial_001",
        "goal_type":           "adversarial_reliability",
        "pipeline_depth_k":    3,
        "goal_text":           "Explain backpropagation.",
        "inputs":              {},
        "ground_truth":        "Backpropagation applies the chain rule.",
        "rubric": [{
            "check_id":     "sub_1",
            "check_text":   "Mentions chain rule.",
            "check_type":   "substring",
            "check_params": {"field": "answer", "must_contain": ["chain rule"]},
        }],
        "corruption_pattern":  "drop_required_field",
    }
    (goals_root / "adversarial_reliability" / "adversarial_001.json").write_text(
        json.dumps(spec))
    return goals_root


def test_judge_all_invokes_judge_on_substring_only_rubric(
    tmp_goals_dir_substring_only: Path, tmp_runs_path: Path,
) -> None:
    """
    Default regrade does NOT call the judge for a substring-only rubric.
    --judge-all MUST call the judge once per eligible run.
    """
    runs = [_stub_run(run_id="r1", final_output='{"answer": "Use the chain rule."}')]
    _write_runs(tmp_runs_path, runs)

    judge_default = _StubJudge(trigger="chain rule")
    regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir_substring_only,
        judge_client  = judge_default,
    )
    assert judge_default.total_api_calls == 0

    _write_runs(tmp_runs_path, runs)   # reset
    judge_all = _StubJudge(trigger="chain rule")
    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir_substring_only,
        judge_all     = True,
        force         = True,
        judge_client  = judge_all,
    )
    assert judge_all.total_api_calls       == 1
    assert result.stats.judge_all_override == 1


def test_judge_all_preserves_original_rubric_scores(
    tmp_goals_dir_substring_only: Path, tmp_runs_path: Path,
) -> None:
    """
    Under --judge-all, the original rubric_scores must be preserved
    under `rubric_scores_pre_judge_all` so reviewers can audit the
    override.
    """
    runs = [_stub_run(
        run_id        = "r1",
        final_output  = '{"answer": "Use the chain rule."}',
        rubric_scores = {"sub_1": True},
    )]
    _write_runs(tmp_runs_path, runs)

    regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir_substring_only,
        judge_all     = True,
        judge_client  = _StubJudge(trigger="chain rule"),
    )

    out = json.loads(tmp_runs_path.read_text().splitlines()[0])
    assert out["rubric_scores_pre_judge_all"] == {"sub_1": True}
    assert out["rubric_scores"]               == {"_judge_overall": True}
    assert out["ground_truth_match"]          is True


def test_judge_all_ground_truth_match_reflects_judge_verdict(
    tmp_goals_dir_substring_only: Path, tmp_runs_path: Path,
) -> None:
    """
    Under --judge-all, ground_truth_match must equal the judge verdict
    even when the deterministic rubric WOULD have passed.
    """
    runs = [_stub_run(
        run_id        = "r1",
        final_output  = '{"answer": "Use the chain rule."}',
    )]
    _write_runs(tmp_runs_path, runs)

    # despite the substring "chain rule" being present in the answer.
    regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir_substring_only,
        judge_all     = True,
        judge_client  = _StubJudge(trigger="never-matches-anything"),
    )

    out = json.loads(tmp_runs_path.read_text().splitlines()[0])
    assert out["ground_truth_match"] is False
    assert out["rubric_scores"]       == {"_judge_overall": False}


def test_judge_all_grading_call_count_is_per_run_delta(
    tmp_goals_dir_substring_only: Path, tmp_runs_path: Path,
) -> None:
    """
    Under --judge-all, grading_api_calls must be the per-run delta (1),
    not the cumulative judge total — same invariant as the default
    regrade path.
    """
    runs = [_stub_run(run_id=f"r{i}",
                      final_output='{"answer": "Use the chain rule."}')
            for i in range(3)]
    _write_runs(tmp_runs_path, runs)

    judge = _StubJudge(trigger="chain rule")
    regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir_substring_only,
        judge_all     = True,
        judge_client  = judge,
    )

    for line in tmp_runs_path.read_text().splitlines():
        out = json.loads(line)
        assert out["grading_api_calls"] == 1
    assert judge.total_api_calls == 3


def test_judge_all_audit_log_records_override_flag(
    tmp_goals_dir_substring_only: Path, tmp_runs_path: Path,
) -> None:
    """The audit log must mark every override event with judge_all=True."""
    runs = [_stub_run(run_id="r1",
                      final_output='{"answer": "Use the chain rule."}')]
    _write_runs(tmp_runs_path, runs)

    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir_substring_only,
        judge_all     = True,
        judge_client  = _StubJudge(trigger="chain rule"),
    )

    audit = [json.loads(l)
             for l in result.audit_log_path.read_text().splitlines()
             if l.strip()]
    assert len(audit)              == 1
    assert audit[0]["judge_all"]   is True


def test_judge_all_does_not_double_overwrite_pre_override_field(
    tmp_goals_dir_substring_only: Path, tmp_runs_path: Path,
) -> None:
    """
    Running --judge-all twice on the same file must keep the ORIGINAL
    rubric_scores in `rubric_scores_pre_judge_all`, not the previous
    pass's `_judge_overall` payload.
    """
    runs = [_stub_run(
        run_id        = "r1",
        final_output  = '{"answer": "Use the chain rule."}',
        rubric_scores = {"sub_1": True},
    )]
    _write_runs(tmp_runs_path, runs)

    judge = _StubJudge(trigger="chain rule")
    regrade(runs_path=tmp_runs_path, judge_backend="ollama",
            goals_dir=tmp_goals_dir_substring_only,
            judge_all=True, judge_client=judge)
    regrade(runs_path=tmp_runs_path, judge_backend="ollama",
            goals_dir=tmp_goals_dir_substring_only,
            judge_all=True, force=True, judge_client=judge)

    out = json.loads(tmp_runs_path.read_text().splitlines()[0])
    # The original substring score MUST still be there, not "_judge_overall".
    assert out["rubric_scores_pre_judge_all"] == {"sub_1": True}


def test_judge_all_summary_warns_about_dev020(
    tmp_goals_dir_substring_only: Path, tmp_runs_path: Path,
) -> None:
    """format_summary() must surface the DEV-020 warning when override applied."""
    runs = [_stub_run(run_id="r1",
                      final_output='{"answer": "Use the chain rule."}')]
    _write_runs(tmp_runs_path, runs)

    from mas.eval.regrade import format_summary
    result = regrade(
        runs_path     = tmp_runs_path,
        judge_backend = "ollama",
        goals_dir     = tmp_goals_dir_substring_only,
        judge_all     = True,
        judge_client  = _StubJudge(trigger="chain rule"),
    )
    summary_text = format_summary(result)
    assert "DEV-020"          in summary_text
    assert "EXPLORATORY ONLY" in summary_text
