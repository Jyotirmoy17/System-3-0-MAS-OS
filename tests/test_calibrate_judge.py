"""Unit tests for `mas/eval/calibrate_judge.py`."""
from __future__ import annotations

import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

from mas.eval.calibrate_judge import (
    CalibrationItem,
    build_candidate_items,
    cohens_kappa,
    confusion_matrix,
    grade_interactively,
    interpret_kappa,
    render_summary,
    sample_items,
    summarise,
)
from mas.eval.calibrate_judge import CalibrationResult


def _item(
    *,
    idx:      int,
    run_id:   str            = "r",
    goal_id:  str            = "adversarial_001",
    goal_type:str            = "adversarial_reliability",
    condition:str            = "FULL_FRAMEWORK",
    seed:     int            = 0,
    h:        bool | None    = None,
    a:        bool | None    = None,
    a_err:    str  | None    = None,
    field:    str            = "answer",
    extracted:str            = "",
    criteria: str            = "",
) -> CalibrationItem:
    it = CalibrationItem(
        item_idx       = idx,
        run_id         = run_id,
        goal_id        = goal_id,
        goal_type      = goal_type,
        condition      = condition,
        seed           = seed,
        check_id       = "c1",
        check_text     = "test check",
        criteria       = criteria,
        field          = field,
        extracted_text = extracted,
        goal_text      = "Goal.",
        ground_truth   = "GT.",
    )
    it.human_verdict = h
    it.ai_verdict    = a
    it.ai_error      = a_err
    return it


def test_kappa_perfect_agreement_mixed_classes_is_one() -> None:
    """Both raters agree on every item, with both classes present."""
    items = [
        _item(idx=1, h=True,  a=True),
        _item(idx=2, h=True,  a=True),
        _item(idx=3, h=False, a=False),
        _item(idx=4, h=False, a=False),
    ]
    assert math.isclose(cohens_kappa(items), 1.0)


def test_kappa_perfect_agreement_one_class_is_undefined_nan() -> None:
    """When every grader marks the same single class, kappa is undefined.
    Returning 1.0 here would falsely claim 'perfect agreement' on data
    that conveys no rater discrimination — flag as NaN instead."""
    items = [_item(idx=i, h=True, a=True) for i in range(5)]
    assert math.isnan(cohens_kappa(items))


def test_kappa_zero_agreement_when_at_chance() -> None:
    """50/50 base rates with random agreement → kappa close to 0."""
    items = [
        _item(idx=1, h=True,  a=True),
        _item(idx=2, h=True,  a=False),
        _item(idx=3, h=False, a=True),
        _item(idx=4, h=False, a=False),
    ]
    # p_o = 2/4 = 0.5; p_e = (0.5*0.5 + 0.5*0.5) = 0.5; kappa = 0
    assert math.isclose(cohens_kappa(items), 0.0)


def test_kappa_negative_when_below_chance() -> None:
    items = [
        _item(idx=1, h=True,  a=False),
        _item(idx=2, h=True,  a=False),
        _item(idx=3, h=False, a=True),
        _item(idx=4, h=False, a=True),
    ]
    assert math.isclose(cohens_kappa(items), -1.0)


def test_kappa_substantial_example_landis_koch() -> None:
    """
    A canonical 'substantial' kappa example.  We construct a 30-item
    table with 25/30 agreement and balanced disagreement so the
    Landis–Koch zone test passes.
    """
    items: list[CalibrationItem] = []
    # 12 (Pass, Pass)
    items += [_item(idx=i, h=True,  a=True)  for i in range(12)]
    # 13 (Fail, Fail)
    items += [_item(idx=i, h=False, a=False) for i in range(13)]
    # 3 (Pass, Fail)
    items += [_item(idx=i, h=True,  a=False) for i in range(3)]
    # 2 (Fail, Pass)
    items += [_item(idx=i, h=False, a=True)  for i in range(2)]
    k = cohens_kappa(items)
    assert 0.6 <= k < 0.8, f"expected substantial kappa range, got {k}"
    assert interpret_kappa(k) == "substantial"


def test_kappa_empty_returns_nan() -> None:
    assert math.isnan(cohens_kappa([]))


def test_kappa_skips_ungraded_items() -> None:
    items = [
        _item(idx=1, h=True,  a=True),     # graded
        _item(idx=2, h=None,  a=True),     # skipped
        _item(idx=3, h=True,  a=None, a_err="boom"),   # AI error
        _item(idx=4, h=False, a=False),    # graded
    ]
    # in the agreement subset is True/True, the other False/False — but
    # both are perfect agreement on a 2-item set with both classes →
    # kappa = 1.0).
    k = cohens_kappa(items)
    assert math.isclose(k, 1.0)


@pytest.mark.parametrize("kappa,expected", [
    (-0.1, "poor (worse than chance)"),
    (0.0,  "slight"),
    (0.1,  "slight"),
    (0.3,  "fair"),
    (0.5,  "moderate"),
    (0.7,  "substantial"),
    (0.9,  "almost perfect"),
    (1.0,  "almost perfect"),
])
def test_interpret_kappa_landis_koch_buckets(kappa: float, expected: str) -> None:
    assert interpret_kappa(kappa) == expected


def test_interpret_kappa_nan() -> None:
    assert interpret_kappa(float("nan")) == "undefined"


def test_confusion_matrix_counts() -> None:
    items = [
        _item(idx=1, h=True,  a=True),
        _item(idx=2, h=True,  a=False),
        _item(idx=3, h=False, a=True),
        _item(idx=4, h=False, a=False),
        _item(idx=5, h=False, a=False),
    ]
    cm = confusion_matrix(items)
    assert cm == {
        "human_pass_ai_pass": 1,
        "human_pass_ai_fail": 1,
        "human_fail_ai_pass": 1,
        "human_fail_ai_fail": 2,
    }


def test_summarise_threshold_pass_fail_logic() -> None:
    items: list[CalibrationItem] = []
    items += [_item(idx=i, h=True,  a=True)  for i in range(10)]
    items += [_item(idx=i, h=False, a=False) for i in range(10)]
    summary_pass = summarise(items, threshold=0.6)
    assert summary_pass.passes_threshold is True
    assert summary_pass.cohens_kappa > 0.6

    chance_items = [
        _item(idx=1, h=True,  a=True),
        _item(idx=2, h=True,  a=False),
        _item(idx=3, h=False, a=True),
        _item(idx=4, h=False, a=False),
    ]
    summary_fail = summarise(chance_items, threshold=0.6)
    assert summary_fail.passes_threshold is False


def test_summarise_counts_skipped_and_ai_errors() -> None:
    items = [
        _item(idx=1, h=True,  a=True),
        _item(idx=2, h=None,  a=True),                 # human skipped
        _item(idx=3, h=True,  a=None, a_err="x"),      # AI errored
        _item(idx=4, h=False, a=False),
    ]
    s = summarise(items, threshold=0.6)
    assert s.n_total      == 4
    assert s.n_skipped    == 1
    assert s.n_ai_errors  == 1
    assert s.n_graded     == 2   # only items with both verdicts


def test_summarise_handles_empty_input() -> None:
    s = summarise([], threshold=0.6)
    assert s.n_total == 0
    assert s.n_graded == 0
    assert math.isnan(s.cohens_kappa)
    assert s.passes_threshold is False


def test_sample_items_round_robin_diversity() -> None:
    """When candidates span multiple goal_ids, sampling should not
    saturate one goal_id before touching another."""
    cands = []
    for goal_id in ("g1", "g2", "g3", "g4"):
        for r in range(5):
            cands.append(_item(idx=0, run_id=f"{goal_id}-r{r}",
                               goal_id=goal_id, goal_type="research_synthesis"))
    chosen = sample_items(cands, n=8, seed=0, prioritise_goal_type=None)
    counts: dict[str, int] = {}
    for it in chosen:
        counts[it.goal_id] = counts.get(it.goal_id, 0) + 1
    assert counts == {"g1": 2, "g2": 2, "g3": 2, "g4": 2}


def test_sample_items_respects_priority_goal_type() -> None:
    cands = []
    cands += [_item(idx=0, run_id=f"adv-{i}", goal_id=f"a{i}",
                    goal_type="adversarial_reliability") for i in range(5)]
    cands += [_item(idx=0, run_id=f"rs-{i}", goal_id=f"r{i}",
                    goal_type="research_synthesis") for i in range(5)]

    chosen = sample_items(cands, n=5, seed=1,
                          prioritise_goal_type="adversarial_reliability")
    assert all(it.goal_type == "adversarial_reliability" for it in chosen)


def test_sample_items_is_deterministic_for_same_seed() -> None:
    cands = [_item(idx=0, run_id=f"r{i}", goal_id=f"g{i // 2}") for i in range(10)]
    a = [it.run_id for it in sample_items(cands, n=5, seed=42)]
    b = [it.run_id for it in sample_items(cands, n=5, seed=42)]
    assert a == b


def test_sample_items_assigns_sequential_indices() -> None:
    cands = [_item(idx=0, run_id=f"r{i}", goal_id=f"g{i}") for i in range(5)]
    chosen = sample_items(cands, n=3, seed=0)
    assert [it.item_idx for it in chosen] == [1, 2, 3]


def test_sample_items_caps_at_available() -> None:
    cands = [_item(idx=0, run_id="r0", goal_id="g0")]
    chosen = sample_items(cands, n=10, seed=0)
    assert len(chosen) == 1


def test_sample_items_empty_candidate_pool() -> None:
    assert sample_items([], n=5, seed=0) == []


# Reuse the goals/_schema.json from the repo via the same fixture pattern
# as test_regrade.py.
_REPO_ROOT   = Path(__file__).resolve().parent.parent
_REAL_SCHEMA = _REPO_ROOT / "goals" / "_schema.json"


@pytest.fixture
def tmp_goals_index(tmp_path: Path) -> dict:
    """Build a tiny goals/ tree and return its in-memory index."""
    goals_root = tmp_path / "goals"
    (goals_root / "research_synthesis").mkdir(parents=True)
    shutil.copy(_REAL_SCHEMA, goals_root / "_schema.json")

    spec_with_sem: dict[str, Any] = {
        "id":                "research_synth_001",
        "goal_type":         "research_synthesis",
        "pipeline_depth_k":  3,
        "goal_text":         "Summarise XYZ.",
        "inputs":            {},
        "ground_truth":      "Expected facts.",
        "rubric": [
            {
                "check_id":     "sub_1",
                "check_text":   "Mentions X.",
                "check_type":   "substring",
                "check_params": {"field": "answer", "must_contain": ["X"]},
            },
            {
                "check_id":     "sem_1",
                "check_text":   "Captures key idea.",
                "check_type":   "semantic_judge",
                "check_params": {"field": "answer",
                                 "criteria": "Captures the key idea about X."},
            },
        ],
        "corruption_pattern": None,
    }
    spec_no_sem: dict[str, Any] = {
        **spec_with_sem,
        "id":     "research_synth_002",
        "rubric": [{
            "check_id":     "sub_1",
            "check_text":   "Mentions X.",
            "check_type":   "substring",
            "check_params": {"field": "answer", "must_contain": ["X"]},
        }],
    }
    (goals_root / "research_synthesis" / "research_synth_001.json").write_text(
        json.dumps(spec_with_sem))
    (goals_root / "research_synthesis" / "research_synth_002.json").write_text(
        json.dumps(spec_no_sem))

    from mas.eval.goal_loader import load_all_goals
    return {g.id: g for g in load_all_goals(goals_dir=goals_root)}


def test_build_candidate_items_skips_goals_without_semantic_judge(
    tmp_goals_index: dict,
) -> None:
    runs = [
        {"run_id": "r1", "goal_id": "research_synth_001",
         "condition": "FULL_FRAMEWORK", "seed": 0,
         "final_output": '{"answer": "X is great."}', "crash_reason": None},
        {"run_id": "r2", "goal_id": "research_synth_002",   # has no sem check
         "condition": "FULL_FRAMEWORK", "seed": 0,
         "final_output": '{"answer": "X is great."}', "crash_reason": None},
    ]
    cands = build_candidate_items(runs, tmp_goals_index)
    assert len(cands) == 1
    assert cands[0].goal_id  == "research_synth_001"
    assert cands[0].check_id == "sem_1"
    assert "X is great" in cands[0].extracted_text


def test_build_candidate_items_skips_crashed_runs(tmp_goals_index: dict) -> None:
    runs = [{
        "run_id": "rcrash", "goal_id": "research_synth_001",
        "condition": "FULL_FRAMEWORK", "seed": 0,
        "final_output": '{"answer": "..."}',
        "crash_reason": "RuntimeError: boom",
    }]
    assert build_candidate_items(runs, tmp_goals_index) == []


def test_build_candidate_items_skips_empty_final_output(tmp_goals_index: dict) -> None:
    runs = [{
        "run_id": "rempty", "goal_id": "research_synth_001",
        "condition": "FULL_FRAMEWORK", "seed": 0,
        "final_output": "", "crash_reason": None,
    }]
    assert build_candidate_items(runs, tmp_goals_index) == []


def test_build_candidate_items_skips_unknown_goal_id(tmp_goals_index: dict) -> None:
    runs = [{
        "run_id": "rorphan", "goal_id": "does_not_exist",
        "condition": "FULL_FRAMEWORK", "seed": 0,
        "final_output": '{"answer": "X"}', "crash_reason": None,
    }]
    assert build_candidate_items(runs, tmp_goals_index) == []


class _ScriptedJudge:
    """Returns canned bool verdicts in order."""
    DEFAULT_MODEL = "scripted-v0"

    def __init__(self, verdicts: list[bool | Exception]) -> None:
        self._verdicts = list(verdicts)
        self._model = "scripted-v0"
        self.total_api_calls = 0
        self.total_cost_usd = 0.0

    def judge(self, answer: str, criteria: str) -> bool:
        v = self._verdicts.pop(0)
        if isinstance(v, Exception):
            raise v
        self.total_api_calls += 1
        return v


def _scripted_input(responses: list[str]):
    """Build a callable mimicking input() that returns *responses* in order."""
    iterator = iter(responses)
    def _r() -> str:
        return next(iterator)
    return _r


def test_grade_interactively_full_session(tmp_path: Path) -> None:
    """Run a 3-item session: human says y, n, y; AI says y, y, n.
    Expect 1 agree (#1), 2 disagree (#2 by AI=y/Human=n, #3 reversed)."""
    items = [
        _item(idx=1, run_id="r1", criteria="X", extracted="...A..."),
        _item(idx=2, run_id="r2", criteria="X", extracted="...B..."),
        _item(idx=3, run_id="r3", criteria="X", extracted="...C..."),
    ]
    judge  = _ScriptedJudge([True, True, False])
    writer = io.StringIO()
    reader = _scripted_input(["y", "n", "y"])
    out_path = tmp_path / "calib.json"

    graded = grade_interactively(
        items, judge=judge, output_path=out_path,
        prompt_reader=reader, prompt_writer=writer,
    )

    assert [it.human_verdict for it in graded] == [True,  False, True]
    assert [it.ai_verdict    for it in graded] == [True,  True,  False]
    # Agreement: only item 1.
    summary = summarise(graded, threshold=0.6)
    assert summary.n_graded == 3
    assert math.isclose(summary.raw_agreement, 1/3)


def test_grade_interactively_skip_records_none(tmp_path: Path) -> None:
    items = [_item(idx=1, run_id="r1", criteria="X", extracted="A")]
    judge  = _ScriptedJudge([True])
    reader = _scripted_input(["s"])

    graded = grade_interactively(
        items, judge=judge, output_path=tmp_path / "calib.json",
        prompt_reader=reader, prompt_writer=io.StringIO(),
    )
    assert graded[0].human_verdict is None
    assert graded[0].ai_verdict    is True


def test_grade_interactively_quit_saves_partial_and_returns(tmp_path: Path) -> None:
    items = [
        _item(idx=1, run_id="r1", criteria="X", extracted="A"),
        _item(idx=2, run_id="r2", criteria="X", extracted="B"),
        _item(idx=3, run_id="r3", criteria="X", extracted="C"),
    ]
    judge  = _ScriptedJudge([True, True, True])
    reader = _scripted_input(["y", "q"])         # quit on item 2
    out_path = tmp_path / "calib.json"

    graded = grade_interactively(
        items, judge=judge, output_path=out_path,
        prompt_reader=reader, prompt_writer=io.StringIO(),
    )
    # Item 1 graded; item 2 has AI verdict but no human verdict (quit
    # was raised AFTER ai_verdict was assigned but BEFORE human verdict).
    assert graded[0].human_verdict is True
    assert graded[1].human_verdict is None
    # Partial JSON snapshot was written.
    assert out_path.exists()
    snapshot = json.loads(out_path.read_text())
    assert snapshot["_partial"] is True


def test_grade_interactively_invalid_then_valid(tmp_path: Path) -> None:
    """Invalid responses re-prompt; the valid one wins."""
    items  = [_item(idx=1, run_id="r1", criteria="X", extracted="A")]
    judge  = _ScriptedJudge([False])
    reader = _scripted_input(["maybe", "k", "n"])

    graded = grade_interactively(
        items, judge=judge, output_path=tmp_path / "calib.json",
        prompt_reader=reader, prompt_writer=io.StringIO(),
    )
    assert graded[0].human_verdict is False


def test_grade_interactively_records_ai_error(tmp_path: Path) -> None:
    items  = [_item(idx=1, run_id="r1", criteria="X", extracted="A")]
    judge  = _ScriptedJudge([RuntimeError("ollama down")])
    reader = _scripted_input(["y"])

    graded = grade_interactively(
        items, judge=judge, output_path=tmp_path / "calib.json",
        prompt_reader=reader, prompt_writer=io.StringIO(),
    )
    assert graded[0].ai_verdict is None
    assert graded[0].ai_error   is not None
    assert "ollama down" in graded[0].ai_error


def test_render_summary_contains_kappa_and_threshold_decision() -> None:
    items: list[CalibrationItem] = []
    items += [_item(idx=i, h=True,  a=True)  for i in range(10)]
    items += [_item(idx=i, h=False, a=False) for i in range(10)]
    summary = summarise(items, threshold=0.6)
    result  = CalibrationResult(
        metadata = {
            "judge_backend": "ollama",
            "judge_model":   "qwen2.5:14b",
            "runs_path":     "runs.jsonl",
            "sample_seed":   42,
            "n_requested":   20,
        },
        items   = items,
        summary = summary,
    )
    text = render_summary(result)
    assert "Cohen's kappa" in text
    assert "DEV-019"       in text
    assert "PASS"          in text
    assert "qwen2.5:14b"   in text
