"""Unit tests for mas/eval/judge.py."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _build_mock_anthropic(
    response_text: str = "YES\nLooks good.",
    input_tokens:  int  = 80,
    output_tokens: int  = 20,
    raise_exc:     Exception | None = None,
) -> types.ModuleType:
    """
    Build a fake ``anthropic`` module whose ``Anthropic.messages.create``
    returns a canned response or raises *raise_exc*.

    Args:
        response_text: Raw text in the first content block.
        input_tokens:  Simulated input token usage.
        output_tokens: Simulated output token usage.
        raise_exc:     If not None, ``messages.create`` raises this.
    """
    mod = types.ModuleType("anthropic")

    usage_mock = MagicMock()
    usage_mock.input_tokens  = input_tokens
    usage_mock.output_tokens = output_tokens

    content_block = MagicMock()
    content_block.text = response_text

    message_mock = MagicMock()
    message_mock.content = [content_block]
    message_mock.usage   = usage_mock

    messages_mock = MagicMock()
    if raise_exc is not None:
        messages_mock.create.side_effect = raise_exc
    else:
        messages_mock.create.return_value = message_mock

    client_instance = MagicMock()
    client_instance.messages = messages_mock

    client_cls = MagicMock(return_value=client_instance)
    mod.Anthropic = client_cls

    return mod


def _install_mock(mock_mod: types.ModuleType):
    """Context manager: temporarily replace sys.modules['anthropic']."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        old = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mock_mod
        try:
            yield
        finally:
            if old is None:
                sys.modules.pop("anthropic", None)
            else:
                sys.modules["anthropic"] = old

    return _ctx()


def test_judge_protocol_structural() -> None:
    """
    Any object with a judge(answer, criteria) -> bool method satisfies
    JudgeProtocol at runtime (isinstance check).
    """
    from mas.eval.judge import JudgeProtocol

    class _StubJudge:
        def judge(self, answer: str, criteria: str) -> bool:
            return True

    assert isinstance(_StubJudge(), JudgeProtocol)


def test_anthropic_judge_not_protocol_without_judge_method() -> None:
    """An object without judge() does NOT satisfy JudgeProtocol."""
    from mas.eval.judge import JudgeProtocol

    class _Bad:
        pass

    assert not isinstance(_Bad(), JudgeProtocol)


def test_judge_yes_verdict_returns_true() -> None:
    """YES on the first line → judge() returns True."""
    from mas.eval.judge import AnthropicJudge

    mock_mod = _build_mock_anthropic(response_text="YES\nThe answer is correct.")
    with _install_mock(mock_mod):
        judge = AnthropicJudge(api_key="sk-fake")
        result = judge.judge(answer="Paris", criteria="Identifies capital of France.")

    assert result is True


def test_judge_no_verdict_returns_false() -> None:
    """NO on the first line → judge() returns False."""
    from mas.eval.judge import AnthropicJudge

    mock_mod = _build_mock_anthropic(response_text="NO\nAnswer does not satisfy criterion.")
    with _install_mock(mock_mod):
        judge = AnthropicJudge(api_key="sk-fake")
        result = judge.judge(answer="London", criteria="Identifies capital of France.")

    assert result is False


def test_judge_ambiguous_first_line_returns_false() -> None:
    """
    A first line that is neither YES nor NO is treated as NO (fail-safe).
    """
    from mas.eval.judge import AnthropicJudge

    mock_mod = _build_mock_anthropic(response_text="MAYBE it is correct?")
    with _install_mock(mock_mod):
        judge = AnthropicJudge(api_key="sk-fake")
        result = judge.judge(answer="something", criteria="criterion")

    assert result is False


def test_judge_yes_with_punctuation() -> None:
    """'YES.' (with trailing period) is parsed as YES."""
    from mas.eval.judge import AnthropicJudge

    mock_mod = _build_mock_anthropic(response_text="YES.\nGood.")
    with _install_mock(mock_mod):
        judge = AnthropicJudge(api_key="sk-fake")
        result = judge.judge("answer", "criterion")

    assert result is True


def test_judge_accumulates_api_calls_and_cost() -> None:
    """
    After two successful judge() calls, total_api_calls == 2 and
    total_cost_usd equals the sum of both calls' costs.
    """
    from mas.eval.judge import AnthropicJudge, _INPUT_COST_PER_TOKEN, _OUTPUT_COST_PER_TOKEN

    mock_mod = _build_mock_anthropic(
        response_text="YES\nOK.", input_tokens=100, output_tokens=20
    )
    expected_per_call = 100 * _INPUT_COST_PER_TOKEN + 20 * _OUTPUT_COST_PER_TOKEN

    with _install_mock(mock_mod):
        judge = AnthropicJudge(api_key="sk-fake")
        judge.judge("a1", "c1")
        judge.judge("a2", "c2")

    assert judge.total_api_calls == 2
    assert abs(judge.total_cost_usd - 2 * expected_per_call) < 1e-12


def test_judge_starts_at_zero_counts() -> None:
    """Freshly constructed AnthropicJudge has zero calls and zero cost."""
    from mas.eval.judge import AnthropicJudge

    judge = AnthropicJudge(api_key="sk-fake")
    assert judge.total_api_calls == 0
    assert judge.total_cost_usd  == 0.0


def test_judge_api_error_falls_back_to_true() -> None:
    """
    If messages.create raises any exception, judge() returns True (fail-open)
    and total_api_calls is NOT incremented (the call didn't complete).
    """
    from mas.eval.judge import AnthropicJudge

    mock_mod = _build_mock_anthropic(raise_exc=RuntimeError("network timeout"))
    with _install_mock(mock_mod):
        judge = AnthropicJudge(api_key="sk-fake")
        result = judge.judge("answer", "criterion")

    assert result is True
    assert judge.total_api_calls == 0


def test_judge_import_error_falls_back_to_true() -> None:
    """
    If ``import anthropic`` raises ImportError inside judge(), it falls back
    to True.  We simulate this by patching builtins.__import__ to raise when
    'anthropic' is requested.
    """
    import builtins
    from mas.eval.judge import AnthropicJudge

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("mocked: anthropic not installed")
        return real_import(name, *args, **kwargs)

    # Also remove from sys.modules so the `import anthropic` inside judge()
    # actually reaches builtins.__import__ rather than the module cache.
    saved = sys.modules.pop("anthropic", None)
    try:
        with patch("builtins.__import__", side_effect=_fake_import):
            judge = AnthropicJudge(api_key="sk-fake")
            result = judge.judge("answer", "criterion")
    finally:
        if saved is not None:
            sys.modules["anthropic"] = saved

    assert result is True


def test_judge_empty_response_returns_false() -> None:
    """Empty content list → first_line is empty → verdict is False."""
    from mas.eval.judge import AnthropicJudge

    mock_mod = _build_mock_anthropic(response_text="")
    # Override: make content empty
    msg = MagicMock()
    msg.content = []
    msg.usage   = MagicMock(input_tokens=10, output_tokens=5)
    mock_mod.Anthropic.return_value.messages.create.return_value = msg

    with _install_mock(mock_mod):
        judge = AnthropicJudge(api_key="sk-fake")
        result = judge.judge("answer", "criterion")

    assert result is False


class _StubOllama:
    """
    Minimal OllamaClient stand-in for OllamaJudge tests.

    Records the arguments passed to ``generate()`` and returns a canned
    response.  No network, no model.  Mirrors the OllamaClient.generate()
    return shape exactly.
    """

    def __init__(
        self,
        text:        str  = '{"verdict": "YES"}',
        success:     bool = True,
        error:       str  = "",
        raise_exc:   Exception | None = None,
    ) -> None:
        self.text        = text
        self.success     = success
        self.error       = error
        self.raise_exc   = raise_exc
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return {
            "text":        self.text,
            "tokens_used": 12,
            "latency_s":   0.01,
            "model":       kwargs.get("model", ""),
            "success":     self.success,
            "error":       self.error or None,
        }


def test_ollama_judge_satisfies_protocol() -> None:
    from mas.eval.judge import JudgeProtocol, OllamaJudge
    assert isinstance(OllamaJudge(ollama=_StubOllama()), JudgeProtocol)


def test_ollama_judge_yes_verdict_returns_true() -> None:
    """JSON {"verdict": "YES"} → True."""
    from mas.eval.judge import OllamaJudge

    stub = _StubOllama(text='{"verdict": "YES"}')
    judge = OllamaJudge(ollama=stub)
    assert judge.judge("Paris", "Identifies capital of France.") is True
    assert judge.total_api_calls == 1
    assert judge.total_cost_usd == 0.0


def test_ollama_judge_no_verdict_returns_false() -> None:
    """JSON {"verdict": "NO"} → False."""
    from mas.eval.judge import OllamaJudge

    judge = OllamaJudge(ollama=_StubOllama(text='{"verdict": "NO"}'))
    assert judge.judge("London", "Capital of France.") is False


def test_ollama_judge_plaintext_yes_fallback() -> None:
    """Non-JSON 'YES\\nbecause...' first-token fallback → True."""
    from mas.eval.judge import OllamaJudge

    judge = OllamaJudge(ollama=_StubOllama(text="YES\nThe answer is correct."))
    assert judge.judge("Paris", "Capital of France.") is True


def test_ollama_judge_plaintext_punctuation_yes() -> None:
    """'YES.' with trailing punctuation → True."""
    from mas.eval.judge import OllamaJudge

    judge = OllamaJudge(ollama=_StubOllama(text="YES."))
    assert judge.judge("a", "c") is True


def test_ollama_judge_ambiguous_returns_false_failsafe() -> None:
    """
    OllamaJudge fails CLOSED on ambiguity (unlike AnthropicJudge which
    is documented as fail-OPEN on API errors).  This is deliberate —
    a judge running locally should not silently pass adversarial outputs
    just because the model emitted 'MAYBE'.
    """
    from mas.eval.judge import OllamaJudge

    judge = OllamaJudge(ollama=_StubOllama(text="MAYBE it is correct?"))
    assert judge.judge("a", "c") is False


def test_ollama_judge_empty_response_returns_false() -> None:
    from mas.eval.judge import OllamaJudge

    judge = OllamaJudge(ollama=_StubOllama(text=""))
    assert judge.judge("a", "c") is False


def test_ollama_judge_unsuccessful_response_returns_false() -> None:
    """If Ollama returns success=False, judge() returns False (fail-closed)."""
    from mas.eval.judge import OllamaJudge

    judge = OllamaJudge(ollama=_StubOllama(success=False, error="connection refused"))
    assert judge.judge("a", "c") is False


def test_ollama_judge_client_exception_returns_false() -> None:
    """A raised exception inside generate() falls back to False."""
    from mas.eval.judge import OllamaJudge

    judge = OllamaJudge(ollama=_StubOllama(raise_exc=RuntimeError("ollama down")))
    assert judge.judge("a", "c") is False
    # Exception path does NOT increment total_api_calls.
    assert judge.total_api_calls == 0


def test_ollama_judge_uses_temperature_zero_and_prose_cot_mode() -> None:
    """OllamaJudge uses temperature=0 and prose (non-JSON) mode for CoT grading."""
    from mas.eval.judge import OllamaJudge

    cot = (
        "<thinking>\nCore math is 8600 doses; formatting is messy.\n</thinking>\n\n"
        "VERDICT: YES"
    )
    stub = _StubOllama(text=cot)
    judge = OllamaJudge(ollama=stub)
    assert judge.judge("answer", "criteria") is True

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["temperature"] == 0.0
    assert call["force_json"] is False
    assert call["max_tokens"] >= 256
    assert "lenient" in str(call["system"]).lower()


def test_parse_verdict_cot_yes_after_thinking() -> None:
    """CoT block followed by VERDICT: YES is parsed as pass."""
    from mas.eval.judge import _parse_verdict

    raw = (
        "<thinking>\nStep 1: check reasoning.\nStep 2: conclusion matches.\n"
        "</thinking>\n\nVERDICT: YES"
    )
    assert _parse_verdict(raw) is True


def test_parse_verdict_cot_no_after_thinking() -> None:
    """CoT block followed by VERDICT: NO is parsed as fail."""
    from mas.eval.judge import _parse_verdict

    raw = (
        "<thinking>\nThe answer misses the required value.\n</thinking>\n"
        "VERDICT: NO"
    )
    assert _parse_verdict(raw) is False


def test_parse_verdict_prefers_post_thinking_verdict_over_thinking_yes() -> None:
    """YES mentioned only inside <thinking> does not override trailing VERDICT: NO."""
    from mas.eval.judge import _parse_verdict

    raw = (
        "<thinking>\nMaybe VERDICT: YES if generous.\n</thinking>\nVERDICT: NO"
    )
    assert _parse_verdict(raw) is False


def test_parse_verdict_json_legacy_still_works() -> None:
    """Legacy JSON verdict responses remain supported."""
    from mas.eval.judge import _parse_verdict

    assert _parse_verdict('{"verdict": "YES"}') is True
    assert _parse_verdict('{"verdict": "NO"}') is False


def test_ollama_judge_default_model_is_qwen_for_bias_reduction() -> None:
    """
    Default model is qwen2.5:14b — different family from the canonical
    agent model llama3.1:8b — to reduce same-model judge bias.
    See OllamaJudge docstring + DEV-019.
    """
    from mas.eval.judge import OllamaJudge
    assert OllamaJudge.DEFAULT_MODEL == "qwen2.5:14b"


def test_ollama_judge_model_override() -> None:
    """Explicit model override is forwarded to OllamaClient.generate()."""
    from mas.eval.judge import OllamaJudge

    stub = _StubOllama()
    judge = OllamaJudge(model="phi3:mini", ollama=stub)
    judge.judge("a", "c")
    assert stub.calls[0]["model"] == "phi3:mini"


def test_score_rubric_semantic_judge_yes(tmp_path) -> None:
    """
    score_rubric() with a judge_client that returns True for a
    semantic_judge check produces scores[check_id] == True.
    """
    from mas.eval.harness import score_rubric
    from mas.eval.judge import JudgeProtocol

    class _YesJudge:
        def judge(self, answer: str, criteria: str) -> bool:
            return True

    rubric = [{
        "check_id":    "sem_1",
        "check_text":  "Answer is factually correct.",
        "check_type":  "semantic_judge",
        "check_params": {"field": "answer", "criteria": "Must name Paris."},
    }]
    scores = score_rubric(rubric, '{"answer": "Paris"}', judge_client=_YesJudge())
    assert scores["sem_1"] is True


def test_score_rubric_semantic_judge_no() -> None:
    """
    score_rubric() with a judge_client that returns False produces
    scores[check_id] == False.
    """
    from mas.eval.harness import score_rubric

    class _NoJudge:
        def judge(self, answer: str, criteria: str) -> bool:
            return False

    rubric = [{
        "check_id":    "sem_2",
        "check_text":  "Answer must name Paris.",
        "check_type":  "semantic_judge",
        "check_params": {"field": "answer", "criteria": "Must name Paris."},
    }]
    scores = score_rubric(rubric, '{"answer": "London"}', judge_client=_NoJudge())
    assert scores["sem_2"] is False


def test_score_rubric_semantic_judge_stub_returns_true(caplog) -> None:
    """
    When judge_client=None (default), semantic_judge returns True and
    logs a WARNING.
    """
    import logging
    from mas.eval.harness import score_rubric

    rubric = [{
        "check_id":    "sem_3",
        "check_text":  "Any semantic check.",
        "check_type":  "semantic_judge",
        "check_params": {"field": "answer", "criteria": "anything"},
    }]

    with caplog.at_level(logging.WARNING, logger="mas.eval.harness"):
        scores = score_rubric(rubric, '{"answer": "text"}', judge_client=None)

    assert scores["sem_3"] is True
    assert any("semantic_judge" in r.message.lower() or "stub" in r.message.lower()
               for r in caplog.records)


def test_score_rubric_must_not_contain_present_fails() -> None:
    """
    A substring check with must_not_contain should FAIL when the forbidden
    word IS present in the output.  This is the DEV-005 regression guard.
    """
    from mas.eval.harness import score_rubric

    rubric = [{
        "check_id":    "mnc_1",
        "check_text":  "Output must not mention corruption marker.",
        "check_type":  "substring",
        "check_params": {
            "field": "answer",
            "must_contain":     [],
            "must_not_contain": ["CORRUPTED"],
        },
    }]
    output = '{"answer": "This CORRUPTED value leaked into the answer."}'
    scores = score_rubric(rubric, output)
    assert scores["mnc_1"] is False, "must_not_contain violation should fail the check"


def test_score_rubric_must_not_contain_absent_passes() -> None:
    """
    A substring check with must_not_contain PASSES when the forbidden word
    is absent.  DEV-005 — happy path.
    """
    from mas.eval.harness import score_rubric

    rubric = [{
        "check_id":    "mnc_2",
        "check_text":  "Output must not mention corruption marker.",
        "check_type":  "substring",
        "check_params": {
            "field": "answer",
            "must_contain":     ["result"],
            "must_not_contain": ["CORRUPTED"],
        },
    }]
    output = '{"answer": "The final result is 42."}'
    scores = score_rubric(rubric, output)
    assert scores["mnc_2"] is True


def test_score_rubric_must_not_contain_case_insensitive() -> None:
    """must_not_contain check is case-insensitive."""
    from mas.eval.harness import score_rubric

    rubric = [{
        "check_id":    "mnc_3",
        "check_type":  "substring",
        "check_params": {
            "field": "answer",
            "must_not_contain": ["corrupted"],
        },
    }]
    output = '{"answer": "This Corrupted value leaked."}'
    scores = score_rubric(rubric, output)
    assert scores["mnc_3"] is False


def test_score_rubric_combined_must_contain_and_not() -> None:
    """Both must_contain AND must_not_contain constraints evaluated together."""
    from mas.eval.harness import score_rubric

    rubric = [{
        "check_id":   "mnc_4",
        "check_type": "substring",
        "check_params": {
            "field":             "answer",
            "must_contain":      ["paris"],
            "must_not_contain":  ["london"],
        },
    }]

    assert score_rubric(rubric, '{"answer": "Paris is the capital."}')["mnc_4"] is True
    assert score_rubric(rubric, '{"answer": "Paris and London."}')["mnc_4"] is False
    assert score_rubric(rubric, '{"answer": "Berlin is nice."}')["mnc_4"] is False
