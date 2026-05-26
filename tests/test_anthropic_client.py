"""Tests for mas/core/anthropic_client.py."""
from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


def _make_anthropic_mock(
    text: str = '{"answer": "42"}',
    input_tokens: int = 100,
    output_tokens: int = 50,
    raise_rate_limit: int = 0,   # raise RateLimitError this many times before succeeding
    raise_generic: bool = False,
) -> types.ModuleType:
    """
    Build a mock ``anthropic`` module that returns a canned response.

    Args:
        text:             Text returned by the model.
        input_tokens:     Simulated input token count.
        output_tokens:    Simulated output token count.
        raise_rate_limit: Number of leading calls that raise RateLimitError.
        raise_generic:    If True the first call raises a generic APIError.
    """
    mock_mod = types.ModuleType("anthropic")

    # Exceptions
    class RateLimitError(Exception):
        pass

    class APIError(Exception):
        pass

    mock_mod.RateLimitError = RateLimitError
    mock_mod.APIError       = APIError

    # Usage object
    usage = MagicMock()
    usage.input_tokens  = input_tokens
    usage.output_tokens = output_tokens

    # Content block
    content_block = MagicMock()
    content_block.text = text

    # Message object
    message = MagicMock()
    message.usage   = usage
    message.content = [content_block]

    call_counter = {"n": 0}

    def _create(**kwargs):
        call_counter["n"] += 1
        n = call_counter["n"]
        if raise_generic and n == 1:
            raise APIError("simulated API error")
        if n <= raise_rate_limit:
            raise RateLimitError("rate limited")
        return message

    client_instance = MagicMock()
    client_instance.messages.create.side_effect = _create

    def _Anthropic(api_key=""):
        return client_instance

    mock_mod.Anthropic = _Anthropic
    return mock_mod


class TestGenerateCalls:
    def test_generate_calls_api(self) -> None:
        """
        generate() must call client.messages.create with the correct shape:
        model, max_tokens, messages list, temperature.
        """
        mock_anthropic = _make_anthropic_mock(text="hello world")

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from mas.core.anthropic_client import AnthropicClient
            client = AnthropicClient(api_key="test-key")

            result = client.generate(
                model="claude-sonnet-4-5",
                prompt="What is 2+2?",
                system="You are a maths tutor.",
                temperature=0.0,
                max_tokens=256,
            )

        assert result["success"] is True
        assert result["text"] == "hello world"
        assert result["model"] == "claude-sonnet-4-5"
        assert result["error"] is None

        # Verify the call shape (captured in the mock)
        created_calls = mock_anthropic.Anthropic("test-key").messages.create.call_args_list
        assert len(created_calls) == 1
        kwargs = created_calls[0].kwargs
        assert kwargs["model"] == "claude-sonnet-4-5"
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.0
        assert any(
            m["role"] == "user" and "2+2" in m["content"]
            for m in kwargs["messages"]
        )
        assert kwargs["system"] == "You are a maths tutor."

    def test_generate_returns_correct_keys(self) -> None:
        """Return dict must contain exactly the OllamaClient-compatible keys."""
        mock_anthropic = _make_anthropic_mock(text="ok")

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from mas.core.anthropic_client import AnthropicClient
            client = AnthropicClient(api_key="test-key")
            result = client.generate("claude-sonnet-4-5", "hello")

        for key in ("text", "tokens_used", "latency_s", "model", "success", "error"):
            assert key in result, f"Missing key: {key}"


class TestGenerateJsonConstrainedFormat:
    def test_json_suffix_in_system_prompt(self) -> None:
        """
        When force_json=True, the system prompt must contain a JSON-only instruction.
        """
        captured_kwargs: dict = {}
        mock_mod = types.ModuleType("anthropic")

        class _RL(Exception): pass
        mock_mod.RateLimitError = _RL

        content_block = MagicMock()
        content_block.text = '{"answer": "yes"}'
        message = MagicMock()
        message.content = [content_block]
        usage = MagicMock()
        usage.input_tokens  = 50
        usage.output_tokens = 20
        message.usage = usage

        def _create(**kwargs):
            captured_kwargs.update(kwargs)
            return message

        client_instance = MagicMock()
        client_instance.messages.create.side_effect = _create
        mock_mod.Anthropic = lambda api_key="": client_instance

        with patch.dict(sys.modules, {"anthropic": mock_mod}):
            from mas.core.anthropic_client import AnthropicClient
            ac = AnthropicClient(api_key="test-key")
            result = ac.generate(
                "claude-sonnet-4-5", "answer this", system="Be concise.",
                force_json=True,
            )

        assert result["success"] is True
        system_sent = captured_kwargs.get("system", "")
        assert "JSON" in system_sent or "json" in system_sent.lower(), (
            f"force_json=True must inject a JSON instruction; got system={system_sent!r}"
        )

    def test_generate_json_parses_dict(self) -> None:
        """generate_json() returns parsed dict on success."""
        mock_anthropic = _make_anthropic_mock(text='{"result": "ok", "value": 42}')

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from mas.core.anthropic_client import AnthropicClient
            client = AnthropicClient(api_key="test-key")
            result = client.generate_json("claude-sonnet-4-5", "give me JSON")

        assert result["success"] is True
        assert result["parsed"] == {"result": "ok", "value": 42}
        assert result["attempts"] == 1


class TestTokenCounting:
    def test_tokens_counted_correctly(self) -> None:
        """tokens_used = input_tokens + output_tokens from usage object."""
        mock_anthropic = _make_anthropic_mock(
            text="hello", input_tokens=120, output_tokens=30
        )

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from mas.core.anthropic_client import AnthropicClient
            client = AnthropicClient(api_key="test-key")
            result = client.generate("claude-sonnet-4-5", "hi")

        assert result["tokens_used"] == 150   # 120 + 30

    def test_total_tokens_accumulates(self) -> None:
        """total_tokens property accumulates across multiple generate() calls."""
        mock_anthropic = _make_anthropic_mock(
            text="ok", input_tokens=100, output_tokens=50
        )

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from mas.core.anthropic_client import AnthropicClient
            client = AnthropicClient(api_key="test-key")
            client.generate("claude-sonnet-4-5", "first call")
            client.generate("claude-sonnet-4-5", "second call")

        assert client.total_tokens == 300   # 150 × 2
        assert client.call_count   == 2


class TestRateLimitBackoff:
    def test_single_retry_on_rate_limit(self) -> None:
        """
        A single RateLimitError triggers a retry (after sleep) and succeeds.
        """
        mock_anthropic = _make_anthropic_mock(
            text='{"ok": true}', raise_rate_limit=1
        )

        sleep_calls: list[float] = []
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
                from mas.core.anthropic_client import AnthropicClient
                client = AnthropicClient(api_key="test-key")
                result = client.generate("claude-sonnet-4-5", "test")

        assert result["success"] is True, result["error"]
        # Must have slept at least once (2 s first backoff)
        assert len(sleep_calls) >= 1
        assert sleep_calls[0] == pytest.approx(2.0)

    def test_exhausted_backoff_returns_failure(self) -> None:
        """
        After all four backoff retries (2/4/8/16 s) the client returns
        success=False without raising.
        """
        mock_anthropic = _make_anthropic_mock(raise_rate_limit=5)

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("time.sleep"):   # suppress actual sleeps in tests
                from mas.core.anthropic_client import AnthropicClient
                client = AnthropicClient(api_key="test-key")
                result = client.generate("claude-sonnet-4-5", "test")

        assert result["success"] is False
        assert "rate limit" in result["error"].lower() or "backoff" in result["error"].lower()

    def test_generic_error_returns_failure(self) -> None:
        """Non-rate-limit API errors return success=False immediately."""
        mock_anthropic = _make_anthropic_mock(raise_generic=True)

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("time.sleep"):
                from mas.core.anthropic_client import AnthropicClient
                client = AnthropicClient(api_key="test-key")
                result = client.generate("claude-sonnet-4-5", "test")

        assert result["success"] is False
        assert result["error"]


class TestCostTracking:
    # claude-sonnet-4-5 pricing
    _INPUT_RATE  = 3.0  / 1_000_000   # $3.00 / MTok
    _OUTPUT_RATE = 15.0 / 1_000_000   # $15.00 / MTok

    def test_cost_computed_per_call(self) -> None:
        """
        total_cost_usd must equal
        input_tokens × $3/MTok + output_tokens × $15/MTok.
        """
        IN, OUT = 200, 100
        mock_anthropic = _make_anthropic_mock(
            text="ok", input_tokens=IN, output_tokens=OUT
        )
        expected_cost = IN * self._INPUT_RATE + OUT * self._OUTPUT_RATE

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from mas.core.anthropic_client import AnthropicClient
            client = AnthropicClient(api_key="test-key")
            client.generate("claude-sonnet-4-5", "prompt")

        assert client.total_cost_usd == pytest.approx(expected_cost, rel=1e-6)

    def test_cost_accumulates_across_calls(self) -> None:
        """total_cost_usd is the cumulative sum across multiple calls."""
        IN, OUT = 100, 50
        mock_anthropic = _make_anthropic_mock(
            text="ok", input_tokens=IN, output_tokens=OUT
        )
        cost_per_call = IN * self._INPUT_RATE + OUT * self._OUTPUT_RATE

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from mas.core.anthropic_client import AnthropicClient
            client = AnthropicClient(api_key="test-key")
            client.generate("claude-sonnet-4-5", "call 1")
            client.generate("claude-sonnet-4-5", "call 2")
            client.generate("claude-sonnet-4-5", "call 3")

        assert client.total_cost_usd == pytest.approx(cost_per_call * 3, rel=1e-6)

    def test_cost_zero_on_rate_limit_exhaustion(self) -> None:
        """Exhausted rate-limit retries (no successful call) → zero cost."""
        mock_anthropic = _make_anthropic_mock(raise_rate_limit=5)

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("time.sleep"):
                from mas.core.anthropic_client import AnthropicClient
                client = AnthropicClient(api_key="test-key")
                client.generate("claude-sonnet-4-5", "prompt")

        assert client.total_cost_usd == pytest.approx(0.0)
