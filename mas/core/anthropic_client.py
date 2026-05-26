"""Thin wrapper around the Anthropic Python SDK that exposes the same."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


_INPUT_COST_PER_TOKEN  = 3.0  / 1_000_000   # $3.00 / MTok
_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000   # $15.00 / MTok


_BACKOFF_DELAYS: List[float] = [2.0, 4.0, 8.0, 16.0]


_JSON_SYSTEM_SUFFIX = (
    "\n\nRespond with valid JSON only. "
    "Do not include any preamble, explanation, or markdown code fences. "
    "Your entire response must be a single JSON object."
)


class AnthropicClient:
    """
    Anthropic Messages API client with an OllamaClient-compatible interface.

    Args:
        api_key: Anthropic API key.  Reads from ANTHROPIC_API_KEY env var
                 when not supplied.
        default_model: Model slug used when the caller passes the generic
                       frontier model name (default: ``claude-sonnet-4-5``).
    """

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        api_key:       str | None = None,
        default_model: str        = DEFAULT_MODEL,
    ) -> None:
        import os  # noqa: PLC0415
        self._api_key       = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._default_model = default_model
        self._total_tokens  = 0
        self._call_count    = 0
        self._total_cost    = 0.0


    def generate(
        self,
        model:       str,
        prompt:      str,
        system:      str   = "",
        force_json:  bool  = False,
        temperature: float = 0.1,
        max_tokens:  int   = 1500,
    ) -> Dict[str, Any]:
        """
        Single-completion call to the Anthropic Messages API.

        Retries on RateLimitError (429) using exponential back-off.

        Args:
            model:       Model slug (e.g. "claude-sonnet-4-5").  If it does not
                         contain "claude", the default_model is used instead.
            prompt:      User turn content.
            system:      System prompt (empty string → no system turn).
            force_json:  When True, appends a JSON-only instruction to the
                         system prompt so the model outputs raw JSON.
            temperature: Sampling temperature (passed to the API).
            max_tokens:  Maximum output tokens.

        Returns::

            {
                "text":        str,    # raw generation (code fences stripped)
                "tokens_used": int,    # prompt + completion tokens
                "latency_s":   float,
                "model":       str,
                "success":     bool,
                "error":       str | None,
            }
        """
        try:
            import anthropic  # noqa: PLC0415
        except ImportError:
            return {
                "text": "", "tokens_used": 0, "latency_s": 0.0,
                "model": model, "success": False,
                "error": "anthropic package not installed; run: pip install anthropic",
            }

        # Normalise model: if caller passed a local model name, fall back.
        effective_model = model if "claude" in model.lower() else self._default_model

        full_system = system
        if force_json:
            full_system = (system + _JSON_SYSTEM_SUFFIX).strip()

        client  = anthropic.Anthropic(api_key=self._api_key)
        t0      = time.perf_counter()
        message = None

        for attempt, delay in enumerate([0.0] + _BACKOFF_DELAYS):
            if delay > 0:
                log.info(
                    "AnthropicClient: rate-limit hit — waiting %.0f s (attempt %d)",
                    delay, attempt,
                )
                time.sleep(delay)
            try:
                kwargs: Dict[str, Any] = {
                    "model":      effective_model,
                    "max_tokens": max_tokens,
                    "messages":   [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
                if full_system:
                    kwargs["system"] = full_system

                message = client.messages.create(**kwargs)
                break   # success

            except anthropic.RateLimitError:
                if attempt == len(_BACKOFF_DELAYS):
                    latency = round(time.perf_counter() - t0, 3)
                    return {
                        "text": "", "tokens_used": 0, "latency_s": latency,
                        "model": effective_model, "success": False,
                        "error": "Rate limit: exhausted backoff retries (2s/4s/8s/16s)",
                    }
                continue

            except Exception as exc:
                latency = round(time.perf_counter() - t0, 3)
                log.error("AnthropicClient: API error: %s", exc)
                return {
                    "text": "", "tokens_used": 0, "latency_s": latency,
                    "model": effective_model, "success": False,
                    "error": str(exc),
                }

        latency = round(time.perf_counter() - t0, 3)

        # Token and cost accounting
        usage   = getattr(message, "usage", None)
        in_tok  = getattr(usage, "input_tokens",  0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        tokens  = in_tok + out_tok

        call_cost = in_tok * _INPUT_COST_PER_TOKEN + out_tok * _OUTPUT_COST_PER_TOKEN
        self._total_tokens += tokens
        self._call_count   += 1
        self._total_cost   += call_cost

        log.info(
            "AnthropicClient: %d in + %d out tokens; $%.6f; model=%s",
            in_tok, out_tok, call_cost, effective_model,
        )

        # Extract text and strip code fences
        raw = ""
        if message and message.content:
            raw = message.content[0].text if hasattr(message.content[0], "text") else ""
        text = self._strip_fences(raw.strip())

        return {
            "text":        text,
            "tokens_used": tokens,
            "latency_s":   latency,
            "model":       effective_model,
            "success":     True,
            "error":       None,
        }

    def generate_json(
        self,
        model:       str,
        prompt:      str,
        system:      str   = "",
        temperature: float = 0.1,
        max_tokens:  int   = 1500,
        max_retries: int   = 3,
    ) -> Dict[str, Any]:
        """
        JSON-mode completion with retry on parse failure.

        Uses force_json=True on every call and slightly increases temperature
        on retries (same strategy as OllamaClient).

        Returns::

            {
                "parsed":      dict,
                "tokens_used": int,
                "latency_s":   float,
                "model":       str,
                "attempts":    int,
                "success":     bool,
                "error":       str | None,
            }
        """
        total_tokens  = 0
        total_latency = 0.0

        for attempt in range(1, max_retries + 1):
            temp = min(temperature + (attempt - 1) * 0.05, 0.95)
            r    = self.generate(
                model=model, prompt=prompt, system=system,
                force_json=True, temperature=temp, max_tokens=max_tokens,
            )
            total_tokens  += r["tokens_used"]
            total_latency += r["latency_s"]

            if not r["success"]:
                if attempt == max_retries:
                    return {
                        "parsed": {}, "tokens_used": total_tokens,
                        "latency_s": total_latency, "model": model,
                        "attempts": attempt, "success": False,
                        "error": r["error"],
                    }
                continue

            text = r["text"]
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected dict, got {type(parsed).__name__}")
                if isinstance(parsed.get("answer"), (int, float)):
                    parsed["answer"] = str(parsed["answer"])
                return {
                    "parsed": parsed, "tokens_used": total_tokens,
                    "latency_s": total_latency, "model": model,
                    "attempts": attempt, "success": True, "error": None,
                }
            except (json.JSONDecodeError, ValueError) as exc:
                log.debug(
                    "JSON parse failed (attempt %d/%d): %s | text=%.80s",
                    attempt, max_retries, exc, text,
                )
                if attempt == max_retries:
                    log.warning(
                        "generate_json: %d retries exhausted for model=%s",
                        max_retries, model,
                    )
                    return {
                        "parsed": {}, "tokens_used": total_tokens,
                        "latency_s": total_latency, "model": model,
                        "attempts": attempt, "success": False,
                        "error": f"JSON parse failed: {exc}",
                    }

        return {
            "parsed": {}, "tokens_used": total_tokens,
            "latency_s": total_latency, "model": model,
            "attempts": max_retries, "success": False, "error": "Exhausted retries",
        }


    def is_available(self) -> bool:
        """Return True if the API key is non-empty (no live check to save quota)."""
        return bool(self._api_key)

    def list_models(self) -> List[str]:
        """Return known frontier model slugs without making an API call."""
        return [self._default_model]

    def has_model(self, name: str) -> bool:
        """Return True if *name* matches the configured default model."""
        return name in self._default_model or self._default_model in name


    @property
    def total_tokens(self) -> int:
        """Cumulative tokens used across all generate() calls."""
        return self._total_tokens

    @property
    def call_count(self) -> int:
        """Number of successful generate() calls."""
        return self._call_count

    @property
    def total_cost_usd(self) -> float:
        """Cumulative USD cost of all API calls."""
        return self._total_cost


    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences (``` … ``` or ```json … ```)."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            inner = lines[1:-1] if len(lines) > 2 else lines[1:]
            return "\n".join(inner).strip()
        return stripped
