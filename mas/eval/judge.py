"""LLM-based judge for semantic rubric evaluation."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# Anthropic pricing constants (claude-sonnet-4-5, as of 2026-05)
# Adjust before submission if Anthropic changes rates.

_INPUT_COST_PER_TOKEN  = 3.0  / 1_000_000   # $3.00 / MTok  (input)
_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000   # $15.00 / MTok (output)


_JUDGE_SYSTEM_PROMPT = """\
You are a strict binary rubric evaluator.
Given an ANSWER and a CRITERION, decide whether the answer satisfies the criterion.
Respond on the FIRST line with exactly YES or NO (no other words on that line).
On the SECOND line write exactly one sentence of justification.
Do not add any other text."""

_JUDGE_USER_TEMPLATE = """\
ANSWER:
{answer}

CRITERION:
{criteria}

Does the answer satisfy the criterion? (YES / NO)"""

# Lenient CoT prompts for OllamaJudge (DEV-019 calibration: reduce hyper-strict bias).
_OLLAMA_JUDGE_SYSTEM_PROMPT = """\
You are a lenient expert human grader evaluating whether a model ANSWER satisfies a CRITERION.

Grading philosophy:
- PASS (YES) when the core reasoning and final conclusion are substantially correct,
  even if presentation is imperfect.
- Overlook minor formatting issues, extra conversational filler, markdown noise, or
  small peripheral hallucinations that do NOT change the main result.
- For numeric or logical tasks: accept equivalent forms (rounded numbers within
  reasonable tolerance, correct method with arithmetic slips only if the final
  conclusion still matches the criterion).
- FAIL (NO) only when the answer misses the criterion materially, contradicts itself
  on the key point, or is empty / non-responsive.

You MUST structure your reply in two parts:
1. Step-by-step reasoning inside <thinking>...</thinking> XML tags.
2. After the closing </thinking> tag, output exactly one final line:
   VERDICT: YES
   or
   VERDICT: NO
   (nothing else is required after that line)."""

_OLLAMA_JUDGE_USER_TEMPLATE = """\
ANSWER:
{answer}

CRITERION:
{criteria}

Grade leniently. Think step-by-step in <thinking>...</thinking>, then end with VERDICT: YES or VERDICT: NO."""


@runtime_checkable
class JudgeProtocol(Protocol):
    """Minimal interface that every judge client must implement."""

    def judge(self, answer: str, criteria: str) -> bool:
        """Return True iff *answer* satisfies *criteria*."""
        ...


class AnthropicJudge:
    """Semantic rubric judge using Claude Sonnet via the Anthropic Messages API.

Makes one API call per ``judge()`` invocation with a structured YES/NO
prompt.  Cumulative usage is tracked in ``total_api_calls`` and
``total_cost_usd`` so callers can populate the ``grading_api_calls`` /
``grading_cost_usd`` fields in the RunLog.

On any API error the method falls back to ``True`` (same as the stub) and
logs a WARNING so a transient network fault does not abort a batch run."""

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        api_key: str | None = None,
        model:   str        = DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model   = model
        self.total_api_calls: int   = 0
        self.total_cost_usd:  float = 0.0


    def judge(self, answer: str, criteria: str) -> bool:
        """Call Claude Sonnet to decide whether *answer* satisfies *criteria*.

Parses the model's first output line for YES or NO; anything else is
treated as NO.  Falls back to True with a WARNING on import or API
errors."""
        try:
            import anthropic  # noqa: PLC0415
        except ImportError:
            log.error(
                "anthropic package not installed; "
                "run: pip install anthropic.  "
                "Defaulting semantic_judge verdict to True."
            )
            return True

        client  = anthropic.Anthropic(api_key=self._api_key)
        content = _JUDGE_USER_TEMPLATE.format(answer=answer, criteria=criteria)

        try:
            message = client.messages.create(
                model      = self._model,
                max_tokens = 128,
                system     = _JUDGE_SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": content}],
            )
        except Exception as exc:
            log.warning(
                "AnthropicJudge: API call failed (%s) — defaulting to True.", exc
            )
            return True

        self.total_api_calls += 1

        # Cost accounting
        usage = getattr(message, "usage", None)
        if usage is not None:
            in_tok    = getattr(usage, "input_tokens",  0)
            out_tok   = getattr(usage, "output_tokens", 0)
            call_cost = (
                in_tok  * _INPUT_COST_PER_TOKEN
                + out_tok * _OUTPUT_COST_PER_TOKEN
            )
            self.total_cost_usd += call_cost
            log.info(
                "AnthropicJudge: %d in + %d out tokens; "
                "call $%.6f; cumulative $%.6f",
                in_tok, out_tok, call_cost, self.total_cost_usd,
            )

        # Parse verdict: first non-empty line, first word, upper-cased
        raw_text   = (message.content[0].text if message.content else "").strip()
        first_line = raw_text.split("\n")[0].strip().upper().rstrip(".,;:") if raw_text else ""
        verdict    = first_line == "YES"

        log.info(
            "AnthropicJudge verdict: %s | raw: %s",
            "PASS" if verdict else "FAIL",
            raw_text[:200],
        )
        return verdict


class OllamaJudge:
    """Semantic rubric judge backed by a local model served via Ollama.

Implements the same ``judge(answer, criteria) -> bool`` interface as
``AnthropicJudge`` and satisfies ``JudgeProtocol``.  Intended for
zero-cost grading when an Anthropic API key is unavailable.

Methodological notes
--------------------
1. **Same-model bias.**  Using the same model for grading as for agent
   inference (e.g. ``llama3.1:8b`` agent + ``llama3.1:8b`` judge)
   produces correlated errors and biases acceptance rates.  The default
   model here is ``qwen2.5:14b``, a different family/size from the
   canonical agent model ``llama3.1:8b``.  Override only with explicit
   justification in DEV-019.
2. **Calibration requirement.**  Before reporting metrics derived from
   OllamaJudge in RESULTS.md, run a calibration set (>= 30 hand-labelled
   items) and report Cohen's kappa or % agreement against ground truth.
   See DEV-019.
3. **Determinism.**  Calls Ollama with ``temperature=0`` and a lenient
   chain-of-thought prompt ending in ``VERDICT: YES|NO``.  Output may
   still vary marginally across hardware due to floating-point reordering
   on GPU kernels (per ``mas/eval/harness.py`` non-determinism note).

Cost accounting
---------------
``total_cost_usd`` is always ``0.0`` (local inference).
``total_api_calls`` increments once per ``judge()`` invocation, so the
existing ``grading_api_calls`` log field correctly reflects judge load."""

    DEFAULT_MODEL = "qwen2.5:14b"

    def __init__(
        self,
        model:      str            = DEFAULT_MODEL,
        ollama_url: str            = "http://localhost:11434",
        ollama:     Optional[Any]  = None,
    ) -> None:
        self._model       = model
        self._ollama_url  = ollama_url
        self._ollama      = ollama
        self.total_api_calls: int   = 0
        self.total_cost_usd:  float = 0.0


    def _client(self) -> Any:
        if self._ollama is None:
            from mas.core.ollama_client import OllamaClient  # noqa: PLC0415
            self._ollama = OllamaClient(self._ollama_url)
        return self._ollama


    def judge(self, answer: str, criteria: str) -> bool:
        """Call the local model to decide whether *answer* satisfies *criteria*.

Uses a lenient chain-of-thought prompt (``<thinking>`` block plus
``VERDICT: YES|NO``) at ``temperature=0``.  ``_parse_verdict`` extracts
the binary outcome via regex so CoT prose does not break parsing."""
        prompt = _OLLAMA_JUDGE_USER_TEMPLATE.format(answer=answer, criteria=criteria)

        try:
            resp = self._client().generate(
                model       = self._model,
                prompt      = prompt,
                system      = _OLLAMA_JUDGE_SYSTEM_PROMPT,
                force_json  = False,
                temperature = 0.0,
                max_tokens  = 512,
            )
        except Exception as exc:
            log.warning(
                "OllamaJudge: client error (%s) — defaulting to False.", exc
            )
            return False

        self.total_api_calls += 1

        if not resp.get("success", False):
            log.warning(
                "OllamaJudge: model %s returned success=False (%s) — defaulting to False.",
                self._model, resp.get("error"),
            )
            return False

        raw_text = (resp.get("text") or "").strip()
        verdict  = _parse_verdict(raw_text)

        log.info(
            "OllamaJudge verdict: %s | model: %s | raw: %s",
            "PASS" if verdict else "FAIL",
            self._model,
            raw_text[:200],
        )
        return verdict


_THINKING_BLOCK_RE = re.compile(
    r"<thinking\b[^>]*>.*?</thinking>",
    flags=re.IGNORECASE | re.DOTALL,
)
_VERDICT_YES_RE = re.compile(r"\bVERDICT\s*:\s*YES\b", flags=re.IGNORECASE)
_VERDICT_NO_RE  = re.compile(r"\bVERDICT\s*:\s*NO\b",  flags=re.IGNORECASE)


def _segment_after_thinking(raw_text: str) -> str:
    """Return the portion of *raw_text* after the last </thinking> tag, if any."""
    parts = re.split(r"</thinking>", raw_text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1]
    return raw_text


def _last_verdict_from_segment(segment: str) -> bool | None:
    """Parse the last explicit ``VERDICT: YES|NO`` in *segment*."""
    yes_positions = [m.start() for m in _VERDICT_YES_RE.finditer(segment)]
    no_positions  = [m.start() for m in _VERDICT_NO_RE.finditer(segment)]
    if not yes_positions and not no_positions:
        return None
    last_yes = yes_positions[-1] if yes_positions else -1
    last_no  = no_positions[-1]  if no_positions  else -1
    if last_yes > last_no:
        return True
    if last_no > last_yes:
        return False
    return None


def _parse_verdict(raw_text: str) -> bool:
    """Parse YES/NO from CoT XML, VERDICT line, JSON, or plain text (fail-closed)."""
    if not raw_text:
        return False

    text = raw_text.strip()

    # Path 1–2: explicit VERDICT label (prefer post-CoT tail)
    for segment in (_segment_after_thinking(text), text):
        verdict = _last_verdict_from_segment(segment)
        if verdict is not None:
            return verdict

    # Path 3: structured JSON {"verdict": "YES"|"NO"}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            v = parsed.get("verdict")
            if isinstance(v, str) and v.strip().upper() == "YES":
                return True
            if isinstance(v, str):
                return False
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Path 4: legacy plain-text (strip thinking so "YES" inside CoT is ignored)
    without_thinking = _THINKING_BLOCK_RE.sub("", text).strip()
    body = without_thinking or text
    first_line = body.split("\n", 1)[0].strip()
    if not first_line:
        return False
    first_token = first_line.split(None, 1)[0].strip().upper().rstrip(".,;:")
    if first_token == "YES":
        return True
    if first_token == "NO":
        return False
    return False
