"""Thin, robust wrapper around the Ollama REST API."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

log = logging.getLogger(__name__)

_RETRY = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503])


class OllamaClient:
    """
    Ollama REST API client.

    All public methods return a dict — they never raise RuntimeError on
    network failure; instead they return a dict with success=False and
    an error message, letting the caller decide how to handle it.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.mount("http://", HTTPAdapter(max_retries=_RETRY))
        self._session.mount("https://", HTTPAdapter(max_retries=_RETRY))
        self._total_tokens = 0
        self._call_count   = 0


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
        Single-completion call to Ollama.

        Returns::

            {
                "text":        str,    # raw generation output
                "tokens_used": int,
                "latency_s":   float,
                "model":       str,
                "success":     bool,
                "error":       str | None,
            }
        """
        payload: Dict[str, Any] = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if force_json:
            payload["format"] = "json"

        t0 = time.perf_counter()
        try:
            resp = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            log.error("Ollama not reachable at %s: %s", self.base_url, exc)
            return {
                "text": "", "tokens_used": 0,
                "latency_s": 0.0, "model": model,
                "success": False,
                "error": f"Connection error: Ollama not running at {self.base_url}",
            }
        except requests.exceptions.HTTPError as exc:
            log.error("Ollama HTTP error: %s", exc)
            return {
                "text": "", "tokens_used": 0,
                "latency_s": 0.0, "model": model,
                "success": False, "error": str(exc),
            }
        except requests.exceptions.Timeout:
            log.error("Ollama request timed out after 300 s")
            return {
                "text": "", "tokens_used": 0,
                "latency_s": 300.0, "model": model,
                "success": False, "error": "Timeout",
            }

        data   = resp.json()
        tokens = (data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0)
        self._total_tokens += tokens
        self._call_count   += 1

        return {
            "text":        data.get("response", "").strip(),
            "tokens_used": tokens,
            "latency_s":   round(time.perf_counter() - t0, 3),
            "model":       model,
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

        Always returns::

            {
                "parsed":      dict,    # {} on complete failure
                "tokens_used": int,
                "latency_s":   float,
                "model":       str,
                "attempts":    int,
                "success":     bool,
                "error":       str | None,
            }
        """
        total_tokens = 0
        total_latency = 0.0

        for attempt in range(1, max_retries + 1):
            # Slightly increase temperature on each retry to escape
            # degenerate outputs (e.g. an empty JSON object {})
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

            text = self._strip_fences(r["text"])
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected dict, got {type(parsed).__name__}")
                # Coerce numeric "answer" to str (very common LLM mistake)
                if isinstance(parsed.get("answer"), (int, float)):
                    parsed["answer"] = str(parsed["answer"])
                return {
                    "parsed": parsed, "tokens_used": total_tokens,
                    "latency_s": total_latency, "model": model,
                    "attempts": attempt, "success": True, "error": None,
                }
            except (json.JSONDecodeError, ValueError) as exc:
                log.debug(
                    "JSON parse failed (attempt %d/%d): %s | text=%.60s",
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

        # Should never reach here
        return {
            "parsed": {}, "tokens_used": total_tokens,
            "latency_s": total_latency, "model": model,
            "attempts": max_retries, "success": False, "error": "Exhausted retries",
        }


    def is_available(self) -> bool:
        """Return True if the Ollama server responds to /api/tags."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return the list of locally available model names."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def has_model(self, name: str) -> bool:
        """Return True if *name* (or a name containing *name*) is available."""
        return any(name in m for m in self.list_models())


    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def call_count(self) -> int:
        return self._call_count


    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences (```json … ``` or ``` … ```)."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Drop first line (``` or ```json) and last line (```)
            inner = lines[1:-1] if len(lines) > 2 else lines[1:]
            return "\n".join(inner).strip()
        return stripped
