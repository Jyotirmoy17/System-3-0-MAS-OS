"""Shared pytest fixtures for the MAS-OS test suite."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_ollama_client() -> MagicMock:
    """Ollama client that returns canned valid JSON without network I/O."""
    client = MagicMock()

    def _generate_json(
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        return {
            "success":    True,
            "parsed":     {
                "answer":     "Canned answer for testing.",
                "confidence": 0.90,
                "reasoning":  "Mocked reasoning.",
                "complexity": "easy",
            },
            "raw":        '{"answer":"Canned answer for testing.","confidence":0.90,"reasoning":"Mocked reasoning.","complexity":"easy"}',
            "tokens_used": 42,
            "error":      None,
        }

    def _generate_text(
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        return {"success": True, "text": "Canned text response.", "tokens_used": 20, "error": None}

    client.generate_json.side_effect  = _generate_json
    client.generate_text.side_effect  = _generate_text
    client.is_available.return_value  = True
    client.list_models.return_value   = ["phi3:mini", "llama3.1:8b"]
    return client


def _make_per_step(
    step_id: int,
    agent_id: str,
    tier: int,
    schema_valid: bool = True,
    would_have_been_invalid: bool = False,
    downstream_consumed: bool = True,
) -> Dict[str, Any]:
    return {
        "step_id":                        step_id,
        "agent_id":                       agent_id,
        "tier":                           tier,
        "raw_output":                     '{"answer":"ok","confidence":0.9,"reasoning":"fine"}',
        "schema_valid":                   schema_valid,
        "would_have_been_invalid":        would_have_been_invalid,
        "downstream_consumed":            downstream_consumed,
        "tokens_used":                    120,
        "latency_ms":                     4500,
        "adversarial_corruption_applied": None,
        "adversarial_injection_status":   None,
    }


@pytest.fixture()
def sample_run_log() -> Dict[str, Any]:
    """A single run-log dict that matches EXPERIMENTS.md §4 schema exactly."""
    return {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            "research_synth_001",
        "goal_type":          "research_synthesis",
        "pipeline_depth_k":   4,
        "condition":          "FULL_FRAMEWORK",
        "routing_skipped":    False,
        "seed":               42,
        "model_slm":          "phi3:mini",
        "model_llm":          "llama3.1:8b",
        "started_at":         "2026-04-29T10:00:00Z",
        "completed_at":       "2026-04-29T10:01:27Z",
        "latency_s":          87.3,
        "success":            1,
        "cascade_failure":    0,
        "tokens": {"slm": 1240, "llm": 3580, "total": 4820},
        "s0": {"catches": 2, "repairs": 1, "would_have_failed": 2},
        "escalations":        {"P1": 0, "P2": 1, "P3": 0, "P4": 0},
        "per_step": [
            _make_per_step(1, "web_agent",  1),
            _make_per_step(2, "react_agent", 2),
            _make_per_step(3, "calc_agent",  1),
            _make_per_step(4, "react_agent", 2),
        ],
        "final_output":       "Synthesised answer.",
        "ground_truth":       "Expected answer.",
        "rubric_scores":      {"check_1": True, "check_2": True},
        "ground_truth_match": True,
        "voc_trace":          [0.8, 0.6, 0.4, 0.1],
        "crash_reason":       None,
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
        "backend":            "ollama",
        "api_cost_usd":       0.0,
    }


@pytest.fixture()
def tmp_runs_jsonl(tmp_path: Path, sample_run_log: Dict[str, Any]) -> Path:
    """
    A temporary runs.jsonl file pre-seeded with two entries:
      line 1 — success, no cascade failure
      line 2 — cascade failure (would_have_been_invalid + downstream + final failure)
    """
    path = tmp_path / "runs.jsonl"

    # Line 1: clean run
    path.write_text(json.dumps(sample_run_log) + "\n")

    # Line 2: cascade failure
    cascade = dict(sample_run_log)
    cascade["run_id"]             = str(uuid.uuid4())
    cascade["success"]            = 0
    cascade["cascade_failure"]    = 1
    cascade["ground_truth_match"] = False
    cascade["per_step"] = [
        _make_per_step(1, "web_agent",   1),
        _make_per_step(2, "react_agent", 2, schema_valid=False,
                       would_have_been_invalid=True, downstream_consumed=True),
        _make_per_step(3, "calc_agent",  1),
        _make_per_step(4, "react_agent", 2),
    ]
    with path.open("a") as f:
        f.write(json.dumps(cascade) + "\n")

    return path
