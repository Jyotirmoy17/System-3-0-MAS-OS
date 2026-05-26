from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def test_kernel_marks_crash_when_step_has_unexpected_error(tmp_path: Path) -> None:
    """
    Regression test: if BaseAgent catches an unexpected error and emits an
    escalation_message starting with 'UNEXPECTED_ERROR', the kernel must log
    success=0 and populate crash_reason.
    """
    from mas.kernel import create_kernel  # local import (slower deps)

    runs_path = tmp_path / "runs.jsonl"
    os.environ["MAS_RUNS_LOG"] = str(runs_path)

    kernel = create_kernel()

    # Monkeypatch the compiled graph to avoid any real inference/tooling.
    def _fake_invoke(_initial: Any) -> dict[str, Any]:
        return {
            "per_step_log": [
                {
                    "step_id": 1,
                    "agent_id": "calc_agent",
                    "tier": 1,
                    "raw_output": "{}",
                    "schema_valid": True,
                    "would_have_been_invalid": False,
                    "downstream_consumed": False,
                    "tokens_used": 0,
                    "latency_ms": 0,
                    "predicate_fired": "p3_capability",
                    "escalation_reason": "p3_capability",
                    "escalation_message": "UNEXPECTED_ERROR ValueError: could not convert string to float",
                    "adversarial_corruption_applied": None,
                    "adversarial_injection_status": None,
                }
            ],
            "step_inputs_used": [],
            "final_answer": "some answer that would otherwise be success",
            "final_confidence": 0.9,
            "total_tokens": 0,
            "escalation_count": 1,
            "messages": [],
            "agent_events": [],
            "done": True,
        }

    kernel._graph.invoke = _fake_invoke  # type: ignore[method-assign]

    kernel.run("dummy goal")

    last = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[-1])
    assert last["success"] == 0
    assert isinstance(last["crash_reason"], str)
    assert last["crash_reason"].startswith("UNEXPECTED_ERROR")
