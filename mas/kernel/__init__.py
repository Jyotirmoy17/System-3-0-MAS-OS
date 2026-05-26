"""MASKernel — the assembled OS kernel."""
from __future__ import annotations

import datetime
import fcntl
import json
import logging
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from .budget    import BudgetManager
from .graph     import build_graph
from .registry  import AgentRegistry, Capability, CostProfile, EscalationReason, Tier
from .scheduler import Scheduler
from .schemas   import SCHEMAS
from .state     import OSState, make_initial_state
from .validator import System0

# Re-export the public types for convenience
__all__ = [
    "MASKernel",
    "create_kernel",
    "AgentRegistry",
    "AgentSpec",
    "Capability",
    "CostProfile",
    "EscalationReason",
    "Tier",
    "System0",
    "BudgetManager",
    "Scheduler",
    "SCHEMAS",
]

from .registry import AgentSpec  # noqa: E402  (after __all__)

log = logging.getLogger(__name__)


class MASKernel:
    """Registry, budget, System-0, scheduler, and LangGraph execution."""

    def __init__(
        self,
        registry:           AgentRegistry,
        tool_reg:           Any,             # mas.tools.base.ToolRegistry
        ollama:             Any,             # mas.core.ollama_client.OllamaClient
        budget:             int   = 50_000,
        tau:                float = 0.72,
        condition:          str   = "FULL_FRAMEWORK",
        temperature_override: Optional[float] = None,
    ) -> None:
        self.registry             = registry
        self.tools                = tool_reg
        self.ollama               = ollama
        self.condition            = condition
        self.temperature_override = temperature_override
        self.budget_mgr           = BudgetManager(budget)
        self.system0              = System0(tau)
        self.scheduler            = Scheduler(registry, self.budget_mgr, tau)

        # Instantiate every registered agent class once
        self._agents: Dict[str, Any] = registry.instantiate_all(ollama, tool_reg)

        # temperature_override without a direct import of MASKernel.
        for agent in self._agents.values():
            agent.kernel = self

        # Compile the LangGraph execution graph
        self._graph = build_graph(
            registry   = registry,
            scheduler  = self.scheduler,
            system0    = self.system0,
            budget_mgr = self.budget_mgr,
            agents     = self._agents,
        )

        log.info(
            "MASKernel ready — %d agents, budget=%d, tau=%.2f, condition=%s, temp_override=%s",
            len(self._agents), budget, tau, condition, temperature_override,
        )


    def run(
        self,
        goal:              str,
        session_id:        Optional[str] = None,
        *,
        adversarial_inject:  bool = False,
        corruption_pattern:  str  = "",
        corruption_seed:     int  = 0,
    ) -> Dict[str, Any]:
        """
        Execute *goal* synchronously, write one JSON line to runs.jsonl, return result dict.

        Args:
            adversarial_inject: When True and corruption_pattern is set, applies
                adversarial corruption at step 2.  See EXPERIMENTS.md §3 and Prompt 3.1.
            corruption_pattern: One of the 5 adversarial_wrapper patterns.
            corruption_seed:    Seed forwarded to adversarial_wrapper.corrupt().

        Exceptions are caught and logged; returns success=0.
        """
        run_id     = str(uuid.uuid4())
        started_at = time.time()
        sid        = session_id or uuid.uuid4().hex[:8]
        initial    = make_initial_state(sid, goal, self.budget_mgr.omega)
        initial["adversarial_inject"]  = adversarial_inject
        initial["adversarial_pattern"] = corruption_pattern
        initial["adversarial_seed"]    = corruption_seed

        crash_reason: Optional[str] = None
        try:
            final = self._graph.invoke(initial)
        except Exception as exc:
            crash_reason = f"{type(exc).__name__}: {exc}"
            log.exception("Graph execution failed for session %s", sid)
            final = {
                "per_step_log":    [],
                "final_answer":    None,
                "final_confidence": 0.0,
                "total_tokens":    self.budget_mgr.spent,
                "escalation_count": 0,
                "messages":        [],
                "agent_events":    [],
                "done":            False,
            }

        completed_at = time.time()

        # DAG-traced downstream_consumed (DEV-013): use the step_ids accumulated
        # in step_inputs_used (a flat list of all step_ids that were read as
        # context by any subsequent step) to decide which steps were consumed.
        per_step: list = list(final.get("per_step_log", []))
        consumed_set: set = set(final.get("step_inputs_used", []))
        for step in per_step:
            step["downstream_consumed"] = bool(step.get("step_id") in consumed_set)

        # If any agent hit an unexpected runtime error (caught in BaseAgent.run),
        # mark the whole run as a crash so the evaluation harness records it as
        # success=0 with crash_reason populated.
        if crash_reason is None:
            for step in per_step:
                msg = str(step.get("escalation_message") or "")
                if msg.startswith("UNEXPECTED_ERROR"):
                    crash_reason = msg
                    break

        # Adversarial injection status: detect runs where step 2 was never reached.
        adversarial_injection_status: Optional[str] = None
        if initial.get("adversarial_inject") and initial.get("adversarial_pattern"):
            step2_reached = any(s.get("step_id") == 2 for s in per_step)
            if not step2_reached:
                adversarial_injection_status = "scheduled_but_not_reached"

        # Escalation counts from per-step predicate log.
        pred_counts: Counter = Counter(
            s.get("predicate_fired") for s in per_step if s.get("predicate_fired")
        )

        # Model names from the registry (first registered S1/S2 agent).
        s1_list   = self.registry.all_in_tier(Tier.S1)
        s2_list   = self.registry.all_in_tier(Tier.S2)
        model_slm = s1_list[0].model_name if s1_list else "unknown"
        model_llm = s2_list[0].model_name if s2_list else "unknown"

        total_tokens = int(final.get("total_tokens", 0))
        success      = 1 if (not crash_reason and final.get("final_answer")) else 0

        s0_tel           = self.system0.telemetry()
        would_have_failed = sum(1 for s in per_step if s.get("would_have_been_invalid"))

        def _iso(ts: float) -> str:
            return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        run_entry: Dict[str, Any] = {
            "run_id":             run_id,
            "goal_id":            "",         # filled by eval harness
            "goal_type":          "",         # filled by eval harness
            "pipeline_depth_k":   len(per_step),
            "condition":          self.condition,
            "routing_skipped":    False,
            "seed":               0,          # filled by eval harness
            "model_slm":          model_slm,
            "model_llm":          model_llm,
            "started_at":         _iso(started_at),
            "completed_at":       _iso(completed_at),
            "latency_s":          round(completed_at - started_at, 3),
            "success":            success,
            "cascade_failure":    None,       # computed post-hoc by cascade_detector
            "tokens":             {"slm": 0, "llm": total_tokens, "total": total_tokens},
            "s0": {
                "catches":         s0_tel.get("failed", 0),
                "repairs":         s0_tel.get("repaired", 0),
                "would_have_failed": would_have_failed,
            },
            "escalations": {
                "P1": pred_counts.get(EscalationReason.P1_CONFIDENCE.value, 0),
                "P2": pred_counts.get(EscalationReason.P2_SCHEMA.value, 0),
                "P3": pred_counts.get(EscalationReason.P3_CAPABILITY.value, 0),
                "P4": pred_counts.get(EscalationReason.P4_CONTEXT.value, 0),
            },
            "per_step":                    per_step,
            "final_output":                final.get("final_answer") or "",
            "ground_truth":                "",         # filled by eval harness
            "rubric_scores":               {},         # filled by eval harness
            "ground_truth_match":          None,       # filled by eval harness
            "voc_trace":                   [],
            "crash_reason":                crash_reason,
            "adversarial_injection_status": adversarial_injection_status,
        }

        # fcntl-locked atomic append to runs.jsonl (safe for parallel batch workers).
        log_path = os.environ.get("MAS_RUNS_LOG", "./runs.jsonl")
        try:
            line = json.dumps(run_entry) + "\n"
            with open(log_path, "a", encoding="utf-8") as fh:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                    fh.write(line)
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            log.exception("Failed to write run log entry to %s", log_path)

        # Surface per_step in the result dict for callers that need it.
        final_aug = dict(final)
        final_aug["per_step_log"] = per_step
        return self._package(sid, goal, final_aug)


    async def run_stream(
        self,
        goal:       str,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute *goal* asynchronously and yield events as they happen.

        Each yielded dict has the shape:
            {"node": "s3_plan" | "run_agent" | "s0_validate",
             "event": {...}}          ← agent activity event
        or
            {"node": "...", "message": {...}}  ← chat message
        or
            {"done": True, "session_id": "..."}  ← terminal event
        """
        sid     = session_id or uuid.uuid4().hex[:8]
        initial = make_initial_state(sid, goal, self.budget_mgr.omega)

        try:
            async for chunk in self._graph.astream(initial, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    for ev in node_output.get("agent_events", []):
                        yield {"node": node_name, "event": ev}
                    for msg in node_output.get("messages", []):
                        yield {"node": node_name, "message": msg}
        except Exception:
            log.exception("Streaming graph execution failed, session=%s", sid)
            yield {"node": "error", "event": {"type": "error", "msg": "Execution failed"}}

        yield {"done": True, "session_id": sid}


    def telemetry(self) -> Dict[str, Any]:
        return {
            "registry": self.registry.summary(),
            "system0":  self.system0.telemetry(),
            "budget":   self.budget_mgr.summary(),
        }


    def _package(
        self,
        sid:   str,
        goal:  str,
        final: Dict[str, Any],
    ) -> Dict[str, Any]:
        answer = final.get("final_answer") or ""
        if not answer:
            # Attempt to recover an answer from the last message
            messages = final.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "agent" and msg.get("content"):
                    answer = msg["content"]
                    break
        return {
            "session_id":   sid,
            "goal":         goal,
            "answer":       answer or "No answer produced.",
            "confidence":   round(final.get("final_confidence", 0.0), 4),
            "complexity":   final.get("complexity", "unknown"),
            "total_tokens": final.get("total_tokens", 0),
            "escalations":  final.get("escalation_count", 0),
            "messages":     final.get("messages", []),
            "agent_events": final.get("agent_events", []),
            "per_step_log": final.get("per_step_log", []),
            "system0":      self.system0.telemetry(),
            "budget":       self.budget_mgr.summary(),
        }


def create_kernel(
    slm_model:            str            = "phi3:mini",
    llm_model:            str            = "llama3.1:8b",
    ollama_url:           str            = "http://localhost:11434",
    budget:               int            = 50_000,
    tau:                  float          = 0.72,
    workspace:            Optional[Path] = None,
    condition:            str            = "FULL_FRAMEWORK",
    _llm_client:          Any            = None,
    temperature_override: Optional[float] = None,
) -> MASKernel:
    """
    Assemble and return a fully-initialised MASKernel.

    This is the canonical entry point.  All imports from lower layers
    happen inside this function to keep the module-level import graph
    clean and avoid any risk of circular imports at startup.

    Usage::

        from mas.kernel import create_kernel
        kernel = create_kernel()
        result = kernel.run("analyse my sales data")
        print(result["answer"])
    """
    # Local imports — kept here intentionally to avoid circular deps
    from mas.core.ollama_client import OllamaClient      # noqa: PLC0415
    from mas.tools.base         import ToolRegistry       # noqa: PLC0415
    from mas.agents.defaults    import build_registry     # noqa: PLC0415

    _repo_root = Path(__file__).resolve().parent.parent.parent
    ws = workspace or (_repo_root / "workspace")
    ws.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "reports", "memory", "logs", "goals"):
        (ws / sub).mkdir(exist_ok=True)

    ollama   = _llm_client if _llm_client is not None else OllamaClient(ollama_url)
    registry = build_registry(slm_model, llm_model)
    tools    = ToolRegistry(ws)

    return MASKernel(
        registry             = registry,
        tool_reg             = tools,
        ollama               = ollama,
        budget               = budget,
        tau                  = tau,
        condition            = condition,
        temperature_override = temperature_override,
    )
