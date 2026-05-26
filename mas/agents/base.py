"""BaseAgent abstract class, TaskInput, and AgentOutput."""
from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from mas.kernel.registry import AgentSpec, EscalationReason

log = logging.getLogger(__name__)


@dataclass
class TaskInput:
    """Input to one agent for one execution step."""

    task_id:         str
    goal:            str
    query:           str
    context:         Dict[str, str] = field(default_factory=dict)
    budget_tokens:   int            = 2000
    parent_agent_id: Optional[str]  = None
    depth:           int            = 0
    metadata:        Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Typed output from one agent execution."""

    agent_id:           str
    task_id:            str
    answer:             str
    confidence:         float
    reasoning:          str
    complexity:         str               = "medium"
    tokens_used:        int               = 0
    latency_s:          float             = 0.0
    schema_valid:       bool              = True
    schema_repaired:    bool              = False
    escalation_reason:  EscalationReason  = EscalationReason.NONE
    escalation_message: str               = ""
    tool_calls:         List[Dict[str, Any]] = field(default_factory=list)
    sub_agents_used:    List[str]            = field(default_factory=list)
    depth:              int               = 0

    @property
    def needs_escalation(self) -> bool:
        return self.escalation_reason != EscalationReason.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":           self.agent_id,
            "task_id":            self.task_id,
            "answer":             self.answer,
            "confidence":         round(self.confidence, 4),
            "reasoning":          self.reasoning,
            "complexity":         self.complexity,
            "tokens_used":        self.tokens_used,
            "latency_s":          round(self.latency_s, 3),
            "schema_valid":       self.schema_valid,
            "schema_repaired":    self.schema_repaired,
            "escalation_reason":  self.escalation_reason.value,
            "escalation_message": self.escalation_message,
            "tool_calls":         self.tool_calls,
            "sub_agents_used":    self.sub_agents_used,
            "depth":              self.depth,
        }


class BaseAgent(ABC):
    """Abstract base class for every agent. Subclasses implement run_internal()."""

    MAX_DEPTH: int = 4

    def __init__(
        self,
        spec:          AgentSpec,
        ollama_client: Any,
        registry:      Any,         # AgentRegistry — typed Any to avoid circular import
        tool_registry: Any,         # ToolRegistry  — typed Any to avoid circular import
    ) -> None:
        self.spec     = spec
        self.agent_id = spec.agent_id
        self.ollama   = ollama_client
        self.registry = registry
        self.tools    = tool_registry
        self.kernel: Any = None


    def run(self, task: TaskInput) -> AgentOutput:
        """Execute the task, apply P3/P4 pre-checks. Always returns AgentOutput."""
        t0 = time.perf_counter()

        ctx_chars = sum(len(v) for v in task.context.values())
        if ctx_chars > self.spec.context_window_tokens * 4:
            return self._escalate(
                task, EscalationReason.P4_CONTEXT,
                f"Context {ctx_chars} chars > limit",
                time.perf_counter() - t0,
            )

        if not self._can_handle(task):
            return self._escalate(
                task, EscalationReason.P3_CAPABILITY,
                "Agent declared incapable (_can_handle returned False)",
                time.perf_counter() - t0,
            )

        try:
            out           = self.run_internal(task)
            out.latency_s = time.perf_counter() - t0
            out.depth     = task.depth
            return out
        except Exception as exc:
            log.exception("[%s] Unexpected error in run_internal", self.agent_id)
            return self._escalate(
                task, EscalationReason.P3_CAPABILITY,
                f"UNEXPECTED_ERROR {type(exc).__name__}: {exc}",
                time.perf_counter() - t0,
            )

    @abstractmethod
    def run_internal(self, task: TaskInput) -> AgentOutput:
        """Implement the agent logic here.  May assume P3/P4 already cleared."""
        ...

    def _can_handle(self, task: TaskInput) -> bool:
        """Override to declare capability limits (triggers P3). Default: True."""
        return True


    def llm_json(
        self,
        prompt:     str,
        system:     str = "",
        max_tokens: int = 1000,
    ) -> Tuple[Dict[str, Any], int]:
        """Call LLM in JSON mode; return (parsed dict, token count)."""
        if self.ollama is None:
            log.warning("[%s] llm_json called with no ollama client", self.agent_id)
            return {"answer": "", "confidence": 0.3, "reasoning": "No LLM", "complexity": "medium"}, 0

        _kernel = getattr(self, "kernel", None)
        temp_override = getattr(_kernel, "temperature_override", None) if _kernel else None
        temperature = temp_override if temp_override is not None else self.spec.model_temp

        result = self.ollama.generate_json(
            model       = self.spec.model_name,
            prompt      = prompt,
            system      = system,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        if not result.get("success", False) and not result.get("parsed"):
            log.debug("[%s] llm_json failed: %s", self.agent_id, result.get("error"))
        return result.get("parsed") or {}, result.get("tokens_used", 0)


    def use_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Call a tool; return str(ToolResult). Returns error string if unavailable."""
        if self.tools is None:
            return f"[Tool '{tool_name}' unavailable — no ToolRegistry]"
        result = self.tools.call(tool_name, **kwargs)
        return str(result)

    def use_tool_raw(self, tool_name: str, **kwargs: Any):
        """Invoke a tool and return the raw ToolResult (access .success, .output)."""
        if self.tools is None:
            from mas.tools.base import err  # noqa: PLC0415
            return err(tool_name, f"No ToolRegistry attached to agent {self.agent_id}")
        return self.tools.call(tool_name, **kwargs)


    def spawn(self, agent_id: str, task: TaskInput, query: str, budget: int = 500) -> Optional[AgentOutput]:
        """Spawn a sub-agent from the registry (depth-bounded, validated by System-0)."""
        if task.depth >= self.MAX_DEPTH:
            log.warning(
                "[%s] spawn(%s) refused: MAX_DEPTH=%d reached",
                self.agent_id, agent_id, self.MAX_DEPTH,
            )
            return None

        if self.registry is None:
            log.error("[%s] spawn(%s) failed: no registry", self.agent_id, agent_id)
            return None

        cls  = self.registry.get_class(agent_id)
        spec = self.registry.get_spec(agent_id)
        if cls is None or spec is None:
            log.warning("[%s] spawn: agent '%s' not in registry", self.agent_id, agent_id)
            return None

        child   = cls(spec=spec, ollama_client=self.ollama,
                      registry=self.registry, tool_registry=self.tools)
        subtask = TaskInput(
            task_id         = f"{task.task_id}_{agent_id}_{uuid.uuid4().hex[:4]}",
            goal            = task.goal,
            query           = query,
            context         = dict(task.context),
            budget_tokens   = budget,
            parent_agent_id = self.agent_id,
            depth           = task.depth + 1,
        )
        log.debug("[%s] spawning sub-agent %s depth=%d", self.agent_id, agent_id, subtask.depth)
        return child.run(subtask)


    def build_context_str(self, context: Dict[str, str], max_per: int = 400) -> str:
        """Extract answer text from prior-step JSON and return as prompt context."""
        if not context:
            return ""
        parts: List[str] = []
        for step_id, raw in context.items():
            try:
                d      = json.loads(raw)
                answer = str(d.get("answer", "")).strip()
                if answer:
                    parts.append(f"[{step_id}]: {answer[:max_per]}")
                    reasoning = str(d.get("reasoning", "")).strip()
                    if reasoning and len(reasoning) > 30:
                        parts.append(f"  Reasoning: {reasoning[:200]}")
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            # Fallback: truncate raw string
            parts.append(f"[{step_id}]: {str(raw)[:max_per]}")

        return "\n\nPrevious step results:\n" + "\n".join(parts)


    def _escalate(
        self,
        task:    TaskInput,
        reason:  EscalationReason,
        message: str,
        latency: float = 0.0,
    ) -> AgentOutput:
        return AgentOutput(
            agent_id           = self.agent_id,
            task_id            = task.task_id,
            answer             = "",
            confidence         = 0.0,
            reasoning          = f"Escalation [{reason.value}]: {message}",
            complexity         = "hard",
            tokens_used        = 0,
            latency_s          = latency,
            schema_valid       = False,
            escalation_reason  = reason,
            escalation_message = message,
            depth              = task.depth,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id!r}>"
