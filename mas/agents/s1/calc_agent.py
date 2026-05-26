"""CalculatorAgent: exact math via Python execution."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from mas.agents.base import AgentOutput, BaseAgent, TaskInput
from mas.kernel.registry import EscalationReason

# Patterns that require symbolic math — beyond this agent's capability.
_SYMBOLIC = frozenset({
    "integrate", "derivative", "differential equation",
    "eigenvalue", "eigenvector", "fourier transform",
    "laplace transform", "taylor series",
})

_CODE_SYS = (
    "Write a complete Python script that solves the problem and prints "
    "the final answer. Use only the standard library. "
    "Output ONLY executable Python code — no markdown, no explanation, no comments."
)

_ANSWER_SYS = (
    "You are given a computed numeric result. Express it as a clear answer. JSON only:\n"
    '{"answer":"<clear answer>","confidence":0.95,"reasoning":"<brief>","complexity":"medium"}'
)

_FALLBACK_SYS = (
    "Solve this calculation problem step by step. JSON only:\n"
    '{"answer":"<result>","confidence":<0-1>,"reasoning":"<steps>","complexity":"medium"}'
)


class CalculatorAgent(BaseAgent):
    """S1 agent for exact arithmetic via generated Python; escalates symbolic math (P3)."""

    def _can_handle(self, task: TaskInput) -> bool:
        q = task.query.lower()
        return not any(sym in q for sym in _SYMBOLIC)

    def run_internal(self, task: TaskInput) -> AgentOutput:
        ctx         = self.build_context_str(task.context)
        tc: List[Dict[str, Any]] = []
        total_tokens = 0

        if self.ollama is None:
            return self._fallback(task, ctx)

        code_result = self.ollama.generate(
            model       = self.spec.model_name,
            prompt      = (
                f"Problem: {task.query}"
                f"{ctx}\n\n"
                "Write Python code that prints the numerical answer:"
            ),
            system      = _CODE_SYS,
            temperature = 0.05,
            max_tokens  = 400,
        )
        total_tokens += code_result.get("tokens_used", 0)

        if not code_result.get("success"):
            return self._fallback(task, ctx)

        code = self._strip_fences(code_result["text"])

        tool_r = self.use_tool_raw("python_exec", code=code)
        tc.append({
            "tool":    "python_exec",
            "code":    code[:200],
            "success": tool_r.success,
            "output":  tool_r.output[:200],
        })

        if tool_r.success and tool_r.output.strip():
            # Exact result from execution
            d, t = self.llm_json(
                f"Problem: {task.query}\nPython computed: {tool_r.output}\n"
                "Express as a clear final answer:",
                system     = _ANSWER_SYS,
                max_tokens = 300,
            )
            total_tokens += t
            return AgentOutput(
                agent_id    = self.agent_id,
                task_id     = task.task_id,
                answer      = str(d.get("answer") or tool_r.output),
                confidence  = 0.95,   # execution result is exact
                reasoning   = f"Python output: {tool_r.output[:300]}",
                complexity  = "medium",
                tokens_used = total_tokens,
                tool_calls  = tc,
            )

        return self._fallback(task, ctx, total_tokens, tc)


    def _fallback(
        self,
        task:   TaskInput,
        ctx:    str,
        tokens: int = 0,
        tc:     List[Dict[str, Any]] = None,
    ) -> AgentOutput:
        d, t = self.llm_json(
            f"Calculate step by step: {task.query}{ctx}",
            system     = _FALLBACK_SYS,
            max_tokens = 600,
        )
        return AgentOutput(
            agent_id    = self.agent_id,
            task_id     = task.task_id,
            answer      = str(d.get("answer", "Calculation failed")),
            confidence  = self._coerce_confidence(d.get("confidence")),
            reasoning   = d.get("reasoning", "LLM fallback (code execution failed)"),
            complexity  = "medium",
            tokens_used = tokens + t,
            tool_calls  = tc or [],
        )

    @staticmethod
    def _coerce_confidence(value: Any, default: float = 0.5) -> float:
        """
        Coerce a model-provided confidence value into a float in [0, 1].

        Accepts floats/ints, numeric strings, and common qualitative strings
        like "high" / "medium" / "low".
        """
        if value is None:
            return default

        # Numbers
        if isinstance(value, (int, float)):
            v = float(value)
            return max(0.0, min(1.0, v))

        # Strings: numeric or qualitative
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"high", "h"}:
                return 0.9
            if s in {"medium", "med", "m"}:
                return 0.6
            if s in {"low", "l"}:
                return 0.3
            try:
                v = float(s)
                return max(0.0, min(1.0, v))
            except ValueError:
                return default

        # Anything else (dict/list/etc.)
        return default

    @staticmethod
    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            lines = t.splitlines()
            inner = lines[1:-1] if len(lines) > 2 else lines[1:]
            return "\n".join(inner).strip()
        return t
