"""VerifierRetryAgent: generate → verify → retry loop."""
from __future__ import annotations

from typing import Any, Dict, List

from mas.agents.base import AgentOutput, BaseAgent, TaskInput

_GEN_SYS = (
    "Think carefully and produce a complete, accurate answer. JSON only:\n"
    '{"answer":"<answer>","confidence":<0-1>,"reasoning":"<explanation>","complexity":"hard"}'
)

_VER_SYS = (
    "You are a strict verifier.  Check whether the answer is correct and complete. "
    "JSON only:\n"
    "{"
    '"is_correct":true/false,'
    '"confidence":<0-1>,'
    '"error_description":"<what is wrong, or empty string if correct>",'
    '"corrected_answer":"<corrected text, or empty string if correct>"'
    "}"
)

_RETRY_SYS = (
    "A previous answer was found to be incorrect.  Produce a corrected answer. "
    "JSON only:\n"
    '{"answer":"<corrected answer>","confidence":<0-1>,'
    '"reasoning":"<explanation>","complexity":"hard"}'
)

MAX_RETRIES = 2


class VerifierRetryAgent(BaseAgent):
    """S2 agent that generates, verifies, and retries for precision."""

    def run_internal(self, task: TaskInput) -> AgentOutput:
        """Generate an answer, optionally verify numeric values, then retry if wrong."""
        ctx          = self.build_context_str(task.context)
        total_tokens = 0
        tc: List[Dict[str, Any]] = []

        d, t = self.llm_json(
            f"Goal: {task.goal}\nQuestion: {task.query}{ctx}",
            system     = _GEN_SYS,
            max_tokens = 800,
        )
        total_tokens  += t
        current        = str(d.get("answer", ""))
        base_conf      = float(d.get("confidence", 0.70))
        base_reasoning = d.get("reasoning", "")

        # Spawn calc_agent to execute the Python verification code;
        # never call python_exec directly from this S2 agent.
        if any(ch.isdigit() for ch in current):
            nums = [
                tok for tok in current.split()
                if any(ch.isdigit() for ch in tok)
            ]
            if nums:
                verify_code = (
                    "# Verify numeric result\n"
                    f"val = {nums[0]}\n"
                    "print('Result:', val)"
                )
                sub_out = self.spawn(
                    "calc_agent",
                    task,
                    f"Execute this Python code and return the result:\n{verify_code}",
                    budget=300,
                )
                tc.append({
                    "tool":    "python_exec",
                    "via":     "calc_agent",
                    "success": sub_out is not None,
                    "output":  (sub_out.answer[:100] if sub_out else "spawn returned None"),
                })
                if sub_out is not None and sub_out.answer.strip():
                    current = sub_out.answer.strip()
                    total_tokens += sub_out.tokens_used

        for _attempt in range(MAX_RETRIES):
            v_d, v_t = self.llm_json(
                f"Question: {task.query}\nAnswer to verify: {current}",
                system     = _VER_SYS,
                max_tokens = 300,
            )
            total_tokens += v_t

            if v_d.get("is_correct", True):
                break

            corrected  = str(v_d.get("corrected_answer", "")).strip()
            error_desc = str(v_d.get("error_description", "")).strip()

            if corrected and corrected != current:
                current = corrected
            elif error_desc:
                r_d, r_t = self.llm_json(
                    f"Question: {task.query}\n"
                    f"Previous answer was wrong: {error_desc}\n"
                    "Produce the correct answer:",
                    system     = _RETRY_SYS,
                    max_tokens = 600,
                )
                total_tokens += r_t
                new_ans = str(r_d.get("answer", "")).strip()
                if new_ans:
                    current = new_ans

        return AgentOutput(
            agent_id    = self.agent_id,
            task_id     = task.task_id,
            answer      = current,
            confidence  = base_conf,
            reasoning   = base_reasoning or f"Verified over {MAX_RETRIES} retry cycles",
            complexity  = "hard",
            tokens_used = total_tokens,
            tool_calls  = tc,
        )
