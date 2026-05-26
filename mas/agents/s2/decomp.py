"""DecompositionAgent: Least-to-Most prompting with sub-agent spawning."""
from __future__ import annotations

from typing import Any, Dict, List

from mas.agents.base import AgentOutput, BaseAgent, TaskInput

_DECOMP_SYS = (
    "Break this task into 2–4 ordered sub-questions. "
    "Each sub-question should be self-contained and answerable. "
    "JSON only: {\"sub_questions\": [\"...\", \"...\"]}"
)

_STEP_SYS = (
    "Answer this sub-question concisely given the context provided. JSON only:\n"
    '{"answer":"<answer>","confidence":<0-1>,"reasoning":"<brief>","complexity":"medium"}'
)

_SYNTH_SYS = (
    "Synthesise the sub-answers into a complete, well-structured final answer. JSON only:\n"
    '{"answer":"<complete answer>","confidence":<0-1>,"reasoning":"<explanation>","complexity":"hard"}'
)

# Ordered list of S1 agents to try when spawning for a sub-question.
# The first one whose capabilities match gets used.
_SPAWN_CANDIDATES = ("web_agent", "calc_agent", "file_agent", "memory_agent")


class DecompositionAgent(BaseAgent):
    """S2 agent that decomposes problems and spawns specialist sub-agents."""

    def run_internal(self, task: TaskInput) -> AgentOutput:
        total_tokens  = 0
        sub_used: List[str]              = []
        ctx_parts:    List[str]          = []

        d, t = self.llm_json(
            f"Break into sub-questions:\n{task.query}",
            system     = _DECOMP_SYS,
            max_tokens = 300,
        )
        total_tokens += t
        sub_qs: List[str] = d.get("sub_questions") or [task.query]
        sub_qs = [str(q).strip() for q in sub_qs if str(q).strip()][:4]

        for sq in sub_qs:
            answered = False

            # Try spawning a specialist S1 sub-agent first
            for candidate in _SPAWN_CANDIDATES:
                if candidate in self.registry:
                    sub_out = self.spawn(candidate, task, sq, budget=400)
                    if sub_out is not None and sub_out.answer.strip():
                        ctx_parts.append(f"Q: {sq}\nA: {sub_out.answer}")
                        total_tokens += sub_out.tokens_used
                        if sub_out.agent_id not in sub_used:
                            sub_used.append(sub_out.agent_id)
                        answered = True
                        break

            if not answered:
                # Fall back to inline LLM call with accumulated context
                prior = "\n".join(ctx_parts[-2:])
                prompt = (
                    f"Context:\n{prior}\n\nAnswer this sub-question:\n{sq}"
                    if prior else f"Answer:\n{sq}"
                )
                d2, t2 = self.llm_json(prompt, system=_STEP_SYS, max_tokens=400)
                total_tokens += t2
                ctx_parts.append(f"Q: {sq}\nA: {d2.get('answer', '(no answer)')}")

        synthesis_prompt = (
            f"Original task: {task.query}\n\n"
            "Sub-question answers:\n" +
            "\n\n".join(ctx_parts)
        )
        d_final, t_final = self.llm_json(
            synthesis_prompt,
            system     = _SYNTH_SYS,
            max_tokens = 800,
        )
        total_tokens += t_final

        return AgentOutput(
            agent_id        = self.agent_id,
            task_id         = task.task_id,
            answer          = str(d_final.get("answer", "")),
            confidence      = float(d_final.get("confidence", 0.80)),
            reasoning       = "Decomposition:\n" + "\n".join(ctx_parts),
            complexity      = "hard",
            tokens_used     = total_tokens,
            sub_agents_used = sub_used,
        )
