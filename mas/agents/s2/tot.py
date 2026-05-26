"""TreeOfThoughtsAgent: N independent branches, scored and pruned."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from mas.agents.base import AgentOutput, BaseAgent, TaskInput

_BRANCH_SYS = (
    "Answer the question using the specified thinking approach. "
    "Be thorough and rigorous. JSON only:\n"
    '{"answer":"<answer>","confidence":<0-1>,"reasoning":"<detailed>","complexity":"hard"}'
)

_SCORE_SYS = (
    "Score this answer on factual accuracy, completeness, and clarity (0–10). "
    "JSON only: {\"score\":<0-10>,\"justification\":\"<one sentence>\"}"
)

N          = 3
APPROACHES = (
    "analytical step-by-step reasoning",
    "intuitive pattern-matching and analogy",
    "first-principles reasoning from axioms",
)


class TreeOfThoughtsAgent(BaseAgent):
    """S2 agent that explores N reasoning branches and returns the best."""

    def run_internal(self, task: TaskInput) -> AgentOutput:
        ctx          = self.build_context_str(task.context)
        total_tokens = 0
        branches:  List[Dict[str, Any]]          = []
        scored:    List[Tuple[float, Dict[str, Any]]] = []

        for approach in APPROACHES[:N]:
            d, t = self.llm_json(
                f"Question: {task.query}{ctx}\n\nUse {approach}:",
                system     = _BRANCH_SYS,
                max_tokens = 600,
            )
            total_tokens += t
            if d:
                branches.append(d)

        if not branches:
            return AgentOutput(
                agent_id    = self.agent_id,
                task_id     = task.task_id,
                answer      = "",
                confidence  = 0.0,
                reasoning   = "No branches generated (LLM unavailable)",
                complexity  = "hard",
                tokens_used = total_tokens,
            )

        for branch in branches:
            s_d, s_t = self.llm_json(
                f"Question: {task.query}\n\n"
                f"Answer: {branch.get('answer', '')}\n\n"
                f"Reasoning: {branch.get('reasoning', '')[:250]}",
                system     = _SCORE_SYS,
                max_tokens = 150,
            )
            total_tokens += s_t
            try:
                score = max(0.0, min(10.0, float(s_d.get("score", 5.0))))
            except (TypeError, ValueError):
                score = 5.0
            scored.append((score, branch))

        best_score, best = max(scored, key=lambda x: x[0])
        confidence = min(0.95, 0.5 + best_score / 20.0)

        return AgentOutput(
            agent_id    = self.agent_id,
            task_id     = task.task_id,
            answer      = str(best.get("answer", "")),
            confidence  = confidence,
            reasoning   = (
                f"[ToT best branch score={best_score:.1f}/10]\n"
                + best.get("reasoning", "")
            ),
            complexity  = "hard",
            tokens_used = total_tokens,
        )
