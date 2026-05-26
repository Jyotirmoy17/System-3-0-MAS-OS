"""FileAgent: real file I/O."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from mas.agents.base import AgentOutput, BaseAgent, TaskInput

_SYS = (
    "You are a file agent. You have been given real tool output.\n"
    "Summarise it clearly and directly. JSON only:\n"
    '{"answer":"<summary>","confidence":<0-1>,"reasoning":"<brief>","complexity":"easy"}'
)

_FILE_WORDS  = frozenset({"list", "files", "directory", "show", "dir", "what"})
_READ_WORDS  = frozenset({"read", "open", "load", "content", "show", "print", "display"})
_WRITE_WORDS = frozenset({"write", "save", "create", "store", "output"})


class FileAgent(BaseAgent):
    """S1 agent that performs real file operations inside the workspace."""

    def run_internal(self, task: TaskInput) -> AgentOutput:
        q_words  = set(task.query.lower().split())
        q_lower  = task.query.lower()
        tc: List[Dict[str, Any]] = []
        result   = None

        if q_words & _WRITE_WORDS:
            content = task.context.get("content_to_write", task.query)
            path    = self._extract_path(task.query) or "reports/output.md"
            result  = self.use_tool_raw("file_write", path=path, content=content)
            tc.append({"tool": "file_write", "path": path,
                        "success": result.success, "output": str(result)[:150]})

        elif q_words & _READ_WORDS:
            path    = self._extract_path(task.query) or "data"
            is_csv  = ".csv" in path or "csv" in q_lower
            tool    = "file_csv" if is_csv else "file_read"
            result  = self.use_tool_raw(tool, path=path)
            tc.append({"tool": tool, "path": path,
                        "success": result.success, "output": str(result)[:150]})

        else:
            # Default: list workspace
            path   = self._extract_path(task.query) or "."
            result = self.use_tool_raw("file_list", path=path)
            tc.append({"tool": "file_list", "path": path,
                        "success": result.success, "output": str(result)[:150]})

        ctx = self.build_context_str(task.context)
        d, tokens = self.llm_json(
            f"Task: {task.query}\n\nTool result:\n{str(result)[:1400]}{ctx}\n\nSummarise:",
            system    = _SYS,
            max_tokens = 500,
        )

        return AgentOutput(
            agent_id   = self.agent_id,
            task_id    = task.task_id,
            answer     = str(d.get("answer") or str(result)[:400]),
            confidence = float(d.get("confidence", 0.85)),
            reasoning  = d.get("reasoning", "File operation completed"),
            complexity = "easy",
            tokens_used = tokens,
            tool_calls  = tc,
        )

    @staticmethod
    def _extract_path(text: str) -> str:
        for pat in (
            r'["\']([\w./\\-]+\.\w{2,6})["\']',  # quoted path
            r'([\w-]+/[\w./\\-]+)',                 # path-like token
            r'([\w-]+\.\w{2,6})',                  # bare filename
        ):
            m = re.search(pat, text)
            if m:
                return m.group(1)
        return ""
