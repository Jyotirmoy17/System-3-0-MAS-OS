"""Whitelisted shell command execution."""
from __future__ import annotations

import subprocess
from typing import Optional

from .base import WORKSPACE, ToolResult, err, ok

_ALLOWED: frozenset = frozenset({
    "awk", "cat", "cut", "date", "df", "du",
    "echo", "find", "grep", "head", "ls",
    "pip", "python3", "pwd", "sort", "tail",
    "uniq", "uname", "wc", "whoami",
})


class ShellTool:
    """
    Executes shell commands from a strict allow-list.

    All commands run with the workspace as the working directory.
    stderr is captured and appended to output.
    """

    def run(self, command: str, timeout: int = 30) -> ToolResult:
        cmd_name = command.strip().split()[0].split("/")[-1]
        if cmd_name not in _ALLOWED:
            return err(
                "shell_exec",
                f"'{cmd_name}' is not on the allow-list. "
                f"Allowed commands: {sorted(_ALLOWED)}",
            )
        try:
            result = subprocess.run(
                command,
                shell=True,          # noqa: S602
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(WORKSPACE),
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr[:400]}"
            return ok(
                "shell_exec", output.strip(),
                returncode=result.returncode,
                command=command,
            )
        except subprocess.TimeoutExpired:
            return err("shell_exec", f"Command timed out after {timeout}s: {command}")
        except Exception as exc:
            return err("shell_exec", str(exc))
