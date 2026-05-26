"""Python execution tool used by S1 agents (e.g., CalculatorAgent)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional

from mas.tools.base import ToolResult, err, ok


@dataclass
class PythonTool:
    """Execute Python code in a subprocess."""

    python_bin: str = "python3"
    timeout_s: float = 5.0

    def execute(self, code: str, timeout_s: Optional[float] = None) -> ToolResult:
        """
        Execute *code* via `python3` and return stdout (or stderr on failure).

        Args:
            code: Python source code to execute.
            timeout_s: Optional override for the execution timeout.

        Returns:
            ToolResult with stdout in output when success=True, otherwise an error.
        """
        if not isinstance(code, str) or not code.strip():
            return err("python_exec", "Empty code")

        to = float(timeout_s) if timeout_s is not None else self.timeout_s
        try:
            proc = subprocess.run(
                [self.python_bin, "-c", code],
                text=True,
                capture_output=True,
                timeout=to,
            )
        except subprocess.TimeoutExpired:
            return err("python_exec", f"Timeout after {to:.1f}s")
        except FileNotFoundError:
            return err("python_exec", f"Python binary not found: {self.python_bin!r}")
        except Exception as exc:
            return err("python_exec", f"{type(exc).__name__}: {exc}")

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode != 0:
            detail = stderr or stdout or f"Non-zero exit code: {proc.returncode}"
            return err("python_exec", detail[:2000])

        return ok("python_exec", stdout[:2000])
