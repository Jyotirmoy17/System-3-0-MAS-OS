"""ToolResult data class and ToolRegistry."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


# Global workspace path
# Set once at kernel startup; default safe for tests.

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE: Path = _REPO_ROOT / "workspace"


def set_workspace(path: Path) -> None:
    """
    Configure the sandbox root for all file and memory operations.
    Must be called before any tool is used.
    """
    global WORKSPACE
    WORKSPACE = path
    path.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "reports", "memory", "logs", "goals"):
        (path / sub).mkdir(exist_ok=True)
    log.debug("Workspace set to %s", WORKSPACE)


@dataclass
class ToolResult:
    """
    The typed result of a tool call.

    Agents receive ToolResult objects from _call_tool().
    __str__ returns a compact, human-readable representation safe to
    inject into an LLM prompt.
    """

    tool:     str
    success:  bool
    output:   str
    error:    Optional[str]    = None
    metadata: Dict[str, Any]   = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            # Truncate very long outputs before they fill the context window
            out = self.output[:2000]
            return f"[{self.tool}] ✓ {out}"
        return f"[{self.tool}] ✗ ERROR: {self.error}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool":     self.tool,
            "success":  self.success,
            "output":   self.output[:2000],
            "error":    self.error,
            "metadata": self.metadata,
        }


def ok(tool: str, output: str, **metadata: Any) -> ToolResult:
    """Convenience constructor for a successful ToolResult."""
    return ToolResult(tool=tool, success=True, output=output, metadata=dict(metadata))


def err(tool: str, error: str) -> ToolResult:
    """Convenience constructor for a failed ToolResult."""
    return ToolResult(tool=tool, success=False, output="", error=error)


class ToolRegistry:
    """
    Central registry for all tool callables.

    Tool implementations are registered by name.  Agents call tools by
    name through call(), which provides a single interception point for
    logging, rate-limiting, or sandboxing.

    Tool classes are loaded lazily in _load_defaults() to avoid the
    circular import that would result from module-level imports here.
    """

    def __init__(self, workspace: Optional[Path] = None) -> None:
        if workspace is not None:
            set_workspace(workspace)

        self._tools: Dict[str, Callable[..., ToolResult]] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """
        Lazy-load all default tool implementations.

        Imports are intentionally local to this method — the tool
        modules import WORKSPACE from this module, so importing them
        at module level would create a circular dependency.
        """
        try:
            # Local imports — this is intentional, not a style issue
            from .file_tool    import FileTool    # noqa: PLC0415
            from .python_tool  import PythonTool  # noqa: PLC0415
            from .web_tool     import WebTool     # noqa: PLC0415
            from .memory_tool  import MemoryTool  # noqa: PLC0415
            from .shell_tool   import ShellTool   # noqa: PLC0415
        except ImportError as exc:
            log.error("Failed to import tool module: %s", exc)
            return

        ft = FileTool()
        pt = PythonTool()
        wt = WebTool()
        mt = MemoryTool()
        st = ShellTool()

        defaults: Dict[str, Callable[..., ToolResult]] = {
            "file_read":      ft.read,
            "file_write":     ft.write,
            "file_append":    ft.append,
            "file_list":      ft.list_dir,
            "file_csv":       ft.read_csv,
            "python_exec":    pt.execute,
            "web_search":     wt.search,
            "web_fetch":      wt.fetch,
            "memory_store":   mt.remember,
            "memory_recall":  mt.recall,
            "memory_list":    mt.list_keys,
            "memory_forget":  mt.forget,
            "shell_exec":     st.run,
        }
        for name, fn in defaults.items():
            self.register(name, fn)

        log.info("ToolRegistry loaded %d tools", len(self._tools))


    def register(self, name: str, fn: Callable[..., ToolResult]) -> None:
        self._tools[name] = fn

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)


    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Invoke a registered tool by name.

        Catches exceptions and returns err() on failure
        so that agents always receive a ToolResult, never an exception.
        """
        fn = self._tools.get(tool_name)
        if fn is None:
            return err(
                tool_name,
                f"Unknown tool '{tool_name}'. Available: {sorted(self._tools)}",
            )
        try:
            result = fn(**kwargs)
            if not isinstance(result, ToolResult):
                return err(tool_name, f"Tool returned {type(result).__name__}, expected ToolResult")
            return result
        except TypeError as exc:
            # Bad keyword arguments — common mistake when building tool calls
            return err(tool_name, f"Bad arguments: {exc}")
        except Exception as exc:
            log.exception("Tool '%s' raised an exception", tool_name)
            return err(tool_name, f"{type(exc).__name__}: {exc}")

    def available(self) -> List[str]:
        return sorted(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools
