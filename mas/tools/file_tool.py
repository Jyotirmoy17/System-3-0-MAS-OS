"""Real file I/O sandboxed to WORKSPACE."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import WORKSPACE, ToolResult, err, ok


class FileTool:
    """Reads, writes, lists, and parses files within the workspace sandbox."""

    def read(self, path: str) -> ToolResult:
        try:
            full = self._resolve(path)
            if not full.exists():
                return err("file_read", f"File not found: {path}")
            if not full.is_file():
                return err("file_read", f"Not a file: {path}")
            content = full.read_text(encoding="utf-8", errors="replace")
            return ok("file_read", content,
                      path=str(full), size=len(content), lines=content.count("\n") + 1)
        except PermissionError:
            return err("file_read", f"Permission denied: {path}")
        except Exception as exc:
            return err("file_read", str(exc))

    def write(self, path: str, content: str) -> ToolResult:
        try:
            full = self._resolve(path)
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            return ok("file_write", f"Written {len(content):,} chars to {path}",
                      path=str(full), size=len(content))
        except PermissionError:
            return err("file_write", f"Permission denied: {path}")
        except Exception as exc:
            return err("file_write", str(exc))

    def append(self, path: str, content: str) -> ToolResult:
        try:
            full = self._resolve(path)
            full.parent.mkdir(parents=True, exist_ok=True)
            with open(full, "a", encoding="utf-8") as f:
                f.write(content)
            return ok("file_append", f"Appended {len(content):,} chars to {path}")
        except Exception as exc:
            return err("file_append", str(exc))

    def list_dir(self, path: str = ".", pattern: str = "*") -> ToolResult:
        try:
            full = self._resolve(path)
            if not full.exists():
                return err("file_list", f"Directory not found: {path}")
            items: List[Dict[str, Any]] = []
            for p in sorted(full.rglob(pattern))[:100]:
                try:
                    stat = p.stat()
                    items.append({
                        "name":     p.name,
                        "path":     str(p.relative_to(WORKSPACE)),
                        "size":     stat.st_size,
                        "type":     "dir" if p.is_dir() else "file",
                    })
                except OSError:
                    continue
            return ok("file_list", json.dumps(items, indent=2), count=len(items))
        except Exception as exc:
            return err("file_list", str(exc))

    def read_csv(self, path: str, max_rows: int = 200) -> ToolResult:
        """Read a CSV and return the first *max_rows* rows as JSON."""
        try:
            full = self._resolve(path)
            if not full.exists():
                return err("file_csv", f"File not found: {path}")
            rows: List[Dict[str, Any]] = []
            with open(full, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows.append(dict(row))
            if not rows:
                return ok("file_csv", "[]", total_rows=0, columns=[])
            sample = json.dumps(rows[:10], indent=2)
            return ok("file_csv", sample,
                      total_rows=len(rows),
                      columns=list(rows[0].keys()),
                      path=str(full))
        except Exception as exc:
            return err("file_csv", str(exc))

    def _resolve(self, path: str) -> Path:
        """Resolve *path* relative to WORKSPACE; allow absolute paths as-is."""
        p = Path(path)
        return p if p.is_absolute() else WORKSPACE / path
