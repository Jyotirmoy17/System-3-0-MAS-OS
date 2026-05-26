"""Persistent key-value memory across sessions."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .base import WORKSPACE, ToolResult, err, ok


class MemoryTool:
    """
    Disk-backed key-value store.

    Data is stored in WORKSPACE/memory/global.json.  Every write is
    immediately flushed so the data survives a kernel restart.
    """

    def __init__(self, namespace: str = "global") -> None:
        self._dir  = WORKSPACE / "memory"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"{namespace}.json"
        self._data: dict = self._load()

    def remember(
        self,
        key:       str,
        value:     Any,
        ttl_hours: Optional[float] = None,
    ) -> ToolResult:
        entry: dict = {"value": value, "stored_at": time.time()}
        if ttl_hours is not None:
            entry["expires_at"] = time.time() + ttl_hours * 3600
        self._data[key] = entry
        self._flush()
        return ok("memory_store", f"Stored: {key}", key=key)

    def recall(self, key: str) -> ToolResult:
        entry = self._data.get(key)
        if entry is None:
            return err("memory_recall", f"No memory found for key: '{key}'")
        # TTL check
        if "expires_at" in entry and time.time() > entry["expires_at"]:
            del self._data[key]
            self._flush()
            return err("memory_recall", f"Memory expired: '{key}'")
        age_h = (time.time() - entry["stored_at"]) / 3600
        return ok("memory_recall", str(entry["value"]), key=key, age_hours=round(age_h, 2))

    def list_keys(self, prefix: str = "") -> ToolResult:
        keys = sorted(k for k in self._data if k.startswith(prefix))
        return ok("memory_list", json.dumps(keys), count=len(keys))

    def forget(self, key: str) -> ToolResult:
        if key in self._data:
            del self._data[key]
            self._flush()
        return ok("memory_forget", f"Forgotten: {key}")

    def _load(self) -> dict:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _flush(self) -> None:
        self._file.write_text(
            json.dumps(self._data, indent=2, default=str),
            encoding="utf-8",
        )
