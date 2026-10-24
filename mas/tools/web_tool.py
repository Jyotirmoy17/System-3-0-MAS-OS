"""Real web search via DuckDuckGo (no API key)."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from .base import ToolResult, err, ok

_UA = "MAS-OS/1.0 (+https://github.com/Jyotirmoy17/System-3-0-MAS-OS)"
_DDG_API  = "https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"
_DDG_LITE = "https://lite.duckduckgo.com/lite/?q={q}"


class WebTool:
    """DuckDuckGo search and URL fetching."""

    def search(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            enc = urllib.parse.quote(query)
            req = urllib.request.Request(
                _DDG_API.format(q=enc),
                headers={"User-Agent": _UA},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read())

            results: List[Dict[str, Any]] = []

            abstract = data.get("Abstract", "")
            if abstract:
                results.append({
                    "title":   data.get("Heading", ""),
                    "snippet": abstract,
                    "url":     data.get("AbstractURL", ""),
                    "source":  "DuckDuckGo Abstract",
                })

            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title":   topic["Text"][:80],
                        "snippet": topic["Text"],
                        "url":     topic.get("FirstURL", ""),
                        "source":  "DuckDuckGo Related",
                    })

            if not results:
                results = self._scrape_lite(query, max_results)

            output = json.dumps(results[:max_results], indent=2)
            return ok("web_search", output, query=query, count=len(results[:max_results]))

        except Exception as exc:
            return err("web_search", f"Search failed: {exc}")

    def fetch(self, url: str, max_chars: int = 3000) -> ToolResult:
        """Fetch a URL and return extracted plain text."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Strip scripts, styles, tags
            text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>",   " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            return ok("web_fetch", text[:max_chars], url=url, total_length=len(text))
        except Exception as exc:
            return err("web_fetch", str(exc))

    def _scrape_lite(self, query: str, n: int) -> List[Dict[str, Any]]:
        """Fallback: scrape DuckDuckGo Lite HTML for snippets."""
        try:
            enc  = urllib.parse.quote(query)
            req  = urllib.request.Request(
                _DDG_LITE.format(q=enc),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            snippets = re.findall(
                r'class="result-snippet"[^>]*>(.*?)</td>', html, re.DOTALL
            )
            results = []
            for s in snippets[:n]:
                text = re.sub(r"<[^>]+>", "", s).strip()
                if text:
                    results.append({"title": text[:80], "snippet": text, "url": "", "source": "DDG Lite"})
            return results
        except Exception:
            return []
