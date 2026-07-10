"""Web search — ZERO's window onto the live internet.

Offline-first, like the rest of ZERO: it queries a configurable search endpoint
(a SearXNG ``/search`` JSON API, or any service returning
``{"results": [{"title", "url", "content"}...]}``) and hands the top hits back to
the router, which lets the LLM phrase them in ZERO's own voice. When the endpoint
is unreachable it SAYS so plainly instead of inventing an answer — the persona's
"never make things up" rule extends to the web.

Kept model-agnostic and injectable (``fetch=``) so it's testable offline and you
can point it at whatever provider you run.
"""
from __future__ import annotations

from zero.tools.base import Tool, ToolContext
from zero.utils.logging import get_logger

log = get_logger("tools.websearch")


class WebSearchTool(Tool):
    name = "web_search"
    description = ("Look something up on the live internet — current events, "
                   "facts you're unsure of, weather, prices.")
    parameters = {"query": "what to search for"}

    def __init__(self, url: str, *, timeout: float = 8.0, max_results: int = 3,
                 fetch=None):
        self._url = url
        self._timeout = float(timeout)
        self._max = max(1, int(max_results))
        self._fetch = fetch or self._http_fetch  # injectable for tests / offline

    def _http_fetch(self, query: str) -> list[dict]:
        import requests  # lazy; already a project dependency

        r = requests.get(self._url,
                         params={"q": query, "format": "json"},
                         timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        return list(data.get("results") or [])

    def run(self, args: dict, ctx: ToolContext) -> str:
        query = str(args.get("query", "") or "").strip()
        if not query:
            return "I didn't catch what to search for."
        try:
            results = self._fetch(query) or []
        except Exception as e:  # offline / endpoint down — be honest, don't invent
            log.warning("web search failed: %s", e)
            return ("I can't reach the internet right now, so I can't look that "
                    "up — I'll go on what I already know.")
        if not results:
            return f"I searched the web for {query} but didn't find anything useful."
        snippets = []
        for r in results[: self._max]:
            title = str(r.get("title") or "").strip()
            content = str(r.get("content") or r.get("snippet") or "").strip()
            line = f"{title}: {content}" if title and content else (title or content)
            if line:
                snippets.append(line[:280])
        if not snippets:
            return f"I found results for {query} but couldn't read them clearly."
        return f"Web results for {query}: " + " | ".join(snippets)
