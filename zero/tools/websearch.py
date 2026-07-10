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
    description = ("Search the live internet for anything you don't know or "
                   "aren't sure of — current events, news, sports scores, "
                   "release dates, weather, prices, recent facts. Prefer this "
                   "over guessing or telling the person to look it up.")
    parameters = {"query": "what to search for"}

    def __init__(self, url: str, *, timeout: float = 8.0, max_results: int = 3,
                 fetch=None):
        self._url = url
        self._timeout = float(timeout)
        self._max = max(1, int(max_results))
        self._fetch = fetch or self._http_fetch  # injectable for tests / offline

    def _http_fetch(self, query: str):
        import requests  # lazy; already a project dependency

        r = requests.get(self._url,
                         params={"q": query, "format": "json"},
                         timeout=self._timeout)
        r.raise_for_status()
        return r.json()  # full payload: results + answers + infoboxes + ...

    def _summarize(self, payload) -> list[str]:
        """Pull the best short lines out of a SearXNG payload, defensively:
        direct ANSWERS first (they settle factual questions), then result
        snippets. Tolerates a bare results list (the injected/test contract) and
        skips any entry that isn't the expected shape — a stray string in the
        results must never crash a turn."""
        results, answers = [], []
        if isinstance(payload, dict):
            results = payload.get("results") or []
            answers = payload.get("answers") or []
        elif isinstance(payload, list):
            results = payload
        lines: list[str] = []
        for a in answers:  # answers may be strings OR dicts, depending on engine
            text = a.get("answer") if isinstance(a, dict) else a
            text = str(text or "").strip()
            if text:
                lines.append(text[:280])
        for r in results:
            if not isinstance(r, dict):
                continue  # a non-dict result: skip it, don't explode
            title = str(r.get("title") or "").strip()
            content = str(r.get("content") or r.get("snippet") or "").strip()
            line = (f"{title}: {content}" if title and content
                    else (title or content)).strip()
            if line:
                lines.append(line[:280])
            if len(lines) >= self._max:
                break
        return lines[: self._max]

    def run(self, args: dict, ctx: ToolContext) -> str:
        # Be forgiving about how args arrive: the model may pass the query as a
        # bare string rather than {"query": ...}.
        if isinstance(args, str):
            args = {"query": args}
        elif not isinstance(args, dict):
            args = {}
        query = str(args.get("query", "") or "").strip()
        if not query:
            return "I didn't catch what to search for."
        try:
            payload = self._fetch(query)
        except Exception as e:  # offline / endpoint down — be honest, don't invent
            log.warning("web search failed: %s", e)
            return ("I can't reach the internet right now, so I can't look that "
                    "up — I'll go on what I already know.")
        snippets = self._summarize(payload)
        if not snippets:
            return f"I searched the web for {query} but didn't find anything useful."
        return f"Web results for {query}: " + " | ".join(snippets)
