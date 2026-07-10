"""Web search tool: summarizes hits, honest when offline, never raises."""
from __future__ import annotations

from zero.tools.base import ToolContext
from zero.tools.websearch import WebSearchTool


def _tool(fetch):
    return WebSearchTool("http://x/search", fetch=fetch, max_results=2)


def test_summarizes_top_results():
    hits = [
        {"title": "Mars", "content": "the fourth planet from the sun"},
        {"title": "Mars facts", "content": "has two moons, Phobos and Deimos"},
        {"title": "extra", "content": "should be dropped by max_results"},
    ]
    out = _tool(lambda q: hits).run({"query": "mars"}, ToolContext())
    assert "Mars" in out and "fourth planet" in out
    assert "should be dropped" not in out          # max_results=2


def test_offline_is_honest_not_invented():
    def boom(q):
        raise ConnectionError("tunnel down")

    out = _tool(boom).run({"query": "weather"}, ToolContext())
    assert "can't reach the internet" in out.lower()


def test_empty_results():
    out = _tool(lambda q: []).run({"query": "asdfqwer"}, ToolContext())
    assert "didn't find anything" in out.lower()


def test_missing_query():
    out = _tool(lambda q: []).run({}, ToolContext())
    assert "didn't catch" in out.lower()


def test_safe_run_never_raises():
    # Even a fetch returning garbage must come back as a sentence, not an error.
    out = _tool(lambda q: [{"nonsense": 1}]).safe_run({"query": "x"}, ToolContext())
    assert isinstance(out, str) and out
