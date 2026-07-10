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


def test_non_dict_result_entry_is_skipped_not_crashed():
    # Regression: SearXNG can yield a stray string in results; '.get' on it used
    # to raise "'str' object has no attribute 'get'" and kill the turn.
    hits = ["a stray string", {"title": "Mars", "content": "fourth planet"}]
    out = _tool(lambda q: hits).run({"query": "mars"}, ToolContext())
    assert "Mars" in out and "fourth planet" in out


def test_full_payload_with_answers_and_results():
    # Real SearXNG shape: a dict with direct answers (great for factual asks)
    # plus result snippets. Answers surface first.
    payload = {
        "answers": ["France won 2-0"],
        "results": [{"title": "ESPN", "content": "match report"}],
    }
    out = _tool(lambda q: payload).run({"query": "world cup"}, ToolContext())
    assert "France won 2-0" in out


def test_dict_answers_form_is_handled():
    # Some engines return answers as dicts, not strings.
    payload = {"answers": [{"answer": "Messi has 13 World Cup goals"}], "results": []}
    out = _tool(lambda q: payload).run({"query": "messi goals"}, ToolContext())
    assert "13 World Cup goals" in out
