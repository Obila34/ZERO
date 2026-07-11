"""Tools: duration parsing, timers firing, registry allow-list, 3-tier router."""
from __future__ import annotations

import time

import pytest

from zero.events import Event, EventBus
from zero.tools.base import Tool, ToolContext
from zero.tools.builtin import (
    RecallTool, RememberTool, ReminderTool, TimeTool, TimerTool,
    parse_duration_s,
)
from zero.tools.registry import ToolRegistry
from zero.tools.router import ToolAwareLLM
from zero.tools.timers import TimerManager, humanize_s


# ── duration parsing ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,seconds", [
    ("5 minutes", 300), ("90 seconds", 90), ("1 hour", 3600),
    ("1 hour 30 minutes", 5400), ("ten minutes", 600), ("an hour", 3600),
    ("2m", 120), ("45s", 45), ("half an hour", 1800),
])
def test_parse_duration(text, seconds):
    assert parse_duration_s(text) == seconds


@pytest.mark.parametrize("text", ["", "tomorrow", "the weekend", "no time here"])
def test_parse_duration_rejects(text):
    assert parse_duration_s(text) is None


def test_humanize():
    assert humanize_s(90) == "1 minute 30 seconds"
    assert humanize_s(3600) == "1 hour"
    assert humanize_s(45) == "45 seconds"


# ── timers fire onto the bus ─────────────────────────────────────────────────
def test_timer_fires_event():
    bus = EventBus()
    tm = TimerManager(bus)
    tm.set(0.6, "tea", kind="timer")
    assert len(tm.active()) == 1
    deadline = time.time() + 5.0
    events = []
    while not events and time.time() < deadline:
        events = bus.drain()
        time.sleep(0.05)
    assert events and events[0].kind == "timer" and "tea" in events[0].text
    assert tm.active() == []


def test_timer_cancel():
    bus = EventBus()
    tm = TimerManager(bus)
    tid = tm.set(30, "later")
    assert tm.cancel(tid) is True
    assert tm.active() == []
    assert tm.cancel(tid) is False


# ── registry / allow-list ────────────────────────────────────────────────────
def test_allowlist_blocks_unlisted():
    reg = ToolRegistry(allow=["time"])
    assert reg.register(TimeTool()) is True
    assert reg.register(RememberTool()) is False
    assert reg.names() == ["time"]
    assert "time:" in reg.spec_block()


# ── fake LLM for router tests ────────────────────────────────────────────────
class FakeLLM:
    """Replays scripted replies, chunked to exercise the sniffing logic."""

    def __init__(self, *replies: str, chunk: int = 7):
        self.replies = list(replies)
        self.chunk = chunk
        self.calls: list[list] = []

    def stream(self, messages):
        self.calls.append(list(messages))
        reply = self.replies.pop(0) if self.replies else ""
        for i in range(0, len(reply), self.chunk):
            yield reply[i:i + self.chunk]


def _registry(memory=None):
    bus = EventBus()
    tm = TimerManager(bus)
    reg = ToolRegistry()
    for t in (TimeTool(), TimerTool(tm), ReminderTool(tm),
              RememberTool(), RecallTool()):
        reg.register(t)
    return reg, bus, tm


def _collect(gen):
    return "".join(gen)


def test_router_plain_chat_passthrough():
    reg, _, _ = _registry()
    llm = FakeLLM("Nairobi is the capital of Kenya.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream([{"role": "user", "content": "capital of Kenya?"}]))
    assert out == "Nairobi is the capital of Kenya."
    assert len(llm.calls) == 1


def test_router_tier1_timer_skips_llm():
    reg, _, tm = _registry()
    llm = FakeLLM("should never be called")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "Set a timer for 5 minutes"}]))
    assert "5 minutes" in out and "Timer set" in out
    assert llm.calls == []          # no LLM round-trip
    assert len(tm.active()) == 1


def test_router_tier1_time():
    reg, _, _ = _registry()
    router = ToolAwareLLM(FakeLLM("no"), reg)
    out = _collect(router.stream([{"role": "user", "content": "What time is it?"}]))
    assert "It's" in out


def test_router_tier1_reminder():
    reg, _, tm = _registry()
    router = ToolAwareLLM(FakeLLM("no"), reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "Remind me to check the oven in 10 minutes"}]))
    assert "check the oven" in out and "10 minutes" in out
    assert len(tm.active()) == 1


def test_router_explicit_websearch_is_forced():
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool(
        "http://x", fetch=lambda q: [{"title": "R", "content": f"about {q}"}]))
    llm = FakeLLM("France won two to one.")   # the rephrase pass
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "search the web for who won the world cup"}]))
    assert out == "France won two to one."
    # The web result (carrying the query) was handed to the rephrase call.
    assert any("who won the world cup" in str(m.get("content", "")).lower()
               for m in llm.calls[-1])


def test_router_websearch_falls_through_without_the_tool():
    reg, _, _ = _registry()   # web_search NOT registered
    llm = FakeLLM("I can't reach the web right now.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "search the web for mars facts"}]))
    assert out == "I can't reach the web right now."   # normal LLM path, no crash


def test_router_auto_searches_live_question_without_keyword():
    # "who won ..." needs live info -> search WITHOUT the user saying "search".
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool(
        "http://x", fetch=lambda q: [{"title": "ESPN", "content": f"result for {q}"}]))
    llm = FakeLLM("France beat Morocco two-nil.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "who won the Morocco France game yesterday?"}]))
    assert out == "France beat Morocco two-nil."
    assert any("morocco france game" in str(m.get("content", "")).lower()
               for m in llm.calls[-1])


@pytest.mark.parametrize("text", [
    "how are you today?",
    "what's your name?",
    "do you like music?",
    "how was your day",
])
def test_router_casual_questions_dont_auto_search(text):
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool("http://x", fetch=lambda q: [{"title": "X"}]))
    llm = FakeLLM("Just chatting along.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream([{"role": "user", "content": text}]))
    assert out == "Just chatting along."     # plain chat, no forced search


def test_router_string_args_are_coerced_not_crashed():
    # Regression: the model emitted {"tool":"web_search","args":"messi goals"} —
    # args.get() on a bare string raised "'str' object has no attribute 'get'".
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool(
        "http://x", fetch=lambda q: [{"title": "T", "content": f"re {q}"}]))
    llm = FakeLLM('{"tool": "web_search", "args": "messi world cup goals"}',
                  "Messi has plenty of them.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream([{"role": "user", "content": "tell me about messi"}]))
    assert out == "Messi has plenty of them."     # no crash, rephrased result
    assert any("messi world cup goals" in str(m.get("content", "")).lower()
               for m in llm.calls[-1])


def test_router_check_online_phrasing_is_forced():
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool(
        "http://x", fetch=lambda q: [{"title": "R", "content": f"re {q}"}]))
    llm = FakeLLM("Here's what's out there.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "check online for the world cup schedule"}]))
    assert out == "Here's what's out there."
    assert any("world cup schedule" in str(m.get("content", "")).lower()
               for m in llm.calls[-1])


def test_router_tonight_question_auto_searches():
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool(
        "http://x", fetch=lambda q: [{"title": "R", "content": f"re {q}"}]))
    llm = FakeLLM("Two matches tonight.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "which teams are playing tonight?"}]))
    assert out == "Two matches tonight."


def test_router_shared_memory_question_is_not_web_searched():
    # "what did we talk about yesterday" is OUR memory — asking the internet
    # would be absurd. Must fall through to the normal LLM+memory path.
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool("http://x", fetch=lambda q: [{"title": "X"}]))
    llm = FakeLLM("We talked about planets, remember?")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "what did we talk about yesterday?"}]))
    assert out == "We talked about planets, remember?"


def test_router_uncertainty_rescue_searches_instead_of_shrugging():
    # "Not in my knowledge base" -> the model admits it -> we search the web and
    # answer from live results instead of speaking the shrug.
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool(
        "http://x", fetch=lambda q: [{"title": "Wiki", "content": f"facts on {q}"}]))
    llm = FakeLLM(
        "I don't have that on the tip of my tongue, honestly.",   # the shrug
        "Season 2 lands this October.",                            # the rescue
    )
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "when does Black Clover season 2 release"}]))
    assert out == "Season 2 lands this October."
    assert "tip of my tongue" not in out                 # shrug never spoken
    # The rescue searched the USER's question.
    assert any("black clover season 2" in str(m.get("content", "")).lower()
               for m in llm.calls[-1])


def test_router_no_rescue_for_statements_or_smalltalk():
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool("http://x", fetch=lambda q: [{"title": "X"}]))
    llm = FakeLLM("I don't really know her, to be honest.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "I met Maria at the market."}]))   # statement
    assert out == "I don't really know her, to be honest."   # spoken, no search


def test_router_embedded_tool_json_is_executed_not_spoken():
    # The model wraps a tool call in chatter — the JSON must be executed and
    # phrased, never read out loud.
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool(
        "http://x", fetch=lambda q: [{"title": "ESPN", "content": "France 2-0"}]))
    llm = FakeLLM(
        'Let me check on that for you. '
        '{"tool": "web_search", "args": {"query": "france morocco score"}}',
        "France took it two-nil.",
    )
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "I wonder about that Morocco game."}]))
    assert '{"tool"' not in out                       # JSON never spoken
    assert "Let me check on that for you." in out     # natural lead-in kept
    assert "France took it two-nil." in out           # phrased result spoken


def test_router_bare_look_up_is_not_web_forced():
    # "look up" alone is ambiguous (memory recall) — must NOT force web search.
    from zero.tools.websearch import WebSearchTool

    reg, _, _ = _registry()
    reg.register(WebSearchTool("http://x", fetch=lambda q: [{"title": "R"}]))
    llm = FakeLLM("Sure, here's what I recall.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream(
        [{"role": "user", "content": "look up what I told you about my sister"}]))
    assert out == "Sure, here's what I recall."


def test_router_tier2_json_tool_call_and_rephrase():
    reg, _, tm = _registry()
    llm = FakeLLM(
        '{"tool": "timer", "args": {"duration": "3 minutes", "label": "eggs"}}',
        "Three minutes for your eggs — timer's running.",
    )
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream([{"role": "user", "content": "boil eggs timing"}]))
    assert out == "Three minutes for your eggs — timer's running."
    assert len(tm.active()) == 1
    # Follow-up call carried the tool result back to the model.
    assert any("Tool result" in str(m.get("content", ""))
               for m in llm.calls[1])


def test_router_tier2_unknown_tool_is_graceful():
    reg, _, _ = _registry()
    llm = FakeLLM('{"tool": "nuke", "args": {}}', "I can't do that one.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream([{"role": "user", "content": "launch"}]))
    assert out == "I can't do that one."
    assert any("don't have a tool called nuke" in str(m.get("content", ""))
               for m in llm.calls[1])


def test_router_malformed_json_is_spoken_not_silent():
    reg, _, _ = _registry()
    llm = FakeLLM('{"tool": broken json here')
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream([{"role": "user", "content": "hi"}]))
    assert out == '{"tool": broken json here'   # spoken as-is, never mute


def test_router_brace_prose_not_treated_as_tool():
    reg, _, _ = _registry()
    llm = FakeLLM("{ that's an odd way to open a sentence but fine.")
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream([{"role": "user", "content": "hm"}]))
    assert out.startswith("{ that's an odd way")


def test_router_llm_death_still_speaks_result():
    reg, _, _ = _registry()
    llm = FakeLLM('{"tool": "time", "args": {}}')   # no second reply: LLM "dies"
    router = ToolAwareLLM(llm, reg)
    out = _collect(router.stream([{"role": "user", "content": "clock?"}]))
    assert "It's" in out                              # raw tool result spoken


# ── memory-backed tools ──────────────────────────────────────────────────────
class DictMemory:
    def __init__(self):
        self.data = {}

    def remember(self, key, value):
        self.data[key] = value

    def facts(self):
        return dict(self.data)


def test_remember_and_recall_roundtrip():
    mem = DictMemory()
    ctx = ToolContext(memory=mem, person_name="David")
    out = RememberTool().run({"fact": "the wifi password is hunter2"}, ctx)
    assert "remember" in out.lower()
    out = RecallTool().run({"query": "wifi"}, ctx)
    assert "hunter2" in out


def test_recall_empty():
    out = RecallTool().run({"query": "quantum"}, ToolContext(memory=DictMemory()))
    assert "don't remember" in out


def test_tool_error_is_contained():
    class Boom(Tool):
        name = "boom"
        description = "explodes"

        def run(self, args, ctx):
            raise RuntimeError("kapow")

    out = Boom().safe_run({}, ToolContext())
    assert "boom" in out and "kapow" in out
