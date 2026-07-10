"""ToolAwareLLM — wraps the LLM so replies can carry actions.

Three tiers, cheapest first:
  1. Regex intents ("set a timer for five minutes", "what time is it") —
     no LLM round-trip at all, the tool result is the reply.
  2. LLM tool call — the system prompt (spec_block) teaches the model to emit
     one line of JSON. The stream is sniffed: if the first non-space character
     is '{', the whole reply is buffered and parsed instead of spoken; the
     tool runs; a follow-up LLM call phrases the outcome naturally. If the
     JSON is malformed, the buffered text is spoken as-is — never silence.
  3. Plain chat — everything else streams through untouched.

Implements the same .stream()/.warmup() surface as OllamaLLM, so main.py's
streaming/barge-in pipeline doesn't know tools exist.
"""
from __future__ import annotations

import json
import re

from zero.tools.base import ToolContext
from zero.tools.builtin import parse_duration_s
from zero.tools.registry import ToolRegistry
from zero.utils.logging import get_logger

log = get_logger("tools.router")

_SNIFF_CHARS = 24  # enough to see '{"tool"' through leading whitespace/fences

# Tier-1 intents. Deliberately narrow: high-precision phrasings only — anything
# ambiguous falls through to the LLM, which can still tool-call.
_T1_TIMER = re.compile(
    r"\b(?:set|start)\b.{0,20}?\btimer\b(?:\s*(?:for|of))?\s*(?P<dur>.{2,40})",
    re.IGNORECASE)
_T1_REMIND = re.compile(
    r"\bremind me\b(?:\s+to\b|\s+about\b)?\s*(?P<about>.+?)\s+in\s+(?P<dur>.+)$",
    re.IGNORECASE)
_T1_TIME = re.compile(
    r"\bwhat(?:'s| is)? (?:the )?(?:time|date|day)\b|\bwhat time is it\b",
    re.IGNORECASE)
# Explicit "search the web for X" / "google X" — captures the query after the
# command. Deliberately narrow: bare "look up" is left OUT (it's ambiguous with
# recalling from memory); only web/internet/online/google count.
_WEBSEARCH_RE = re.compile(
    r"\b(?:search(?:\s+(?:the|on|from|through))?\s+(?:the\s+)?(?:web|internet|online)"
    r"|google|look\s+.+\s+up\s+online)\b"
    r"\s*(?:for|about)?\s*(?P<q>.*)$",
    re.IGNORECASE)
# A QUESTION that plainly needs LIVE information — force a web search so ZERO
# answers from the internet instead of stale training data or a deflection, WITHOUT
# the user saying "search". Requires a factual signal (a score, a release, a price,
# a recent year...) on top of question shape, so casual talk ("how are you today")
# never triggers a search.
_Q_SHAPE = re.compile(
    r"^\s*(?:who|what|whats|when|where|which|whose|how|is|are|was|were|did|"
    r"does|do|has|have|will)\b", re.IGNORECASE)
_LIVE_SIGNAL = re.compile(
    r"\b(won|win|winning|winner|scored?|finals?|semi[- ]?finals?|"
    r"quarter[- ]?finals?|standings?|schedule|fixtures?|playing|"
    r"release\s+date|comes?\s+out|premieres?|out\s+yet|"
    r"latest|current(?:ly)?|price|costs?|news|headlines?|happened|results?|"
    r"weather|forecast|stock|20(?:2[4-9]|3\d))\b", re.IGNORECASE)


class ToolAwareLLM:
    def __init__(self, llm, registry: ToolRegistry, context_provider=None):
        self._llm = llm
        self._registry = registry
        # Callable returning a fresh ToolContext each call (person changes).
        self._ctx = context_provider or (lambda: ToolContext())

    # ── the LLM surface main.py expects ──────────────────────────────────────
    def warmup(self, messages=None):
        warmup = getattr(self._llm, "warmup", None)
        if callable(warmup):
            return warmup(messages)

    def stream(self, messages):
        user_text = self._last_user_text(messages)

        # Tier 1: instant intents, no LLM round-trip.
        t1 = self._tier1(user_text)
        if t1 is not None:
            log.info("tier-1 tool reply: %r", t1[:80])
            yield t1
            return

        # Web search, forced two ways so the model's flaky tool-calling isn't in
        # the loop: an EXPLICIT "search the web for X", or a live-info QUESTION
        # ("who won...", "when does X release") that plainly needs the internet.
        # Either way we run the tool and let the LLM phrase the result.
        wq = self._forced_websearch(user_text) or self._auto_websearch(user_text)
        if wq is not None:
            log.info("web_search (query=%r)", wq[:80])
            yield from self._run_and_rephrase(
                messages, "", {"tool": "web_search", "args": {"query": wq}})
            return

        # Tier 2/3: stream and sniff for a JSON tool call.
        stream = self._llm.stream(messages)
        buffered = ""
        for chunk in stream:
            buffered += chunk
            probe = buffered.lstrip()
            if probe and not probe.startswith("{"):
                # Plain chat: flush what we held, then pass through.
                yield buffered
                yield from stream
                return
            if len(probe) >= _SNIFF_CHARS and '"tool"' not in probe[:_SNIFF_CHARS]:
                yield buffered
                yield from stream
                return
        # Stream ended while still '{'-shaped: try the tool path.
        reply = buffered.strip()
        if not reply:
            return
        call = self._parse_call(reply)
        if call is None:
            yield buffered  # malformed JSON — speak it rather than go mute
            return
        yield from self._run_and_rephrase(messages, reply, call)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _last_user_text(messages) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                # Strip the ephemeral perception notes appended by main.py.
                return content.split("\n\n(", 1)[0]
        return ""

    def _tier1(self, text: str) -> str | None:
        if not text:
            return None
        ctx = self._ctx()
        m = _T1_REMIND.search(text)
        if m and self._registry.get("reminder") is not None:
            if parse_duration_s(m.group("dur")):
                return self._registry.get("reminder").safe_run(
                    {"about": m.group("about"), "duration": m.group("dur")}, ctx)
        m = _T1_TIMER.search(text)
        if m and self._registry.get("timer") is not None:
            if parse_duration_s(m.group("dur")):
                return self._registry.get("timer").safe_run(
                    {"duration": m.group("dur")}, ctx)
        if _T1_TIME.search(text) and self._registry.get("time") is not None:
            return self._registry.get("time").safe_run({}, ctx)
        return None

    def _forced_websearch(self, text: str) -> str | None:
        """The query from an explicit 'search the web for X' request, when the
        web_search tool is available. None when there's no web-search intent, no
        tool, or no inline query (those fall through — the LLM can still tool-call
        now that the tool exists, e.g. 'search the web and tell me' about the
        previous turn)."""
        if not text or self._registry.get("web_search") is None:
            return None
        m = _WEBSEARCH_RE.search(text)
        if not m:
            return None
        q = m.group("q").strip(" ?.!,\"'")
        # Trim trailing niceties so the query is just the thing to look up.
        q = re.sub(r"\b(?:and\s+)?(?:tell|let|show)\s+me\b.*$", "", q,
                   flags=re.IGNORECASE)
        q = re.sub(r"\b(?:for me|please|right now)\b", "", q, flags=re.IGNORECASE)
        q = q.strip(" ?.!,\"'")
        return q if len(q) >= 3 else None

    def _auto_websearch(self, text: str) -> str | None:
        """The query to look up when the utterance is a QUESTION that clearly
        needs live information (scores, releases, prices, current events) — so the
        user never has to say 'search the web' for those. None otherwise, or when
        the web tool isn't available."""
        if not text or self._registry.get("web_search") is None:
            return None
        is_question = text.strip().endswith("?") or bool(_Q_SHAPE.search(text))
        if is_question and _LIVE_SIGNAL.search(text):
            return text.strip(" ?.!,")
        return None

    @staticmethod
    def _parse_call(reply: str) -> dict | None:
        # Tolerate code fences and trailing prose after the JSON line.
        cleaned = reply.strip().strip("`").strip()
        cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE)
        start = cleaned.find("{")
        if start < 0:
            return None
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:i + 1])
                    except ValueError:
                        return None
                    if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
                        return obj
                    return None
        return None

    def _run_and_rephrase(self, messages, raw_reply: str, call: dict):
        name = call["tool"]
        tool = self._registry.get(name)
        if tool is None:
            result = f"I don't have a tool called {name}."
        else:
            # The model sometimes emits "args" as a bare string (the query)
            # instead of an object — coerce it so args.get() never explodes.
            raw_args = call.get("args")
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                args = {"query": raw_args}
            else:
                args = {}
            result = tool.safe_run(args, self._ctx())
            log.info("tool %s -> %r", name, result[:100])
        # Second pass: let the model phrase the outcome in its own voice.
        followup = [
            *messages,
            {"role": "assistant", "content": raw_reply},
            {"role": "user", "content":
                f"(Tool result: {result}) Tell the user this outcome in one "
                "natural sentence. Do not mention tools or JSON."},
        ]
        spoke = False
        for chunk in self._llm.stream(followup):
            spoke = True
            yield chunk
        if not spoke:  # LLM died mid-turn — the raw result still gets spoken
            yield result
