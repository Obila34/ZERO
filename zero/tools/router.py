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
    r"\b(?:(?:search|check|browse|look)(?:\s+(?:it|this|that))?"
    r"(?:\s+(?:the|on|from|through|up))?\s+(?:the\s+)?(?:web|internet|online)"
    r"|google|look\s+.+\s+up\s+online)\b"
    r"\s*(?:for|about|and\s+(?:tell|find)\s+(?:me|out))?\s*(?P<q>.*)$",
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
    r"weather|forecast|stock|champions?|elections?|"
    r"today|tonight|yesterday|tomorrow|this\s+(?:week|month|year|season)|"
    r"20(?:2[4-9]|3\d))\b", re.IGNORECASE)
# ...but a question about OUR OWN shared past ("what did we talk about
# yesterday?") is a MEMORY question — asking the internet would be absurd.
_MEMORY_SHAPE = re.compile(
    r"\b(?:we|us|our|remember|talked|told|said|conversation|discussed)\b",
    re.IGNORECASE)
# The reply-side safety net ("uncertainty rescue"): the persona instructs the
# model to admit not knowing in recognisable phrasings. If a reply to a QUESTION
# opens with one of these, we suppress it, web-search the question instead, and
# answer from live results — that's "not in my knowledge base -> search" driven
# by the model's own admission, no keyword list required.
_UNSURE_RE = re.compile(
    r"\b(?:i\s+(?:honestly\s+|really\s+)?(?:don'?t|do\s+not)\s+"
    r"(?:know|have|recall|remember)|i'?m\s+not\s+(?:really\s+)?sure|"
    r"tip\s+of\s+my\s+tongue|i\s+have\s+no\s+idea|i\s+wish\s+i\s+(?:knew|could))",
    re.IGNORECASE)
_RESCUE_SNIFF_CHARS = 120   # absolute cap on how much of the reply to inspect
_RESCUE_MIN_RELEASE = 20    # a sentence this long ending cleanly = not a shrug
# A completed sentence (not counting a tiny interjection like "Hmm.") with no
# unsure marker means the reply is a real answer — release it to TTS
# IMMEDIATELY instead of holding for the full sniff window. Holding short
# replies to stream-end was adding seconds of silence on question turns.
_SENT_END_RE = re.compile(r"[.!?…]")
_JSON_MARKER = '{"tool'     # an embedded tool call inside spoken prose


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
        # While the reply still *could* be an admission of not knowing, hold it
        # back (up to _RESCUE_SNIFF_CHARS) — if it opens with "I don't know...",
        # we search the web instead of speaking the shrug.
        rescue_armed = self._rescue_armed(user_text)
        stream = self._llm.stream(messages)
        buffered = ""
        for chunk in stream:
            buffered += chunk
            probe = buffered.lstrip()
            if probe and not probe.startswith("{"):
                # Prose. Uncertainty rescue: an "I don't know" opener to a real
                # question becomes a web search, not a spoken shrug.
                if rescue_armed and _UNSURE_RE.search(probe):
                    self._close(stream)
                    log.info("uncertainty rescue -> web_search(%r)",
                             user_text[:80])
                    yield from self._run_and_rephrase(
                        messages, "",
                        {"tool": "web_search",
                         "args": {"query": user_text.strip(" ?.!,")}})
                    return
                if rescue_armed and len(probe) < _RESCUE_SNIFF_CHARS:
                    # Early release: a real sentence completed with no unsure
                    # marker = a genuine answer. Ship it to TTS NOW — holding
                    # short replies to stream-end cost seconds of silence.
                    # (Tiny interjections like "Hmm." keep holding: the shrug
                    # often follows them.)
                    if not _SENT_END_RE.search(probe, _RESCUE_MIN_RELEASE - 1):
                        continue  # not proven an answer yet — keep holding
                # Plain chat: flush what we held, then pass through (guarding
                # against a tool call embedded mid-prose).
                yield from self._passthrough(buffered, stream, messages)
                return
            if len(probe) >= _SNIFF_CHARS and '"tool"' not in probe[:_SNIFF_CHARS]:
                yield from self._passthrough(buffered, stream, messages)
                return
        # Stream ended while still held: last chance for both rescues.
        reply = buffered.strip()
        if not reply:
            return
        if not reply.lstrip().startswith("{"):
            if rescue_armed and _UNSURE_RE.search(reply):
                log.info("uncertainty rescue (full reply) -> web_search(%r)",
                         user_text[:80])
                yield from self._run_and_rephrase(
                    messages, "",
                    {"tool": "web_search",
                     "args": {"query": user_text.strip(" ?.!,")}})
                return
            yield from self._passthrough(reply, iter(()), messages)
            return
        call = self._parse_call(reply)
        if call is None:
            yield buffered  # malformed JSON — speak it rather than go mute
            return
        yield from self._run_and_rephrase(messages, reply, call)

    def _rescue_armed(self, user_text: str) -> bool:
        """Uncertainty rescue only makes sense for a substantive QUESTION when
        the web tool exists — never for smalltalk or statements."""
        if not user_text or self._registry.get("web_search") is None:
            return False
        if len(user_text.split()) < 3:
            return False
        return (user_text.strip().endswith("?")
                or bool(_Q_SHAPE.search(user_text)))

    @staticmethod
    def _close(stream) -> None:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _passthrough(self, prefix: str, stream, messages):
        """Yield prose to the speaker, but watch for a tool call EMBEDDED
        mid-reply ('Let me check. {"tool": ...}') — the model does this and the
        raw JSON used to be spoken aloud. Everything before the marker is
        spoken; the call is executed and its result phrased instead of read."""
        pending = prefix

        def feed(text: str):
            nonlocal pending
            pending += text
            marker = pending.find(_JSON_MARKER)
            if marker >= 0:
                return pending[:marker], True
            # Hold back a tail that could be the start of a split marker.
            safe = len(pending) - (len(_JSON_MARKER) - 1)
            out, keep = (pending[:safe], pending[safe:]) if safe > 0 else ("", pending)
            pending = keep
            return out, False

        out, hit = feed("")
        if not hit:
            if out:
                yield out
            for chunk in stream:
                out, hit = feed(chunk)
                if hit:
                    break
                if out:
                    yield out
        if not hit:
            if pending:
                yield pending
            return
        # Tool call mid-prose: speak the lead-in, run the call, phrase result.
        lead_in = out
        if lead_in.strip():
            yield lead_in
        rest = pending[pending.find(_JSON_MARKER):] + "".join(stream)
        call = self._parse_call(rest)
        if call is None:
            log.debug("embedded tool-ish text was malformed; dropped: %r",
                      rest[:80])
            return
        log.info("embedded tool call rescued: %s", call.get("tool"))
        yield from self._run_and_rephrase(messages, lead_in, call)

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
        if self._registry.get("gaze") is not None:
            # Fast lexical gaze ('look left', 'face forward', 'look at Sam') —
            # move the head with no LLM round trip. High-precision parser; bare
            # 'look, I think…' / 'what's over there?' return None and fall to the
            # LLM, which can still call the gaze tool (with then_answer).
            from zero.head.commands import parse_gaze_command

            if parse_gaze_command(text) is not None:
                return self._registry.get("gaze").safe_run({"text": text}, ctx)
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
        if _MEMORY_SHAPE.search(text):
            return None  # "what did we talk about yesterday" = memory, not web
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
        # The assistant turn is included ONLY when it has content: most tool
        # routes pass raw_reply="" (forced/auto web search never asked the
        # model first), and an EMPTY assistant turn renders as a degenerate
        # `<start_of_turn>model<end_of_turn>` in Gemma's chat template — which
        # is exactly the shape that provoked leaked "thought ..." reasoning
        # prefixes on tool-router turns, despite enable_thinking=false.
        followup = [
            *messages,
            *([{"role": "assistant", "content": raw_reply}]
              if raw_reply.strip() else []),
            {"role": "user", "content":
                f"(Tool result: {result}) State the answer to the user "
                "directly, in one or two spoken sentences. Never narrate the "
                "process — no 'querying', 'searching', 'I looked it up', "
                "'results say', 'it looks like' — and never mention tools, "
                "sources or JSON. If the result doesn't actually answer the "
                "question, say plainly that you couldn't find it."},
        ]
        spoke = False
        for chunk in self._llm.stream(followup):
            spoke = True
            yield chunk
        if not spoke:  # LLM died mid-turn — the raw result still gets spoken
            yield result
