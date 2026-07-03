"""The starter tools: time/date, timers, reminders, remember/recall.

Reminders are double-written: a live timer (fires an announcement) AND a
memory fact, so a reboot doesn't silently eat a promise.
"""
from __future__ import annotations

import datetime as _dt
import re

from zero.tools.base import Tool, ToolContext
from zero.tools.timers import TimerManager, humanize_s

_DUR_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b)?\s*"
    r"(?:(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b)?\s*"
    r"(?:(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b)?",
    re.IGNORECASE,
)
_WORD_NUM = {
    "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "ninety": 90, "half": 0.5, "quarter": 0.25,
}


def parse_duration_s(text: str) -> float | None:
    """'5 minutes', '1 hour 30 minutes', '90s', 'ten minutes', 'half an hour'
    -> seconds, or None if no duration is present."""
    t = text.lower()
    # Idioms first, before number-word substitution rewrites their articles.
    t = (t.replace("half an hour", "30 minutes")
          .replace("half hour", "30 minutes")
          .replace("quarter of an hour", "15 minutes"))
    t = re.sub(r"\b(" + "|".join(_WORD_NUM) + r")\b",
               lambda m: str(_WORD_NUM[m.group(1)]), t)
    m = _DUR_RE.search(t)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (float(g) if g else 0.0 for g in m.groups())
    total = h * 3600 + mi * 60 + s
    return total if total > 0 else None


class TimeTool(Tool):
    name = "time"
    description = "Tell the current time and date."
    parameters = {}

    def run(self, args: dict, ctx: ToolContext) -> str:
        now = _dt.datetime.now()
        return (f"It's {now.strftime('%I:%M %p').lstrip('0')} on "
                f"{now.strftime('%A, %B %d')}.")


class TimerTool(Tool):
    name = "timer"
    description = "Set a countdown timer."
    parameters = {"duration": "e.g. '5 minutes' or seconds as a number",
                  "label": "optional, what the timer is for"}

    def __init__(self, timers: TimerManager):
        self._timers = timers

    def run(self, args: dict, ctx: ToolContext) -> str:
        dur = args.get("duration", "")
        seconds = (float(dur) if isinstance(dur, (int, float))
                   else parse_duration_s(str(dur)))
        if not seconds:
            return "I didn't catch how long the timer should be."
        label = str(args.get("label", "") or "").strip()
        self._timers.set(seconds, label, kind="timer", person_id=ctx.person_id)
        return (f"Timer set for {humanize_s(seconds)}"
                + (f", for {label}" if label else "") + ".")


class ReminderTool(Tool):
    name = "reminder"
    description = "Remind the user about something after a delay."
    parameters = {"duration": "when, e.g. 'in 20 minutes'",
                  "about": "what to remind them of"}

    def __init__(self, timers: TimerManager):
        self._timers = timers

    def run(self, args: dict, ctx: ToolContext) -> str:
        seconds = parse_duration_s(str(args.get("duration", "")))
        about = str(args.get("about", "") or "").strip()
        if not seconds:
            return "I didn't catch when to remind you."
        if not about:
            return "I didn't catch what to remind you about."
        self._timers.set(seconds, about, kind="reminder", person_id=ctx.person_id)
        # Reboot-safe trace: the promise also lands in long-term memory.
        if ctx.memory is not None:
            who = ctx.person_name or "user"
            ctx.memory.remember(
                f"reminder for {who}",
                f"{about} (asked at {_dt.datetime.now():%H:%M}, "
                f"due in {humanize_s(seconds)})")
        return f"Okay — I'll remind you about {about} in {humanize_s(seconds)}."


class RememberTool(Tool):
    name = "remember"
    description = "Store a lasting fact the user tells you."
    parameters = {"fact": "the fact to remember"}

    def run(self, args: dict, ctx: ToolContext) -> str:
        fact = str(args.get("fact", "") or "").strip()
        if not fact:
            return "I didn't catch what to remember."
        if ctx.memory is None:
            return "My memory is switched off right now, so I can't keep that."
        key = f"note ({ctx.person_name or 'user'})"
        ctx.memory.remember(key, fact)
        return f"Got it — I'll remember that {fact}."


class RecallTool(Tool):
    name = "recall"
    description = "Look up what you remember about a topic."
    parameters = {"query": "the topic to look up"}

    def run(self, args: dict, ctx: ToolContext) -> str:
        query = str(args.get("query", "") or "").strip()
        if ctx.memory is None:
            return "My memory is switched off right now."
        search = getattr(ctx.memory, "search", None)
        if callable(search):                      # Phase-3 semantic recall
            hits = search(query, person_id=ctx.person_id)
        else:                                     # substring fallback
            q = query.lower()
            hits = [f"{k}: {v}" for k, v in ctx.memory.facts().items()
                    if q in k.lower() or q in v.lower()]
        if not hits:
            return f"I don't remember anything about {query or 'that'}."
        return "Here's what I remember: " + "; ".join(str(h) for h in hits[:4]) + "."
