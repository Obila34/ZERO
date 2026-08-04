"""Continual preference learning — corrections that change BEHAVIOUR.

"Speak slower", "keep it shorter", "stop calling me buddy" are not facts to
recite back; they're standing instructions. Detected here, stored in the
procedural memory layer (injected into every future system prompt), and where
an engine knob exists (Piper's length_scale for speaking rate) applied to the
engine immediately too.
"""
from __future__ import annotations

import re

# (regex, normalized preference sentence, rate delta or None)
# Rate delta is applied to Piper's length_scale: bigger = slower speech.
_PREF_PATTERNS: list[tuple[re.Pattern, str, float | None]] = [
    (re.compile(r"\b(?:speak|talk)\s+(?:more\s+slowly|slower)\b", re.I),
     "speak more slowly", +0.15),
    (re.compile(r"\bslow\s+down\b", re.I),
     "speak more slowly", +0.15),
    (re.compile(r"\b(?:speak|talk)\s+(?:faster|quicker|more\s+quickly)\b", re.I),
     "speak faster", -0.15),
    (re.compile(r"\b(?:keep\s+it|be)\s+(?:more\s+)?(?:brief|short(?:er)?|concise)\b", re.I),
     "keep replies short and concise", None),
    (re.compile(r"\b(?:give|use)\s+(?:more\s+detail|longer\s+answers)\b", re.I),
     "give more detailed answers", None),
    (re.compile(r"\b(?:stop|don't|do\s+not)\s+call(?:ing)?\s+me\s+([a-z' \-]{2,25})\b", re.I),
     "never call the user '{0}'", None),
    (re.compile(r"\bdon't\s+ask\s+so\s+many\s+questions\b", re.I),
     "ask fewer follow-up questions", None),
    (re.compile(r"\bstop\s+(?:the\s+)?(?:jokes|joking|being\s+funny)\b", re.I),
     "keep the tone straightforward, fewer jokes", None),
]

# Volume asked for OUT LOUD. Kept separate from the rate table because it turns
# a different knob: rate changes the engine, level changes the speaker. People
# genuinely ask a voice to change loudness ("keep it down", "I can't hear
# you"), and it should STICK rather than snap back on the next sentence.
_VOLUME_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r"\bwhisper\b", re.I), "speak very quietly", 0.5),
    (re.compile(r"\b(?:talk|speak|say it)\s+(?:more\s+)?"
                r"(?:quiet(?:ly|er)?|soft(?:ly|er)?|lower)\b", re.I),
     "speak more quietly", 0.72),
    (re.compile(r"\b(?:lower|turn down)\s+(?:your\s+)?(?:voice|volume)\b", re.I),
     "speak more quietly", 0.72),
    (re.compile(r"\b(?:keep it down|not so loud|too loud)\b", re.I),
     "speak more quietly", 0.72),
    (re.compile(r"\b(?:speak|talk)\s+(?:up|louder|more loudly)\b", re.I),
     "speak louder", 1.35),
    (re.compile(r"\b(?:turn up|raise)\s+(?:your\s+)?(?:voice|volume)\b", re.I),
     "speak louder", 1.35),
    (re.compile(r"\bcan'?t hear you\b", re.I), "speak louder", 1.35),
    (re.compile(r"\b(?:normal|regular)\s+volume\b", re.I),
     "speak at normal volume", 1.0),
]


def parse_volume(text: str) -> tuple[str, float] | None:
    """(normalized_preference, level_multiplier) when the utterance asks ZERO
    to change how LOUD it is, else None. Short utterances only — a story that
    mentions whispering is not an instruction."""
    t = (text or "").strip()
    if len(t.split()) > 10:
        return None
    for rx, pref, level in _VOLUME_PATTERNS:
        if rx.search(t):
            return pref, level
    return None


def parse_preference(text: str) -> tuple[str, float | None] | None:
    """(normalized_preference, rate_delta) if this utterance is a behavioural
    correction, else None."""
    t = text.strip()
    if len(t.split()) > 10:      # corrections are short; don't mine stories
        return None
    for rx, template, delta in _PREF_PATTERNS:
        m = rx.search(t)
        if m:
            pref = (template.format(*(g.strip() for g in m.groups()))
                    if m.groups() else template)
            return pref, delta
    return None


def apply_rate_delta(voice, delta: float) -> bool:
    """Nudge the active TTS engine's speaking rate if it has one (Piper's
    length_scale). Returns True when a knob was actually turned."""
    for attr_holder in (voice, getattr(voice, "tts", None),
                        getattr(voice, "_tts", None)):
        if attr_holder is None:
            continue
        for obj in (attr_holder, getattr(attr_holder, "_fallback", None),
                    getattr(attr_holder, "_local", None)):
            if obj is not None and hasattr(obj, "length_scale"):
                obj.length_scale = float(
                    min(1.6, max(0.6, obj.length_scale + delta)))
                return True
    return False
