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
