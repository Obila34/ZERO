"""Parse spoken arm/hand commands into a gesture intent — the FAST lexical path.

Mirrors zero/head/commands.py: high precision so ordinary speech never moves
an arm ('wave goodbye to Sam for me' is a request to speak, but a bare 'wave'
is a request to move). Anything ambiguous returns None and falls through to
the LLM, which can still call the arm tool.

Shapes returned:  {"kind": "gesture", "name": "<gesture>"}  or None.
"""
from __future__ import annotations

import re

_SIDE = r"(?P<side>right|left)"

# "wave" as an imperative — short utterances or explicit "wave your hand".
_WAVE = re.compile(
    rf"\b(?:wave|waving)\b(?:\s+(?:your|the)\s+(?:{_SIDE}\s+)?(?:hand|arm))?",
    re.IGNORECASE)
# "raise/lift your right arm/hand", "put your arm up"
_RAISE = re.compile(
    rf"\b(?:raise|rais(?:e|ing)|lift(?:ing)?|put(?:ting)?)\s+(?:up\s+)?"
    rf"(?:your|the)\s+(?:{_SIDE}\s+)?(?:arm|hand)s?(?:\s+up)?\b",
    re.IGNORECASE)
_RAISE_UP = re.compile(r"\bup\b", re.IGNORECASE)
# "put your arms down", "lower your arm", "arms down", "rest your arms"
_DOWN = re.compile(
    rf"\b(?:(?:put|lower|drop|rest)(?:ing)?\s+(?:down\s+)?(?:your|the)\s+"
    rf"(?:{_SIDE}\s+)?(?:arm|hand)s?(?:\s+down)?|arms?\s+down)\b",
    re.IGNORECASE)
_DOWN_WORD = re.compile(r"\b(?:down|lower|drop|rest)\b", re.IGNORECASE)
# "open/close your (right/left) hand"
_HAND = re.compile(
    rf"\b(?P<verb>open|close|clos(?:e|ing)|opening)\s+(?:your|the)\s+"
    rf"(?:{_SIDE}\s+)?(?:hand|fist|fingers)\b", re.IGNORECASE)
# "shake my hand" / "give me a handshake"
_SHAKE = re.compile(
    r"\b(?:shake\s+(?:my|their|his|her)\s+hand|hand\s*shake)\b", re.IGNORECASE)

_WAVE_MAX_WORDS = 5     # a bare "wave"/"wave your hand" is a command;
                        # "wave goodbye to everyone at the door" is not


def _side(m, default="right") -> str:
    s = (m.groupdict().get("side") or default)
    return s.lower()


def parse_arm_command(text: str) -> dict | None:
    """Return an arm gesture intent or None. High precision by design."""
    if not text:
        return None
    t = text.strip()

    m = _HAND.search(t)
    if m:
        verb = "open" if m.group("verb").lower().startswith("open") else "close"
        return {"kind": "gesture", "name": f"{verb}_{_side(m)}_hand"}

    m = _DOWN.search(t)
    if m:
        return {"kind": "gesture", "name": "rest"}

    m = _RAISE.search(t)
    if m:
        # "put your arm down" also matches _RAISE's verb set — the down words
        # decide (checked above via _DOWN, but "putting your hand down please"
        # style phrasings can land here).
        if _DOWN_WORD.search(t) and not _RAISE_UP.search(t):
            return {"kind": "gesture", "name": "rest"}
        return {"kind": "gesture", "name": f"raise_{_side(m)}"}

    if _SHAKE.search(t):
        return {"kind": "gesture", "name": "handshake"}

    m = _WAVE.search(t)
    if m:
        explicit = "hand" in m.group(0).lower() or "arm" in m.group(0).lower()
        if explicit or len(t.split()) <= _WAVE_MAX_WORDS:
            return {"kind": "gesture", "name": f"wave_{_side(m)}"}

    return None
