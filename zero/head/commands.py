"""Parse spoken gaze commands into a structured intent — the FAST lexical path.

'look left', 'turn to your right', 'look up', 'look back at me', 'look at Sam',
'face forward' must move the head with NO ~3 s LLM round trip, exactly like the
timer/time tier-1 intents. This is the pure, high-precision parser; the router
dispatches its result to the gaze tool instantly, and anything ambiguous ('look
at that', 'what's over there?') falls through to the LLM, which can still call
the same tool with look_then_speak.

Frame of reference: relative directions are ROBOT-EGOCENTRIC — 'your right' and a
bare 'right' both mean the robot's right — the single frame the plan commits to
(the neck's limits and the digital crop are all egocentric). Pure logic, no I/O.
"""
from __future__ import annotations

import re

# Default magnitude for a bare direction (no explicit degrees). ~25° is a clear
# glance that still engages the neck (past the ~15-20° eyes-only threshold).
DEFAULT_DEG = 25.0
# Sentinel for "go all the way" — the controller clamps it to the axis limit, so
# 'turn completely left' lands at the mechanical max without the parser needing
# to know the configured range.
FULL_DEG = 999.0
# "far left/right" = a big deliberate turn, between a glance (25) and the max.
FAR_DEG = 60.0

# "as far as you can" / "all the way" / "completely" / "fully" / "hard" / "max"
_FULL = re.compile(
    r"\b(?:all\s+the\s+way|completely|complete|fully|full|entirely|totally|"
    r"as\s+far\s+(?:(?:to\s+the\s+)?(?:left|right|up|down)\s+)?as\s+(?:you\s+can|possible)|"
    r"maximum|max)\b", re.IGNORECASE)

# intensifiers meaning "a large turn" (not necessarily the mechanical max).
_FAR = re.compile(r"\b(?:far|hard|a\s+lot|much|right\s+over)\b", re.IGNORECASE)

# manual servo control: freeze where it is / hand control back to tracking.
# A bare stop-word is NOT enough on a long utterance — "hold on, what's the
# time?" or "stop joking" must never hijack the turn into a head-hold. The hold
# fires only when the utterance is a short imperative (a few words) or carries
# explicit head/motion context ("stop moving your head", "keep your head still").
_HOLD = re.compile(r"\b(?:stop|freeze|halt|hold(?:\s+(?:it|still|there|on|steady|position|your\s+head))?|don'?t\s+move|stay\s+(?:there|still|put))\b", re.IGNORECASE)
_HOLD_CTX = re.compile(r"\b(?:head|neck|mov(?:e|ing)|turn(?:ing)?|look(?:ing)?|"
                       r"track(?:ing)?|follow(?:ing)?|still|there|put)\b",
                       re.IGNORECASE)
# "stop the timer / music / talking" is about something else entirely.
_HOLD_VETO = re.compile(r"\b(?:timer|alarm|reminder|music|song|video|recording|"
                        r"talk(?:ing)?|speak(?:ing)?|jok(?:e|ing)|count(?:ing)?|"
                        r"sing(?:ing)?|play(?:ing)?)\b", re.IGNORECASE)
_HOLD_MAX_WORDS = 4
_TRACK = re.compile(r"\b(?:follow\s+(?:me|my\s+(?:head|hand|face))|track\s+(?:me|my\s+(?:head|hand))|keep\s+(?:following|tracking)|start\s+(?:following|tracking)|resume(?:\s+tracking)?|as\s+you\s+were|go\s+back\s+to\s+(?:following|tracking))\b", re.IGNORECASE)

_DIR_AXIS = {"left": ("pan", -1.0), "right": ("pan", +1.0),
             "up": ("tilt", +1.0), "down": ("tilt", -1.0)}

# A gaze verb near the start: look / turn / glance / face / gaze (+ optional
# 'your'/'to the'/'over'/'back'). "watch"/"see" excluded (too ambiguous).
_VERB = (r"(?:look(?:ing)?|turn(?:ing)?|glanc(?:e|ing)|gaz(?:e|ing)|fac(?:e|ing)|point(?:ing)?|mov(?:e|ing)|swivel(?:l?ing)?|rotat(?:e|ing)|spin(?:ning)?|swing(?:ing)?|aim(?:ing)?|pan(?:ning)?|tilt(?:ing)?|go(?:ing)?|shift(?:ing)?)")
# Filler words allowed between the gaze verb and the direction, so natural
# phrasings parse: 'turn head to its complete left', 'look all the way to the
# right', 'turn a little to your left'. Kept to connective/quantifier words so
# 'look, I went left' still won't fire (no verb→direction adjacency there).
# NOTE: the pronoun "it" is deliberately NOT a filler — "turn it up / turn it
# down" is volume/appliance talk, not gaze, and used to hijack the turn (audit
# M1). The possessive "its" ("turn head to its complete left") stays.
_FILL = (r"back|over|round|around|again|to|your|my|the|towards?|at|its|head|"
         r"all|way|as|far|you|can|possible|completely|complete|fully|full|"
         r"entirely|totally|maximum|max|hard|side|a|little|bit|slightly|please|"
         r"more|much|now|and")
_LEAD = rf"\b{_VERB}\b(?:\s+(?:{_FILL}))*"

_DEG = re.compile(r"(?P<deg>\d{1,3}(?:\.\d+)?)\s*(?:deg|degrees?|°)", re.IGNORECASE)
# direction, optionally with a degrees phrase sitting between verb and direction
# ('look 15° up') — the magnitude itself is pulled out separately by _DEG.
_DIR = re.compile(
    rf"{_LEAD}\s+(?:\d{{1,3}}(?:\.\d+)?\s*(?:deg|degrees?|°)\s+)?"
    rf"(?P<dir>left|right|up|down)\b", re.IGNORECASE)
_CENTER = re.compile(
    rf"{_LEAD}\s+(?:center|centre|forward|forwards|ahead|straight|front)\b"
    rf"|\bface\s+(?:me\s+)?(?:forward|forwards|ahead|straight|front)\b",
    re.IGNORECASE)
_AT_ME = re.compile(rf"{_LEAD}\s+(?:at\s+|to\s+)?me\b", re.IGNORECASE)
# 'look at <Name>' — a capitalised or known token. We capture a single word and
# let the caller resolve it against known people; unresolved → best-effort.
_AT_NAME = re.compile(
    rf"{_VERB}\b(?:\s+(?:back|over|to|your|the|towards?))*\s+at\s+(?P<name>[a-z][a-z'-]{{1,20}})\b",
    re.IGNORECASE)

_NOT_A_NAME = {"me", "you", "it", "that", "this", "there", "them", "us", "the",
               "my", "him", "her", "his", "your", "yourself", "everyone",
               "someone", "anybody", "camera", "left", "right", "up", "down",
               "center", "centre", "forward", "ahead", "straight", "front"}


# An utterance that names an ARM part is not a gaze command, however much it
# looks like one: "rotate right bicep" matched the gaze verb + "right" and
# turned the HEAD instead of the bicep (2026-08-17). The arm parser owns these.
_ARM_PART = re.compile(
    r"\b(?:bicep|biceps|elbow|elbows|forearm|shoulder|shoulders|wrist|"
    r"wrists|finger|fingers|arm|arms)\b", re.IGNORECASE)


def parse_gaze_command(text: str) -> dict | None:
    """Return a gaze intent dict or None. Shapes:
        {"kind": "direction", "axis": "pan"|"tilt", "sign": +/-1, "degrees": N}
        {"kind": "center"}
        {"kind": "person", "name": "me"|<name>}
    High precision: only fires on clear gaze phrasings so normal speech ('look,
    I think…') never moves the head."""
    if not text:
        return None
    t = text.strip()
    if _ARM_PART.search(t):
        return None            # names an arm part -> the arm parser's business

    # center / face-forward first (most specific)
    if _CENTER.search(t):
        return {"kind": "center"}

    # manual servo control: hold/stop where it is, or resume tracking. Guarded:
    # short imperative ("stop", "hold on", "stay put") or explicit head/motion
    # context — a stop-word buried in a longer sentence falls through.
    if _HOLD.search(t) and not _HOLD_VETO.search(t):
        if len(t.split()) <= _HOLD_MAX_WORDS or _HOLD_CTX.search(t):
            return {"kind": "hold"}
    if _TRACK.search(t):
        return {"kind": "track"}

    m = _DIR.search(t)
    if m:
        axis, sign = _DIR_AXIS[m.group("dir").lower()]
        dm = _DEG.search(t)
        if dm:
            deg = float(dm.group("deg"))        # explicit "30 degrees left"
        elif _FULL.search(t):
            deg = FULL_DEG                       # "all the way / completely left"
        elif _FAR.search(t):
            deg = FAR_DEG                        # "far left" = a big turn
        else:
            deg = DEFAULT_DEG                     # bare "look left" = a glance
        return {"kind": "direction", "axis": axis, "sign": sign, "degrees": deg}

    if _AT_ME.search(t):
        return {"kind": "person", "name": "me"}

    m = _AT_NAME.search(t)
    if m:
        name = m.group("name").lower()
        if name not in _NOT_A_NAME:
            return {"kind": "person", "name": name}

    return None
