"""Parse spoken arm/hand commands into a gesture intent — the FAST lexical path.

Mirrors zero/head/commands.py: high precision so ordinary speech never moves
an arm ('wave goodbye to Sam for me' is a request to speak, but a bare 'wave'
is a request to move). Anything ambiguous returns None and falls through to
the LLM, which can still call the arm tool.

Shapes returned:
    {"kind": "gesture", "name": "<gesture>"}
    {"kind": "joint", "joints": [...], "degrees": +/-N, "part": ..., "side": ...}
    None
"""
from __future__ import annotations

import re

_SIDE = r"(?P<side>right|left)"

# Default magnitudes for a joint move with no explicit degrees. 15 was measured
# as too small to read as movement across a room (2026-08-17) — the joint moved
# exactly as commanded, it just did not LOOK like anything.
STEP_DEG = 30.0     # a plain "raise your elbow"
SMALL_DEG = 10.0    # "a bit", "slightly"
FULL_DEG = 999.0    # "all the way" — the envelope clamps it to the real stop


def set_default_step(deg: float) -> None:
    """Override the default move size from config (arms.step_deg), so how big
    a plain "raise your arm" is can be tuned without editing code."""
    global STEP_DEG, SMALL_DEG
    STEP_DEG = float(deg)
    SMALL_DEG = max(2.0, float(deg) / 3.0)

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
    rf"\b(?:(?:put|lower|drop|rest)(?:ing)?\s+(?:down\s+)?(?:your\s+|the\s+)?"
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

# ── direct joint control ────────────────────────────────────────────────────
# "raise your right elbow", "bend your left elbow 20 degrees", "lower your arm
# a bit". Spoken body-part words map onto gateway joints; the in/out pair and
# the wrists have no mapping at all, so they can't be driven even by name.
_PART_JOINT = {
    "elbow": "elbow_joint",
    "forearm": "elbow_joint",
    "shoulder": "up_down_joint",
    "arm": "up_down_joint",
    "bicep": "bicep_joint",
    "biceps": "bicep_joint",
    "upper arm": "bicep_joint",
    "hand": "up_down_joint",     # "raise your hand" = lift the whole arm
}
# "arm"/"hand" name a limb, not a joint. Sent DOWN with no stated amount they
# mean "go back to rest", not "nudge 15 degrees" — so those fall through to the
# rest gesture. A specific joint, or any explicit amount, is a real joint move.
_GENERIC_PARTS = frozenset({"arm", "hand"})
# Verb -> direction. "bend"/"curl" close the joint, "straighten"/"extend" open
# it; which way the metal actually turns is per-joint calibration, corrected
# with arms.joint_sign rather than by editing this table.
_UP_VERB = re.compile(r"\b(?:raise|rais(?:e|ing)|lift(?:ing)?|up|straighten|"
                      r"extend|open)\b", re.IGNORECASE)
_DOWN_VERB = re.compile(r"\b(?:lower|drop(?:ping)?|down|bend|curl|close|"
                        r"fold|tuck)\b", re.IGNORECASE)
_JOINT_RE = re.compile(
    r"\b(?:move|turn|rotate|raise|rais(?:e|ing)|lift(?:ing)?|lower|drop|bend|"
    r"curl|straighten|extend|fold|tuck|put)\w*\b[^.?!]*?"
    rf"\b(?:your|the)?\s*(?P<side>right|left|both)?\s*"
    rf"(?P<part>{'|'.join(sorted(_PART_JOINT, key=len, reverse=True))})s?\b",
    re.IGNORECASE)
_BARE_RE = re.compile(
    rf"^\s*(?:your\s+|the\s+)?(?P<side>right|left|both)?\s*"
    rf"(?P<part>{'|'.join(sorted(_PART_JOINT, key=len, reverse=True))})s?\s+"
    rf"(?P<dir>up|down)\b", re.IGNORECASE)
_DEG_RE = re.compile(r"(?P<deg>\d{1,3}(?:\.\d+)?)\s*(?:deg|degrees?|°)",
                     re.IGNORECASE)
_SMALL = re.compile(r"\b(?:a\s+(?:bit|little|touch)|slightly|small|tiny)\b",
                    re.IGNORECASE)
_FULL = re.compile(r"\b(?:all\s+the\s+way|fully|completely|max(?:imum)?|"
                   r"as\s+far\s+as\s+you\s+can)\b", re.IGNORECASE)


def _side(m, default="right") -> str:
    s = (m.groupdict().get("side") or default)
    return s.lower()


def parse_arm_command(text: str) -> dict | None:
    """Return an arm gesture intent or None. High precision by design."""
    if not text:
        return None
    t = text.strip()

    # Bare "<part> up/down" — no verb, the way people actually talk ("arms up",
    # "left elbow down"). Checked first: it is unambiguous.
    b = _BARE_RE.search(t)
    if b:
        part = b.group("part").lower()
        if part not in _PART_JOINT:
            part = part.rstrip("s")
        side = (b.group("side") or "right").lower()
        # "arms up"/"arms down" with no side named means BOTH — a plural part
        # or the word "both".
        if b.group("side") is None and b.group(0).lower().find(part + "s") >= 0:
            side = "both"
        if part in _GENERIC_PARTS and b.group("dir").lower() == "down":
            return {"kind": "gesture", "name": "rest"}      # "arms down"
        sides = ("right", "left") if side == "both" else (side,)
        deg = float(_DEG_RE.search(t).group("deg")) if _DEG_RE.search(t) else STEP_DEG
        sign = 1.0 if b.group("dir").lower() == "up" else -1.0
        suffix = _PART_JOINT[part]
        return {"kind": "joint",
                "joints": [f"{s}_{suffix}" for s in sides],
                "degrees": sign * deg, "part": part, "side": side}

    # Direct joint control first: "raise your right elbow" names a specific
    # joint and must not be swallowed by the looser gesture patterns below.
    m = _JOINT_RE.search(t)
    if m:
        part = m.group("part").lower()
        if part not in _PART_JOINT:           # spoken plural ("shoulders")
            part = part.rstrip("s")
        up, down = bool(_UP_VERB.search(t)), bool(_DOWN_VERB.search(t))
        dm = _DEG_RE.search(t)
        full, small = bool(_FULL.search(t)), bool(_SMALL.search(t))
        explicit = bool(dm) or full or small
        # A limb sent DOWN with no stated amount means "back to rest", so it
        # falls through to the rest gesture instead of nudging a joint.
        if not (part in _GENERIC_PARTS and down and not up and not explicit):
            if dm:
                amount = float(dm.group("deg"))
            elif full:
                amount = FULL_DEG
            elif small:
                amount = SMALL_DEG
            else:
                amount = STEP_DEG             # "rotate your bicep": no
            sign = -1.0 if (down and not up) else 1.0   # direction given, so
            side = (m.group("side") or "right").lower()  # move it positively
            sides = ("right", "left") if side == "both" else (side,)
            suffix = _PART_JOINT[part]
            return {"kind": "joint",
                    "joints": [f"{s}_{suffix}" for s in sides],
                    "degrees": sign * amount,
                    "part": part, "side": side}

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
