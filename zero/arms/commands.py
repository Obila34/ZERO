"""Parse spoken arm/hand commands into a gesture intent — the FAST lexical path.

Mirrors zero/head/commands.py: high precision so ordinary speech never moves
an arm ('wave goodbye to Sam for me' is a request to speak, but a bare 'wave'
is a request to move). Anything ambiguous returns None and falls through to
the LLM, which can still call the arm tool.

Shapes returned:
    {"kind": "gesture", "name": "<gesture>"}
    {"kind": "joint", "joints": [...], "degrees": +/-N, "part": ..., "side": ...}
    {"kind": "hand_gesture", "name": "<pose>", "side": ...}     peace, fist, ...
    {"kind": "hand", "state": "open"|"close", "side": ...}
    {"kind": "finger", "finger": ..., "side": ..., "closure": 0..1|None,
     "degrees": N|None}
    {"kind": "spell", "word": "<WORD>"}          fingerspell a word in KSL
    {"kind": "spell_name"}                       "spell my name" — the tool
                                                 resolves it from who's talking
    {"kind": "letter", "letter": "<A-Z>"}        sign one letter
    {"kind": "sign", "gloss": "<word>"}          a lexicon sign ("sign hello");
                                                 the tool falls back to
                                                 spelling, out loud, when the
                                                 lexicon doesn't have it
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
    rf"(?P<part>{'|'.join(sorted(_PART_JOINT, key=len, reverse=True))})"
    rf"(?P<plural>s)?\b(?!\s*side\b)",
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
                   r"as\s+(?:far|high|low)\s+(?:up\s+|down\s+)?as\s+you\s+can|"
                   r"right\s+up|as\s+high\s+as\s+possible)\b", re.IGNORECASE)


# ── hand poses + sign language (KSL) ────────────────────────────────────────
# From the Pi's sign build (2026-08-24 merge), tightened: each pattern names
# the pose explicitly, so ordinary speech ("that's a fair point") never moves
# a hand. Side defaults to BOTH for hand poses — the robot signs bilaterally.
_ILY = re.compile(
    r"\b(?:i\s+love\s+you\s+sign|sign\s+i\s+love\s+you|ily\s+sign|"
    r"show\s+(?:me\s+|us\s+)?i\s+love\s+you)\b", re.IGNORECASE)
_PEACE = re.compile(
    r"\b(?:peace|victory)\s+sign\b|\bsign\s+peace\b|"
    r"\bshow\s+(?:me\s+|us\s+)?(?:the\s+)?peace\s+sign\b", re.IGNORECASE)
_THUMBS_UP = re.compile(
    r"\bthumbs?\s+up\b", re.IGNORECASE)
_FIST_G = re.compile(
    r"\b(?:make|show)\s+(?:me\s+|us\s+)?a\s+fist\b|\bfist\s+bump\b"
    # "clench/squeeze/ball up (your/a/my) fist(s)/hand(s)" — second-person
    # possessives only, so narrated speech ("he clenched his fists") never
    # moves the robot.
    r"|\b(?:clench(?:ing)?|squeez(?:e|ing)|ball(?:ing)?(?:\s+up)?)\s+"
    r"(?:(?:your|the|a)\s+)?(?:fists?|hands?)\b",
    re.IGNORECASE)
# "unclench/release your fist" -> open hand
_UNCLENCH = re.compile(
    r"\b(?:unclench(?:ing)?|release|relax)\s+(?:(?:your|the|my)\s+)?"
    r"(?:fists?|hands?|fingers)\b", re.IGNORECASE)
_OK_SIGN = re.compile(
    r"\b(?:ok(?:ay)?\s+sign|give\s+(?:me\s+|us\s+)?(?:an?\s+)?ok(?:ay)?\b)",
    re.IGNORECASE)
_ROCK_ON = re.compile(r"\b(?:rock\s+on|rock\s+sign|the\s+horns)\b",
                      re.IGNORECASE)
# "a pinch of salt" / "pinching pennies" must not move a hand: the pose
# must be NAMED, not the bare verb (audit 2026-08-25 sign #1).
_PINCH = re.compile(
    r"\bpinch\s+(?:your\s+)?(?:fingers|thumb)\b"
    r"|\b(?:do|make|show)\s+(?:me\s+|us\s+)?a\s+pinch\b", re.IGNORECASE)
_WIGGLE = re.compile(
    rf"\b(?:wiggle|wave)\s+(?:your\s+|the\s+)?(?:{_SIDE}\s+)?fingers?\b",
    re.IGNORECASE)
_POINT_FINGER = re.compile(
    rf"\bpoint\s+(?:your\s+|the\s+)?(?:{_SIDE}\s+)?(?:finger|index)\b",
    re.IGNORECASE)
# "curl your index", "bend your left thumb 45 degrees", "straighten your pinky"
_FINGER_RE = re.compile(
    rf"\b(?P<verb>curl|bend|close|flex|extend|straighten|open|raise|lift|"
    rf"move)\s+(?:your\s+|the\s+)?(?:{_SIDE}\s+)?"
    r"(?P<finger>thumb|index|middle|ring|pinky|pinkie)(?:\s+finger)?\b",
    re.IGNORECASE)
_FINGER_CLOSE_VERB = re.compile(r"\b(?:curl|bend|close|flex)\b", re.IGNORECASE)
# Fingerspelling. "spell"/"fingerspell"/"sign the word/name X"; the stoplist
# keeps "can you spell that" style fragments from spelling the word "that".
_SPELL_NAME = re.compile(
    r"\b(?:spell|fingerspell|sign)\s+(?:out\s+)?my\s+name\b", re.IGNORECASE)
# The noun "spell" ("a dry spell lately", "put a spell on me") must never
# fingerspell — the verb is only imperative when not preceded by an
# article/adjective/possessive (audit sign #1).
_ASL_SPELL = re.compile(
    r"\b(?:finger\s*spell|fingerspell|"
    r"(?<!\ba\s)(?<!dry\s)(?<!the\s)(?<!his\s)(?<!her\s)(?<!my\s)"
    r"(?<!its\s)spell(?:\s+out)?|"
    r"how\s+do\s+you\s+spell|can\s+you\s+spell|sign\s+the\s+(?:word|name))\s+"
    r"(?:the\s+(?:word|name)\s+)?(?P<word>[A-Za-z]+)\b", re.IGNORECASE)
_SPELL_STOP = frozenset({
    "THE", "THAT", "THIS", "IT", "ME", "MY", "YOUR", "OUT", "WORD", "NAME",
    "SOMETHING", "ANYTHING", "A", "AN",
    "ON", "UP", "OF", "FOR", "IN", "OFF", "LATELY", "IS", "WAS", "TO"})
# Known STT mishears of words people actually ask for — extend as they crop up.
_SPELL_CORRECTIONS = {"PIT": "PETER", "PITA": "PETER"}
_ASL_LETTER = re.compile(
    r"\b(?:show(?:\s+me|\s+us)?|sign|do|make)\s+(?:the\s+)?letter\s+"
    r"(?P<letter>[A-Za-z])\b"
    r"|\bwhat(?:'s|\s+is)\s+(?:the\s+letter\s+)?(?P<l2>[A-Za-z])\s+in\s+"
    r"(?:sign(?:\s+language)?|ksl|asl)\b", re.IGNORECASE)
# "sign hello" — a lexicon gloss. Checked AFTER the specific sign phrases
# above so "sign peace"/"sign the word cow" resolve to their own kinds.
_SIGN_WORD = re.compile(
    r"\b(?:(?<!\ba\s)(?<!the\s)(?<!no\s)(?<!any\s)sign|"
    r"show\s+(?:me\s+|us\s+)?the\s+sign\s+for)\s+"
    r"(?P<gloss>[A-Za-z]+)\b", re.IGNORECASE)
_SIGN_STOP = frozenset({
    "the", "a", "an", "language", "letter", "word", "name", "here", "it",
    "that", "this", "something", "me", "please",
    # "sign up for the class", "sign in", "sign off", "a sign of the times"
    "up", "in", "out", "on", "off", "of", "for", "with", "them", "him",
    "her", "us", "there"})


def _both_side(t: str) -> str:
    low = t.lower()
    if "left" in low and "right" not in low:
        return "left"
    if "right" in low and "left" not in low:
        return "right"
    return "both"


def _side(m, default="right") -> str:
    s = (m.groupdict().get("side") or default)
    return s.lower()


def parse_arm_command(text: str) -> dict | None:
    """Return an arm gesture intent or None. High precision by design."""
    if not text:
        return None
    t = text.strip()

    # ── sign language first: these phrasings are the most specific ─────────
    if _SPELL_NAME.search(t):
        return {"kind": "spell_name"}
    m = _ASL_SPELL.search(t)
    if m:
        w = _SPELL_CORRECTIONS.get(m.group("word").upper(),
                                   m.group("word").upper())
        if w not in _SPELL_STOP:
            return {"kind": "spell", "word": w}
    m = _ASL_LETTER.search(t)
    if m:
        let = m.group("letter") or m.group("l2")
        if let:
            return {"kind": "letter", "letter": let.upper()}

    # ── named hand poses (before the generic sign-gloss catch-all) ──────────
    if _ILY.search(t):
        return {"kind": "hand_gesture", "name": "i_love_you",
                "side": _both_side(t)}
    if _PEACE.search(t):
        return {"kind": "hand_gesture", "name": "peace", "side": _both_side(t)}
    if _THUMBS_UP.search(t) and len(t.split()) <= 6:
        return {"kind": "hand_gesture", "name": "thumbs_up",
                "side": _both_side(t)}
    if _UNCLENCH.search(t):
        return {"kind": "hand", "state": "open", "side": _both_side(t)}
    if _FIST_G.search(t):
        return {"kind": "hand_gesture", "name": "fist", "side": _both_side(t)}
    if _OK_SIGN.search(t):
        return {"kind": "hand_gesture", "name": "ok_sign",
                "side": _both_side(t)}
    if _ROCK_ON.search(t):
        return {"kind": "hand_gesture", "name": "rock_on",
                "side": _both_side(t)}
    if _PINCH.search(t):
        return {"kind": "hand_gesture", "name": "pinch", "side": _both_side(t)}
    if _WIGGLE.search(t):
        return {"kind": "hand_gesture", "name": "wiggle",
                "side": _both_side(t)}
    if _POINT_FINGER.search(t):
        return {"kind": "hand_gesture", "name": "point_hand",
                "side": _both_side(t)}

    # "sign hello" — a lexicon gloss for the sign engine. Last of the sign
    # family so the specific phrasings above win.
    m = _SIGN_WORD.search(t)
    if m:
        gloss = m.group("gloss").lower()
        if gloss not in _SIGN_STOP:
            return {"kind": "sign", "gloss": gloss}

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
            side = (m.group("side") or                   # move it positively
                    ("both" if m.group("plural") else "right")).lower()
            sides = ("right", "left") if side == "both" else (side,)
            suffix = _PART_JOINT[part]
            return {"kind": "joint",
                    "joints": [f"{s}_{suffix}" for s in sides],
                    "degrees": sign * amount,
                    "part": part, "side": side}

    # Individual finger control ("curl your index", "bend the left thumb 40
    # degrees"). After the joint block: elbow/shoulder verbs must win there.
    m = _FINGER_RE.search(t)
    if m:
        finger = m.group("finger").lower().replace("pinkie", "pinky")
        dm = _DEG_RE.search(t)
        out = {"kind": "finger", "finger": finger,
               "side": (m.group("side") or "both").lower(),
               "closure": None, "degrees": None}
        if dm:
            out["degrees"] = float(dm.group("deg"))
        else:
            out["closure"] = (1.0 if _FINGER_CLOSE_VERB.search(
                m.group("verb")) else 0.0)
        return out

    m = _HAND.search(t)
    if m:
        # Real finger actuation now — the hands are live (2026-08-24). Open =
        # spread palm; close = fist. Side defaults to both, like the poses.
        verb = "open" if m.group("verb").lower().startswith("open") else "close"
        return {"kind": "hand", "state": verb,
                "side": (m.groupdict().get("side") or "both").lower()}

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
