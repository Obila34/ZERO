"""Parse spoken arm/hand commands into a gesture intent — the FAST lexical path.
Supports:
  1. Hand gestures ("peace sign", "I love you sign", "point", "thumbs up", "make a fist", "open hands", "ok sign", "rock on", "pinch")
  2. Individual finger control ("move your thumb", "bend right index finger", "open left pinky", "wiggle fingers")
  3. Kenyan Sign Language (KSL) Letters ("show me letter K", "sign the letter A")
  4. Sign Language Word Spelling ("spell PETER", "spell COW", "fingerspell ROBOT")
  5. Arm & Shoulder movements ("raise arm", "combat guard", "punch", "wave")
"""
from __future__ import annotations
import re

_SIDE = r"(?P<side>right|left|both)?"
STEP_DEG = 30.0
SMALL_DEG = 10.0

def set_default_step(deg: float) -> None:
    global STEP_DEG, SMALL_DEG
    STEP_DEG = float(deg)
    SMALL_DEG = max(2.0, float(deg) / 3.0)

# Action Gestures
_WAVE = re.compile(rf"\b(?:wave|waving)\b(?:\s+(?:your|the)\s+(?:{_SIDE}\s+)?(?:hand|arm))?", re.IGNORECASE)
_PUNCH = re.compile(r"\b(?:punch|strike|throw a punch|jab)\b", re.IGNORECASE)
_GUARD = re.compile(r"\b(?:guard|combat guard|hands up|defend)\b", re.IGNORECASE)
_DOWN = re.compile(rf"\b(?:(?:put|lower|drop|rest)(?:ing)?\s+(?:down\s+)?(?:your\s+|the\s+)?(?:{_SIDE}\s+)?(?:arm|hand)s?(?:\s+down)?|arms?\s+down)\b", re.IGNORECASE)

# Hand & Finger Gestures
_ILY = re.compile(rf"\b(?:i\s+love\s+you(?:\s+sign)?|love\s+sign|ily|sign\s+i\s+love\s+you|show\s+(?:me\s+)?i\s+love\s+you)\b", re.IGNORECASE)
_PEACE = re.compile(rf"\b(?:peace(?:\s+sign)?|victory(?:\s+sign)?|show\s+(?:me\s+)?(?:the\s+)?peace\s+sign|sign\s+peace)\b", re.IGNORECASE)
_POINT = re.compile(rf"\b(?:point|pointing)\b(?:\s+(?:with\s+)?(?:your|the\s+)?(?:{_SIDE}\s+)?(?:hand|finger|index))?", re.IGNORECASE)
_THUMBS_UP = re.compile(rf"\b(?:thumbs?\s+up|give\s+(?:me\s+)?a\s+thumbs?\s+up)\b", re.IGNORECASE)
_FIST = re.compile(rf"\b(?:(?:make\s+a\s+)?fist|close\s+(?:your|the)?\s*(?:{_SIDE}\s*)?(?:hands?|fists?|fingers?))\b", re.IGNORECASE)
_OPEN_HAND = re.compile(rf"\b(?:open\s+(?:your|the)?\s*(?:{_SIDE}\s*)?(?:hands?|fingers?)|high\s*five)\b", re.IGNORECASE)
_OK_SIGN = re.compile(rf"\b(?:ok\s+sign|okay\s+sign|give\s+(?:me\s+)?(?:an\s+)?ok)\b", re.IGNORECASE)
_ROCK_ON = re.compile(rf"\b(?:rock\s+on|horns|rock\s+sign)\b", re.IGNORECASE)
_PINCH = re.compile(rf"\b(?:pinch|pinching)\b", re.IGNORECASE)
_WIGGLE = re.compile(rf"\b(?:wiggle|wave)\s+(?:your|the)?\s*(?:{_SIDE}\s*)?fingers?\b", re.IGNORECASE)

# Individual Finger Movement ("move thumb", "bend index", "open pinky", "move middle finger")
_FINGER_ACTION = re.compile(
    rf"\b(?P<act>move|bend|open|close|wiggle|flex)\s+(?:your|the)?\s*(?:{_SIDE}\s*)?"
    rf"(?P<finger>thumb|index(?:\s+finger)?|middle(?:\s+finger)?|ring(?:\s+finger)?|pinky(?:\s+finger)?|pinkie)"
    rf"(?:\s+by\s+(?P<deg>\d+)(?:\s*degrees?)?)?\b",
    re.IGNORECASE
)

# Spelling & Letters
_ASL_SPELL = re.compile(
    r"\b(?:spell|fingerspell|how\s+do\s+you\s+spell|can\s+you\s+spell|sign\s+the\s+word)\s+"
    r"(?:the\s+(?:name|word)\s+)?(?P<word>[a-zA-Z0-9]+)\b",
    re.IGNORECASE
)

_ASL_LETTER = re.compile(
    r"\b(?:show\s+(?:me\s+)?(?:the\s+)?letter|sign\s+(?:the\s+)?letter|how\s+do\s+you\s+sign(?:\s+the\s+letter)?|"
    r"what\s+is\s+(?:the\s+letter\s+)?(?P<l_direct>[a-zA-Z])\s+in\s+(?:sign\s+language|ksl|asl))\s*(?P<letter>[a-zA-Z])?\b",
    re.IGNORECASE
)

_SPELL_CORRECTIONS = {
    "PIT": "PETER",
    "PITA": "PETER",
}

def parse_arm_command(text: str) -> dict | None:
    t = text.strip()
    if not t:
        return None

    # 1. I Love You (ILY) Sign
    if _ILY.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "i_love_you", "side": side}

    # 2. Peace Sign
    if _PEACE.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "peace", "side": side}

    # 3. Spelling
    m_spell = _ASL_SPELL.search(t)
    if m_spell:
        w = m_spell.group("word").upper()
        w = _SPELL_CORRECTIONS.get(w, w)
        if w not in ("YOUR", "THE", "IT", "THAT", "ME", "WORD", "NAME", "THUMB", "INDEX", "FINGER", "HAND", "PEACE", "LOVE"):
            return {"kind": "spell", "word": w}

    # 4. Single Letter
    m_let = _ASL_LETTER.search(t)
    if m_let:
        let = m_let.group("letter") or m_let.group("l_direct")
        if let:
            return {"kind": "asl_letter", "letter": let.upper()}

    # 5. Hand Gestures
    if _POINT.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "point", "side": side}

    if _THUMBS_UP.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "thumbs_up", "side": side}

    if _FIST.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "fist", "side": side}

    if _OPEN_HAND.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "open_hands", "side": side}

    if _OK_SIGN.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "ok_sign", "side": side}

    if _ROCK_ON.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "rock_on", "side": side}

    if _PINCH.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "pinch", "side": side}

    if _WIGGLE.search(t):
        side = "left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both")
        return {"kind": "gesture", "name": "wiggle", "side": side}

    # 6. Individual Finger Control ("move thumb", "bend index", "open pinky")
    m_fa = _FINGER_ACTION.search(t)
    if m_fa:
        act = m_fa.group("act").lower()
        finger_raw = m_fa.group("finger").lower().replace(" finger", "").strip()
        finger = "pinky" if finger_raw == "pinkie" else finger_raw
        side = m_fa.group("side") or ("left" if "left" in t.lower() else ("right" if "right" in t.lower() else "both"))
        deg_str = m_fa.group("deg")
        if deg_str:
            target_deg = float(deg_str)
        else:
            target_deg = 90.0 if act == "open" else (0.0 if finger != "thumb" else 140.0)
        return {"kind": "finger", "finger": finger, "side": side, "degrees": target_deg, "action": act}

    # 7. Arm Gestures
    if _WAVE.search(t):
        side = "left" if "left" in t.lower() else "right"
        return {"kind": "gesture", "name": f"wave_{side}"}

    if _PUNCH.search(t):
        return {"kind": "gesture", "name": "punch"}

    if _GUARD.search(t):
        return {"kind": "gesture", "name": "combat_guard"}

    if _DOWN.search(t):
        return {"kind": "gesture", "name": "rest"}

    return None
