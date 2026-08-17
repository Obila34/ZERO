"""Gesture cues — how the language layer asks for a hand gesture.

The model writes a cue INLINE in its reply, exactly where the gesture belongs:

    "Yeah [beat] the big one, over on the left [point_left]."

That inline position IS the synchronisation: the cue sits against the stressed
word it supports, so the gesture fires as that word reaches the speaker. ZERO
streams speech sentence-by-sentence (first audio out ~1.3 s while the model is
still generating), so a JSON envelope carrying a word index would mean holding
the whole reply back before a single sound came out — the cue convention gets
the same tight coupling without paying that latency. It is also the convention
the model already uses for [laughs]/[sighs], so no new protocol is invented.

Gestures are classified by FUNCTION, following gesture-studies practice:

  BEAT       — short rhythmic motion on a stressed word. Marks structure
               ("first... then..."), not content. The commonest real gesture.
  DEICTIC    — pointing. Only ever at something actually perceived; the arm
               layer refuses a direction it cannot ground.
  ICONIC     — depicts size/shape/motion the words describe. Rare: it must add
               information the words don't already carry.
  EMBLEMATIC — conventional and stand-alone (wave, shrug). Unambiguous alone.

IDLE is the default and most turns should have no cue at all — over-gesturing
reads as far less natural than under-gesturing, so the arm layer additionally
rate-limits (see ArmSystem.express).

Cue -> gesture name; the gesture itself must exist and its joints must be
calibrated, or ArmSystem refuses it out loud rather than faking success.
"""
from __future__ import annotations

import re

# The vocabulary the model is taught (zero/main.py builds the prompt block).
CUE_TO_GESTURE: dict[str, str] = {
    "[wave]": "wave_right",          # emblematic: greeting / farewell
    "[shrug]": "shrug",              # emblematic: "no idea", "up to you"
    "[beat]": "beat_right",          # beat: rhythmic emphasis
    "[beat_both]": "beat_both",      # beat: heavier emphasis, both hands
    "[point_left]": "point_left",    # deictic: something to ZERO's left
    "[point_right]": "point_right",  # deictic: something to ZERO's right
    "[show_big]": "show_big",        # iconic: size
    "[offer]": "offer_right",        # emblematic: "here", "after you"
}

# Function class per cue — the arm layer uses this for grounding rules
# (deictics need a perceived target) and for pacing.
CUE_FUNCTION: dict[str, str] = {
    "[wave]": "emblematic",
    "[shrug]": "emblematic",
    "[offer]": "emblematic",
    "[beat]": "beat",
    "[beat_both]": "beat",
    "[point_left]": "deictic",
    "[point_right]": "deictic",
    "[show_big]": "iconic",
}

_CUE_RE = re.compile(r"\[(?:" + "|".join(
    c[1:-1] for c in sorted(CUE_TO_GESTURE, key=len, reverse=True)) + r")\]")


def find_cues(text: str) -> list[str]:
    """Gesture cues in a reply, in the order they appear."""
    if not text:
        return []
    return [m.group(0).lower() for m in _CUE_RE.finditer(text.lower())]


def cue_positions(text: str) -> list[tuple[str, int, int]]:
    """[(cue, word_index, total_words)] — WHERE in the sentence each cue sits.

    This is the timing information: McNeill's phonological synchrony rule says
    the stroke coincides with, or slightly precedes, the stressed syllable of
    the word it belongs to, and never follows it. The cue's word position is
    what lets the stroke be scheduled onto that word instead of firing at the
    start of the sentence regardless.
    """
    if not text:
        return []
    out, words = [], 0
    for tok in text.split():
        low = tok.lower()
        hit = _CUE_RE.fullmatch(low) or _CUE_RE.search(low)
        if hit and low.startswith("["):
            out.append((hit.group(0), words, 0))   # cue stands alone
        else:
            words += 1
            if hit:                                # cue attached to a word
                out.append((hit.group(0), max(0, words - 1), 0))
    return [(c, i, words) for c, i, _ in out]


def strip_cues(text: str) -> str:
    """The same text with gesture cues removed, for the speech path.

    Load-bearing: an unrecognised [cue] reaching the Piper synthesiser is
    spoken aloud as a word, so ZERO would literally say "wave".
    """
    if not text:
        return text
    out = _CUE_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", out).strip()


def prompt_block(gestures: list[str]) -> str:
    """The system-prompt section teaching the cue vocabulary, limited to the
    gestures that are actually available (calibrated) this session. Returns ""
    when nothing can move, so the model is never taught a gesture it cannot
    perform and never promises one."""
    usable = [(c, g) for c, g in CUE_TO_GESTURE.items() if g in set(gestures)]
    if not usable:
        return ""
    lines = "\n".join(
        f"  {cue} — {CUE_FUNCTION[cue]}" for cue, _g in usable)
    return (
        "Your hands: you have real arms and can gesture while you speak. Put a "
        "cue inline in your reply at the exact word the gesture belongs to — "
        "the hand moves as that word is spoken:\n" + lines + "\n"
        "Gesture RARELY. Most replies should carry no cue at all; a hand that "
        "moves on every sentence reads as nervous, not alive. Never point at "
        "something you cannot actually see, never use more than one cue in a "
        "sentence, and never mention the gesture in words — the cue is not "
        "spoken, so write it instead of describing what you're doing."
    )


# ── inferring a gesture from what ZERO is actually saying ───────────────────
# The model rarely writes a cue, so without this only the automatic beat ever
# fires and the other four gesture classes are dead code. These read the
# CONTENT of a sentence and pick the gesture a person would make there —
# emblematic for greetings and shrugs, iconic for size, deictic for pointing.
# Each pattern also yields the word it belongs to, so the stroke can be placed
# on that word (the phonological synchrony rule) rather than the sentence.
_INFER: list[tuple[str, "re.Pattern"]] = [
    ("wave_right", re.compile(
        r"\b(?:hi|hey|hello|good\s+(?:morning|afternoon|evening)|"
        r"nice\s+to\s+(?:meet|see)\s+you|welcome|"
        r"bye|goodbye|see\s+you|take\s+care)\b", re.IGNORECASE)),
    ("shrug", re.compile(
        r"\b(?:i\s+(?:don'?t|do\s+not)\s+know|no\s+idea|not\s+sure|"
        r"who\s+knows|either\s+way|up\s+to\s+you|hard\s+to\s+say|"
        r"i'?m\s+not\s+certain)\b", re.IGNORECASE)),
    ("show_big", re.compile(
        r"\b(?:huge|massive|enormous|gigantic|tiny|enormously|"
        r"so\s+(?:big|small|large)|this\s+big)\b", re.IGNORECASE)),
    ("offer_right", re.compile(
        r"\b(?:here\s+you\s+(?:go|are)|after\s+you|be\s+my\s+guest|"
        r"go\s+ahead|help\s+yourself)\b", re.IGNORECASE)),
    ("point_right", re.compile(
        r"\b(?:over\s+there|right\s+there|that\s+one|that\s+way|"
        r"just\s+there)\b", re.IGNORECASE)),
    ("point_left", re.compile(r"\b(?:over\s+here|this\s+one|right\s+here)\b",
                              re.IGNORECASE)),
]


def infer_gesture(text: str) -> tuple[str, int] | None:
    """(gesture, word_index) a sentence's own words call for, or None.

    Only fires on phrases where a person would reliably gesture; ordinary
    sentences get nothing, because most speech carries no gesture at all and
    over-gesturing reads as far less natural than under-gesturing.
    """
    if not text:
        return None
    best = None
    for name, pat in _INFER:
        m = pat.search(text)
        if m and (best is None or m.start() < best[2]):
            word_index = len(text[:m.start()].split())
            best = (name, word_index, m.start())
    return (best[0], best[1]) if best else None
