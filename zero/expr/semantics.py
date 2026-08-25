"""Semantic hand mapper — hand shapes the words themselves call for.

Where the prosody analyzer answers WHEN the hands should move, this module
answers WHAT shape they should take: enumeration counts on fingers, size
words set the hands' aperture, epistemic hedges turn a palm up, negation
flicks the wrist away, temporal spans sweep it. Everything is expressed in
CLOSURE space (0 open .. 1 closed per finger) plus symbolic wrist intent,
so the per-side servo asymmetry stays the hand model's problem, never ours.

Precision over recall, the lesson the gesture layer already learned:
ordinary sentences should trigger NOTHING here — over-gesturing reads as
nervous. Each entry returns at most one gesture per sentence, positioned by
word index so the scheduler can land its apex on the nearest accent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FINGER_ORDER = ("index", "middle", "ring", "pinky", "thumb")   # counting order

# A hand plan: per-finger closure targets, symbolic wrist ("up"/"forward"/
# "in" or None = stay), amplitude of wrist offset for dynamic kinds, and how
# long the shape holds at apex.
@dataclass
class HandGesture:
    kind: str                       # count | aperture | palm_up | negation | sweep
    word_index: int                 # where in the sentence it belongs
    total_words: int
    closure: dict = field(default_factory=dict)   # finger -> 0..1 target
    wrist: str | None = None        # symbolic orientation at apex
    sides: tuple = ("left", "right")
    hold_s: float = 0.8
    magnitude: float = 1.0          # 0..1, scales amplitude (sizing)


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_COUNT = re.compile(
    r"\b(?P<n>one|two|three|four|five|[1-5])\s+"
    r"(?:things|reasons|steps|ways|points|options|parts|ideas|rules|kinds)\b",
    re.IGNORECASE)
_BIG = re.compile(
    r"\b(?:huge|massive|enormous|gigantic|really\s+big|this\s+big|so\s+(?:big|large))\b",
    re.IGNORECASE)
_SMALL = re.compile(
    r"\b(?:tiny|minuscule|really\s+small|this\s+small|so\s+small)\b",
    re.IGNORECASE)
# Deliberately WITHOUT "I think"/"I guess": those are among the most
# frequent phrases in conversational English, and palm-up firing at the
# rate cap through ordinary talk is exactly the over-gesturing this module
# forbids (audit expr #11). Only explicit uncertainty gestures.
_EPISTEMIC = re.compile(
    r"\b(?:maybe|perhaps|who\s+knows|not\s+sure|no\s+idea|"
    r"hard\s+to\s+say|could\s+be|can'?t\s+say)\b", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:no|never|not\s+at\s+all|absolutely\s+not|nope|no\s+way)\b[,.!]?",
    re.IGNORECASE)
_SWEEP = re.compile(
    r"\b(?:over\s+time|from\s+\w+\s+(?:to|until)\s+\w+|through\s+the\s+"
    r"(?:day|week|year[s]?)|step\s+by\s+step|all\s+the\s+way)\b", re.IGNORECASE)


def _word_index(text: str, start: int) -> int:
    return len(text[:start].split())


def analyze(text: str) -> HandGesture | None:
    """At most ONE semantic gesture per sentence — the first, strongest
    trigger. None for ordinary sentences, which is the common case."""
    if not text:
        return None
    total = max(1, len(text.split()))

    m = _COUNT.search(text)
    if m:
        raw = m.group("n").lower()
        n = _NUM_WORDS.get(raw, None)
        if n is None:
            try:
                n = int(raw)
            except ValueError:
                n = 0
        if 1 <= n <= 5:
            closure = {f: (0.0 if i < n else 1.0)
                       for i, f in enumerate(FINGER_ORDER)}
            return HandGesture("count", _word_index(text, m.start()), total,
                               closure=closure, wrist="forward",
                               sides=("right",), hold_s=1.0)

    m = _BIG.search(text)
    if m:
        return HandGesture("aperture", _word_index(text, m.start()), total,
                           closure={f: 0.0 for f in FINGER_ORDER},
                           wrist="forward", hold_s=0.8, magnitude=1.0)
    m = _SMALL.search(text)
    if m:
        return HandGesture("aperture", _word_index(text, m.start()), total,
                           closure={"thumb": 0.55, "index": 0.65,
                                    "middle": 1.0, "ring": 1.0, "pinky": 1.0},
                           wrist="forward", sides=("right",), hold_s=0.8,
                           magnitude=0.3)

    m = _EPISTEMIC.search(text)
    if m:
        return HandGesture("palm_up", _word_index(text, m.start()), total,
                           closure={f: 0.0 for f in FINGER_ORDER},
                           wrist="up", hold_s=0.9)

    m = _NEGATION.search(text)
    if m and _word_index(text, m.start()) <= 1:
        # meaningful at utterance-initial position ("No, ..."); a
        # mid-sentence "not" is grammar, not a gesture — and it must FALL
        # THROUGH to the later checks, not suppress them (audit expr #10).
        return HandGesture("negation", _word_index(text, m.start()),
                           total, closure={f: 0.15 for f in FINGER_ORDER},
                           wrist=None, sides=("right",), hold_s=0.5)

    m = _SWEEP.search(text)
    if m:
        return HandGesture("sweep", _word_index(text, m.start()), total,
                           closure={f: 0.1 for f in FINGER_ORDER},
                           wrist="forward", sides=("right",), hold_s=1.1)
    return None
