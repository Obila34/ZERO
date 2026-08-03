"""Interrupt classification — what did the user MEAN by talking over ZERO?

Humans don't treat every overlap as "stop talking". The taxonomy here mirrors
the conversational reality:

* BACKCHANNEL — "yeah", "mhm", "right": active listening. The reply should
  keep flowing (the caller un-ducks and carries on) — stopping on these is
  exactly the "freezes when you agree with it" failure of smart speakers.
* CORRECTION — "no, I meant—", "wait, stop": a genuine "you're wrong / stop"
  signal. The reply is cancelled immediately, mid-word.
* QUEUE — anything else: a new thought. Like a polite human, ZERO finishes
  the sentence it's saying, then responds to the new input.
* NOISE — empty/junk transcript: ignore, keep talking at full volume.

Classification is lexical and instant (<1 ms) — the latency budget can't
afford an LLM round trip, and the FIRST words of an interruption carry the
intent ("no," / "wait" / "actually") almost unambiguously in practice.
"""
from __future__ import annotations

import re
from enum import Enum


class InterruptKind(Enum):
    NOISE = "noise"
    BACKCHANNEL = "backchannel"
    QUEUE = "queue"
    CORRECTION = "correction"


# Hard stop/correction anywhere in the utterance — these are commands about
# THIS reply, not content ("okay stop", "no no stop talking").
_HARD_RE = re.compile(
    r"\b(?:stop|shut up|be quiet|quiet|shush|enough|never mind|nevermind|"
    r"cancel that|forget it)\b")

# Correction openers — only meaningful at the START of the interruption
# ("no, I meant Tuesday"), because a "no" deep inside a sentence is usually
# content ("there's no milk left").
_CORRECTION_OPEN_RE = re.compile(
    r"^(?:no+|nope|nah|wait|whoa|woah|hold on|hang on|actually|"
    r"that'?s (?:wrong|not (?:it|right|what i))|not (?:that|what i)|"
    r"i (?:said|meant|didn'?t (?:say|mean))|wrong|listen)\b")

# The active-listening lexicon. An interruption made ONLY of these (and at
# most ~3 words) is agreement/acknowledgement, not a bid for the floor.
_BACKCHANNEL_WORDS = frozenset((
    "yeah", "yep", "yes", "ya", "mhm", "mm", "hmm", "mmhmm", "mm-hmm",
    "uh-huh", "uhhuh", "right", "ok", "okay", "kay", "sure", "true", "wow",
    "whoa", "nice", "cool", "great", "good", "huh", "really", "interesting",
    "exactly", "totally", "definitely", "gotcha", "i", "see", "got", "it",
    "makes", "sense", "oh", "ah", "aha", "fair",
))


def classify_interrupt(text: str | None) -> InterruptKind:
    """Instant lexical classification of an interruption transcript.
    Never raises; unknown/odd input degrades to QUEUE (the polite default)."""
    t = (text or "").strip().lower()
    # Whisper writes hallucinated punctuation-only / one-letter blips on noise.
    words = re.findall(r"[a-z']+", t)
    if not words:
        return InterruptKind.NOISE
    if _HARD_RE.search(t):
        return InterruptKind.CORRECTION
    if _CORRECTION_OPEN_RE.match(t):
        return InterruptKind.CORRECTION
    if len(words) <= 3 and all(w in _BACKCHANNEL_WORDS for w in words):
        return InterruptKind.BACKCHANNEL
    return InterruptKind.QUEUE
