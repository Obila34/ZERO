"""Speculative reply — start answering before the person has finished.

Humans don't wait for silence to begin planning a reply; they predict the end
of your sentence from prosody and start assembling their response, which is
why the gap between turns is ~200 ms rather than a full reaction time. This is
that, mechanically:

  1. At a pause, the turn model reads the waveform and says how likely it is
     the thought is complete.
  2. If it's confident AND the live transcript already reads like a whole
     request, we start generating a reply against that text — while the
     person may still be talking.
  3. When the turn really ends, we compare what they ACTUALLY said against
     what we bet on. Match -> the reply is already streaming, so it starts
     essentially instantly. Mismatch -> the bet is thrown away and we
     generate properly.

Step 3 is the part that matters and the part that is easy to get wrong. A
speculative reply that gets spoken after the premise changed is worse than any
latency — it is ZERO answering a question nobody asked. So the gate is strict
and closed by default: identical normalised text, or the speculation dies.
"""
from __future__ import annotations

import re
import threading

from zero.utils.logging import get_logger

log = get_logger("speculate")

_NORM_RE = re.compile(r"[^a-z0-9' ]+")
# A bet is only worth placing on something that already reads like a complete
# request. These are the shapes that mean "still going", so we hold off.
_TRAILING = frozenset((
    "and", "or", "but", "so", "because", "then", "if", "when", "while", "with",
    "to", "of", "for", "from", "in", "on", "at", "by", "as", "than", "the",
    "a", "an", "my", "your", "our", "their", "um", "uh", "like", "that",
    "is", "are", "was", "were", "can", "could", "would", "should", "i",
))


def normalize(text: str) -> str:
    """Compare on words, not punctuation/case — the recogniser is inconsistent
    about both, and neither changes what was asked."""
    return " ".join(_NORM_RE.sub(" ", (text or "").lower()).split())


def worth_speculating(text: str, min_words: int = 3) -> bool:
    """Is this partial transcript a sane thing to bet on? Long enough to carry
    an intent, and not visibly mid-sentence."""
    words = normalize(text).split()
    if len(words) < min_words:
        return False
    return words[-1] not in _TRAILING


class Speculation:
    """One in-flight bet: the text it was made on, its token stream, and the
    stop switch that abandons it."""

    def __init__(self, text: str, chunks, stop: threading.Event):
        self.text = text
        self.chunks = chunks
        self.stop = stop
        self._key = normalize(text)

    def matches(self, final_text: str) -> bool:
        """Only true when what they actually said is word-for-word what we bet
        on. Deliberately strict: a speculation that ran on 'book me a flight'
        must NOT be spoken when the finished sentence was 'book me a flight —
        actually, no, cancel that'. Extra words can invert the meaning, so
        'the speculation is a prefix' is not good enough."""
        return bool(self._key) and self._key == normalize(final_text)

    def abandon(self) -> None:
        """Kill the generation — the bet lost, or the turn was interrupted."""
        self.stop.set()
