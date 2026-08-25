"""SpeechTap — the Living Hands layer's only contact with the speech path.

Two calls, placed inline in main.py's speech loop (the declared contact
surface of docs/LIVING_HANDS_PLAN.md §05):

    TAP.audio(idx, sentence, piece, sr)   producer thread, per synthesized
                                          audio piece — BEFORE playout
    TAP.playout(idx, n_samples)           consumer, per piece yielded to the
                                          speaker — the playout clock

Both are unconditional no-ops until a listener attaches (feature-flagged in
build_expr), and both swallow everything: the speech path must be unable to
notice this module exists, whatever state it is in. That includes cost —
the unattached fast path is one attribute load and a None check.
"""
from __future__ import annotations


class SpeechTap:
    __slots__ = ("_listener",)

    def __init__(self):
        self._listener = None

    def attach(self, listener) -> None:
        """listener needs .on_audio(idx, sentence, piece, sr) and
        .on_playout(idx, n_samples). One listener; attach replaces."""
        self._listener = listener

    def detach(self) -> None:
        self._listener = None

    @property
    def attached(self) -> bool:
        return self._listener is not None

    def audio(self, idx: int, sentence: str, piece, sr: int) -> None:
        li = self._listener
        if li is None:
            return
        try:
            li.on_audio(idx, sentence, piece, sr)
        except Exception:
            pass                      # never into the synthesis thread

    def playout(self, idx: int, n_samples: int) -> None:
        li = self._listener
        if li is None:
            return
        try:
            li.on_playout(idx, n_samples)
        except Exception:
            pass                      # never into the playback generator


TAP = SpeechTap()
