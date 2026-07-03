"""Turn-level speaker tracking — who is talking, and did that just change?

A single-mic robot can't do word-level diarization, but each utterance already
yields a voice embedding (identity, Phase 1). Comparing consecutive turns
catches the moment a DIFFERENT person takes over the conversation, so ZERO can
switch address ("oh hey Jane") instead of blindly continuing.
"""
from __future__ import annotations

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("perception.diarize")


class SpeakerTracker:
    def __init__(self, change_threshold: float = 0.40):
        self.change_threshold = float(change_threshold)
        self._last_emb: np.ndarray | None = None
        self._last_pid: int | None = None
        self._had_turn = False

    def reset(self) -> None:
        self._last_emb, self._last_pid, self._had_turn = None, None, False

    def update(self, *, person_id: int | None = None, name: str | None = None,
               voice_emb: np.ndarray | None = None) -> str:
        """Feed this turn's identity + voice embedding. Returns a note for the
        LLM ('' most turns): only when the speaker CHANGED mid-conversation."""
        note = ""
        changed = False
        if self._had_turn:
            if person_id is not None or self._last_pid is not None:
                changed = person_id != self._last_pid
            elif (voice_emb is not None and self._last_emb is not None
                  and voice_emb.size == self._last_emb.size):
                sim = float(np.dot(voice_emb, self._last_emb))
                changed = sim < self.change_threshold
        if changed:
            if name:
                note = f"(A different person is speaking now — it's {name}.)"
            else:
                note = "(A different person seems to be speaking now.)"
            log.info("speaker change detected -> %s", name or "unknown")
        self._had_turn = True
        self._last_pid = person_id
        if voice_emb is not None:
            self._last_emb = voice_emb
        return note
