"""TTS failover — try the primary (remote/GPU) voice, fall back to a local one.

If the SSH tunnel to the Orpheus server drops, ZERO would otherwise go mute with
no explanation. This wrapper tries the primary for every sentence (so the good
voice comes back the moment the tunnel does) and, when the primary produces no
audio, speaks through a lazily-built local engine (Piper) instead. The different
voice IS the signal to the user that something is degraded.

Two impedance mismatches are handled here so the rest of the pipeline never
notices the swap:

* Sample rate — playback opens its stream at the primary's rate, so fallback
  audio is resampled to match (e.g. Piper 22.05 kHz -> Orpheus 24 kHz).
* Cue tags — the orchestrator passes cue-tagged text straight through on the
  Orpheus path, but a plain local engine would read "[laughs]" aloud, so tags
  and *emphasis* are stripped before the fallback speaks.
"""
from __future__ import annotations

import re
from typing import Callable, Iterator, Optional

import numpy as np

from zero.tts.base import TTS, resample_linear
from zero.utils.logging import get_logger

log = get_logger("tts.fallback")

_CUE = re.compile(r"\[[a-z]+\]")
_EMPHASIS = re.compile(r"\*(\w+)\*")


def _plain_text(text: str) -> str:
    """Strip cue tags and emphasis for an engine that can't perform them."""
    return " ".join(_EMPHASIS.sub(r"\1", _CUE.sub(" ", text)).split())


class FallbackTTS(TTS):
    def __init__(self, primary: TTS, fallback_builder: Optional[Callable[[], TTS]] = None):
        self._primary = primary
        self._builder = fallback_builder
        self._fallback: Optional[TTS] = None
        self._fallback_broken = False  # builder failed once — don't retry every turn
        self.degraded = False  # True while the last utterance used the fallback
                               # voice — self-state narration reads this

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self._primary.sample_rate

    def prewarm(self) -> None:
        """Pass a pre-connect request through to the primary engine (the one
        whose connect latency sits on the first-audio path)."""
        pw = getattr(self._primary, "prewarm", None)
        if callable(pw):
            pw()

    def _get_fallback(self) -> Optional[TTS]:
        if self._fallback is None and self._builder is not None and not self._fallback_broken:
            try:
                log.info("building local fallback TTS (first primary failure)...")
                self._fallback = self._builder()
            except Exception as e:
                self._fallback_broken = True
                log.error("could not build fallback TTS — ZERO will be mute while "
                          "the primary is down: %s", e)
        return self._fallback

    def _fallback_audio(self, text: str) -> np.ndarray:
        fallback = self._get_fallback()
        if fallback is None:
            return np.zeros(0, dtype=np.float32)
        try:
            audio = fallback.synthesize(_plain_text(text))
        except Exception as e:
            log.error("fallback TTS failed too: %s", e)
            return np.zeros(0, dtype=np.float32)
        return resample_linear(audio, fallback.sample_rate, self.sample_rate)

    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        audio = np.zeros(0, dtype=np.float32)
        try:
            audio = self._primary.synthesize(text)
        except Exception as e:
            log.warning("primary TTS failed: %s", e)
        if getattr(audio, "size", 0):
            self.degraded = False
            return audio
        # Non-empty text but no audio back = the primary is down.
        self.degraded = True
        return self._fallback_audio(text)

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        yielded = False
        try:
            for piece in self._primary.synthesize_stream(text):
                yielded = True
                yield piece
        except Exception as e:
            log.warning("primary TTS stream failed: %s", e)
        if yielded:
            self.degraded = False
            return  # got at least partial audio; don't repeat the sentence
        self.degraded = True
        audio = self._fallback_audio(text)
        if getattr(audio, "size", 0):
            yield audio

    def close(self) -> None:
        self._primary.close()
        if self._fallback is not None:
            self._fallback.close()
