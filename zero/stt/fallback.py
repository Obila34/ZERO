"""STT failover — try the primary (remote/GPU) engine, fall back to a local one.

The SSH tunnel to the GPU can drop mid-session. Without this wrapper a dead
tunnel makes ZERO silently ignore everything said to it. With it, every
utterance still tries the (better) primary first — so service recovers the
moment the tunnel is back — and only on failure runs the local engine, which is
built lazily on the first failure so startup never pays for a fallback that is
never needed.

EMPTY IS A FAILURE TOO (when the clip plainly holds speech). Kyutai's failure
mode at a live venue was not an exception — the marker came back with ZERO
words on quiet speech and on Swahili/Sheng (the model is EN/FR only). Every
consumer treated that empty success as "nothing was said": real interruptions
were discarded as noise and whole turns were dropped. So a primary that
returns '' on audio that clearly contains speech now escalates to the
fallback exactly like a raised error would.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from zero.stt.base import STT
from zero.utils.logging import get_logger

log = get_logger("stt.fallback")

# "Plainly contains speech": long enough to hold a word, loud enough that a
# recogniser returning nothing is suspicious. Below these, an empty result is
# taken at face value (breath, rustle, a genuine blip).
_MIN_SPEECH_S = 0.6
_MIN_SPEECH_RMS = 120.0   # int16 scale


def _has_clear_speech(audio: np.ndarray, sample_rate: int) -> bool:
    n = getattr(audio, "size", 0)
    if not n or n < sample_rate * _MIN_SPEECH_S:
        return False
    a = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(a))))
    if audio.dtype != np.int16:
        rms *= 32768.0
    return rms >= _MIN_SPEECH_RMS


class FallbackSTT(STT):
    def __init__(self, primary: STT, fallback_builder: Optional[Callable[[], STT]] = None):
        self._primary = primary
        self._builder = fallback_builder
        self._fallback: Optional[STT] = None
        self._fallback_broken = False  # builder failed once — don't retry every turn

    def _get_fallback(self) -> Optional[STT]:
        if self._fallback is None and self._builder is not None and not self._fallback_broken:
            try:
                log.info("building local fallback STT (first primary failure)...")
                self._fallback = self._builder()
            except Exception as e:
                self._fallback_broken = True
                log.error("could not build fallback STT — transcription will drop "
                          "while the primary is down: %s", e)
        return self._fallback

    degraded = False  # True while the last transcription used the fallback —
                      # self-state narration reads this

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        try:
            text = self._primary.transcribe(audio, sample_rate)
            if text.strip() or not _has_clear_speech(audio, sample_rate):
                self.degraded = False
                return text
            # Empty on audio that plainly holds speech: the Kyutai failure
            # mode. Escalate — a real interruption or a Swahili turn is on
            # the line. degraded stays False: the primary is UP, it just
            # couldn't hear this clip.
            log.warning("primary STT returned EMPTY on %.1fs of clear speech "
                        "— escalating to the fallback engine",
                        getattr(audio, "size", 0) / max(1, sample_rate))
            fallback = self._get_fallback()
            if fallback is None:
                return text
            try:
                return fallback.transcribe(audio, sample_rate)
            except Exception as e:
                log.error("fallback STT failed on the empty-rescue: %s", e)
                return text
        except Exception as e:
            log.warning("primary STT failed: %s", e)
        self.degraded = True
        fallback = self._get_fallback()
        if fallback is None:
            return ""
        try:
            return fallback.transcribe(audio, sample_rate)
        except Exception as e:
            log.error("fallback STT failed too: %s", e)
            return ""

    def rescue_transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Fallback-FIRST transcription, for when the primary has already
        demonstrably failed on this very audio (the live streaming session
        finalized empty). Retrying the primary in batch mode costs 3-5s to
        fail the same way — go straight to the engine that can hear it."""
        fallback = self._get_fallback()
        if fallback is not None:
            try:
                text = fallback.transcribe(audio, sample_rate)
                if text.strip():
                    return text
            except Exception as e:
                log.warning("rescue STT (fallback engine) failed: %s", e)
        try:
            return self._primary.transcribe(audio, sample_rate)
        except Exception as e:
            log.warning("rescue STT (primary retry) failed too: %s", e)
            return ""

    def close(self) -> None:
        self._primary.close()
        if self._fallback is not None:
            self._fallback.close()
