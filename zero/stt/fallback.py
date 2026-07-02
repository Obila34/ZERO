"""STT failover — try the primary (remote/GPU) engine, fall back to a local one.

The SSH tunnel to the GPU can drop mid-session. Without this wrapper a dead
tunnel makes ZERO silently ignore everything said to it. With it, every
utterance still tries the (better) primary first — so service recovers the
moment the tunnel is back — and only on failure runs the local engine, which is
built lazily on the first failure so startup never pays for a fallback that is
never needed.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from zero.stt.base import STT
from zero.utils.logging import get_logger

log = get_logger("stt.fallback")


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

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        try:
            return self._primary.transcribe(audio, sample_rate)
        except Exception as e:
            log.warning("primary STT failed: %s", e)
        fallback = self._get_fallback()
        if fallback is None:
            return ""
        try:
            return fallback.transcribe(audio, sample_rate)
        except Exception as e:
            log.error("fallback STT failed too: %s", e)
            return ""

    def close(self) -> None:
        self._primary.close()
        if self._fallback is not None:
            self._fallback.close()
