"""Utterance endpointing: after the wake word, capture audio until the user
stops talking (trailing silence) or a hard length cap is hit.

Two backends:
  * silero  — neural VAD, robust to noise (preferred). Loaded via torch.
  * webrtc  — webrtcvad, ultra-light, no torch. Fallback.

Both expose the same `capture()` API: consume frames from an iterator, return
the collected utterance as a float32 mono array (or None if nothing was said).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("vad")


class _BaseEndpointer:
    def __init__(self, sample_rate: int, silence_ms: int, max_utterance_ms: int,
                 speech_pad_ms: int, block_ms: int):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.silence_blocks = max(1, silence_ms // block_ms)
        self.max_blocks = max(1, max_utterance_ms // block_ms)
        self.pad_blocks = max(0, speech_pad_ms // block_ms)

    def _is_speech(self, frame: np.ndarray) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def capture(self, frames: Iterable[np.ndarray]) -> np.ndarray | None:
        collected: list[np.ndarray] = []
        trailing_silence = 0
        started = False
        n = 0

        for frame in frames:
            n += 1
            speech = self._is_speech(frame)

            if speech:
                started = True
                trailing_silence = 0
            elif started:
                trailing_silence += 1

            if started:
                collected.append(frame)

            if started and trailing_silence >= self.silence_blocks:
                break
            if n >= self.max_blocks:
                log.info("utterance hit max length cap")
                break

        if not collected:
            return None
        pcm = np.concatenate(collected).astype(np.float32) / 32768.0
        return pcm


class SileroEndpointer(_BaseEndpointer):
    def __init__(self, **kw):
        super().__init__(**kw)
        import torch  # lazy

        self._torch = torch
        # silero-vad ships via torch.hub; cache after first download.
        model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._model = model
        self._threshold = 0.5
        log.info("silero VAD loaded")

    def _is_speech(self, frame: np.ndarray) -> bool:
        # silero wants float32 in [-1, 1]; it expects 512-sample chunks at 16 kHz.
        x = self._torch.from_numpy(frame.astype(np.float32) / 32768.0)
        prob = float(self._model(x, self.sample_rate).item())
        return prob >= self._threshold


class WebrtcEndpointer(_BaseEndpointer):
    def __init__(self, **kw):
        super().__init__(**kw)
        import webrtcvad  # lazy

        self._vad = webrtcvad.Vad(2)  # 0-3, higher = more aggressive filtering
        log.info("webrtc VAD loaded")

    def _is_speech(self, frame: np.ndarray) -> bool:
        # webrtcvad needs 10/20/30 ms int16 frames at 8/16/32/48 kHz.
        return self._vad.is_speech(frame.tobytes(), self.sample_rate)


def build_endpointer(engine: str, **kw) -> _BaseEndpointer:
    if engine == "webrtc":
        return WebrtcEndpointer(**kw)
    return SileroEndpointer(**kw)
