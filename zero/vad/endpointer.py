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
                 speech_pad_ms: int, block_ms: int, min_speech_ms: int = 350,
                 threshold: float = 0.6):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.threshold = threshold  # used by silero; webrtc ignores it
        self.min_speech_blocks = max(1, min_speech_ms // block_ms)
        self.silence_blocks = max(1, silence_ms // block_ms)
        self.max_blocks = max(1, max_utterance_ms // block_ms)
        self.pad_blocks = max(0, speech_pad_ms // block_ms)

    def _is_speech(self, frame: np.ndarray) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def _reset(self) -> None:
        """Hook: clear per-utterance state before a new capture. Optional."""

    def capture(self, frames: Iterable[np.ndarray],
                start_timeout_ms: int | None = None) -> np.ndarray | None:
        """Collect one utterance. If start_timeout_ms is set and no speech begins
        within that window, return None (used by conversation mode to go to sleep)."""
        self._reset()
        collected: list[np.ndarray] = []
        trailing_silence = 0
        speech_blocks = 0
        started = False
        pre_speech = 0
        n = 0
        start_timeout_blocks = (start_timeout_ms // self.block_ms) if start_timeout_ms else None

        for frame in frames:
            n += 1
            speech = self._is_speech(frame)

            if speech:
                started = True
                trailing_silence = 0
                speech_blocks += 1
            elif started:
                trailing_silence += 1
            else:
                pre_speech += 1
                if start_timeout_blocks and pre_speech >= start_timeout_blocks:
                    return None  # nobody spoke in time

            if started:
                collected.append(frame)

            if started and trailing_silence >= self.silence_blocks:
                break
            if n >= self.max_blocks:
                log.info("utterance hit max length cap")
                break

        if not collected:
            return None
        # Reject blips of noise: not enough actual speech to be a real utterance.
        # This is what stops whisper hallucinating "Thank you" / "..." on silence.
        if speech_blocks < self.min_speech_blocks:
            log.info("utterance too short (%d ms speech) — ignoring",
                     speech_blocks * self.block_ms)
            return None
        pcm = np.concatenate(collected).astype(np.float32) / 32768.0
        return pcm


class SileroEndpointer(_BaseEndpointer):
    # silero v5 requires EXACTLY this many samples per call at 16 kHz (256 at 8 kHz).
    WINDOW = 512

    def __init__(self, **kw):
        super().__init__(**kw)
        import torch  # lazy

        self._torch = torch
        # silero-vad ships via torch.hub; cache after first download.
        model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._model = model
        self._window = self.WINDOW if self.sample_rate == 16000 else 256
        self._buf = np.zeros(0, dtype=np.int16)
        self._last = False
        log.info("silero VAD loaded")

    def _reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        self._last = False
        try:
            self._model.reset_states()
        except Exception:  # noqa: BLE001
            pass

    def _is_speech(self, frame: np.ndarray) -> bool:
        # Our frames are 480 samples (30 ms) but silero needs exact 512-sample
        # windows. Buffer incoming audio and run the model on each full window.
        self._buf = np.concatenate([self._buf, frame])
        while len(self._buf) >= self._window:
            chunk = self._buf[: self._window]
            self._buf = self._buf[self._window :]
            x = self._torch.from_numpy(chunk.astype(np.float32) / 32768.0)
            prob = float(self._model(x, self.sample_rate).item())
            self._last = prob >= self.threshold
        return self._last


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
