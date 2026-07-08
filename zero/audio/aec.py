"""Acoustic echo cancellation (speexdsp) — subtract ZERO's own playback from
the mic signal, so the mic can stay fully live while ZERO speaks: cleaner
barge-in, and the groundwork for backchannels during the user's own speech.

Wiring: Speaker pushes every chunk it plays into an ``EchoReference`` ring
(resampled to the mic rate); MicCapture pulls the matching far-end frame and
runs speex NLMS cancellation per block. Speex adapts to a fixed device delay
(wired jack/speaker); a Bluetooth sink's drifting latency is beyond its filter
— keep AEC off for BT.

Opt-in (audio.aec.enabled) and dependency-guarded: needs `pip install
speexdsp` (apt: libspeexdsp-dev first). Any failure disables AEC, never audio.
"""
from __future__ import annotations

import threading

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("audio.aec")


class EchoReference:
    """Thread-safe ring of what the speaker is playing, kept at the mic rate."""

    def __init__(self, sample_rate: int, max_seconds: float = 2.0):
        self._sr = int(sample_rate)
        self._max = int(sample_rate * max_seconds)
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray, src_rate: int) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if src_rate != self._sr and chunk.size:
            n = max(1, int(round(chunk.size * self._sr / src_rate)))
            chunk = np.interp(np.linspace(0, chunk.size - 1, n),
                              np.arange(chunk.size), chunk).astype(np.float32)
        with self._lock:
            self._buf = np.concatenate([self._buf, chunk])[-self._max:]

    def pop(self, n: int) -> np.ndarray:
        """Oldest ``n`` samples as int16 (zero-padded when playback is idle)."""
        with self._lock:
            take, self._buf = self._buf[:n], self._buf[n:]
        out = np.zeros(n, dtype=np.float32)
        out[:take.size] = take
        return np.clip(out * 32768.0, -32768, 32767).astype(np.int16)


class SpeexAEC:
    """Per-frame NLMS echo canceller. Raises at build if speexdsp is missing —
    the caller logs and runs without AEC."""

    def __init__(self, ref: EchoReference, frame_size: int, sample_rate: int,
                 filter_ms: int = 200):
        from speexdsp import EchoCanceller  # hard dep only when enabled

        self._ref = ref
        self._frame = int(frame_size)
        self._ec = EchoCanceller.create(
            self._frame, int(sample_rate * filter_ms / 1000), int(sample_rate))
        log.info("speex AEC active (frame=%d, filter=%dms)", frame_size, filter_ms)

    def process(self, frame_i16: np.ndarray) -> np.ndarray:
        """Cancel the echo in one mic frame. Never raises; falls back to the
        raw frame so a hiccup can't deafen the pipeline."""
        try:
            if frame_i16.size != self._frame:
                return frame_i16  # resampled odd-size block — pass through
            far = self._ref.pop(self._frame)
            out = self._ec.process(frame_i16.tobytes(), far.tobytes())
            return np.frombuffer(out, dtype=np.int16)
        except Exception as e:
            log.debug("AEC frame failed: %s", e)
            return frame_i16
