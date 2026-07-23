"""openWakeWord engine — always-on, low CPU, supports custom wake words.

Heavy deps are imported lazily inside __init__ so the package imports fine on a
dev machine without the model installed.
"""
from __future__ import annotations

import numpy as np

from zero.utils.logging import get_logger
from zero.wake.base import WakeWord

log = get_logger("wake.oww")


# openWakeWord's models are trained on 80 ms windows at 16 kHz. Feeding smaller
# chunks (our mic hands out 30 ms / 480-sample frames because WebRTC VAD needs
# them small) makes some versions return an empty scores dict — and then the
# wake word can never fire. Buffer frames here until we have a full chunk.
_CHUNK_SAMPLES = 1280


class OpenWakeWordEngine(WakeWord):
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5):
        from openwakeword.model import Model  # lazy

        self.threshold = threshold
        self.model_name = model
        # Force the ONNX backend: tflite-runtime has no wheel for Python 3.13,
        # so onnxruntime is the only viable inference path on a current Pi OS.
        # Pass a known keyword name or a path to a custom .onnx model.
        kwargs = {"inference_framework": "onnx"}
        self._model = (
            Model(wakeword_models=[model], **kwargs) if model else Model(**kwargs)
        )
        self._buf: np.ndarray = np.empty(0, dtype=np.int16)
        # Diagnostics window: without this the idle loop is a black box — a
        # near-miss (score 0.4) and dead silence (score 0.0, wrong mic) look
        # identical in the logs but need opposite fixes.
        self._win_chunks = 0
        self._win_score = 0.0
        self._win_rms = 0.0
        log.info("openWakeWord loaded (model=%s, threshold=%.2f)", model, threshold)

    def process(self, frame: np.ndarray) -> bool:
        # openWakeWord expects int16 mono samples in 1280-sample chunks. Small
        # mic frames get concatenated; we run predict once per completed chunk
        # and fire on the FIRST chunk to cross threshold in this call.
        if frame.dtype != np.int16:
            frame = frame.astype(np.int16, copy=False)
        self._buf = np.concatenate((self._buf, frame)) if self._buf.size else frame
        fired = False
        while self._buf.size >= _CHUNK_SAMPLES:
            chunk, self._buf = self._buf[:_CHUNK_SAMPLES], self._buf[_CHUNK_SAMPLES:]
            scores = self._model.predict(chunk)
            score = max(scores.values()) if scores else 0.0
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            self._win_score = max(self._win_score, score)
            self._win_rms = max(self._win_rms, rms)
            self._win_chunks += 1
            if self._win_chunks >= 50:  # 50 x 80ms = every ~4s of audio
                # Near-misses surface at INFO so a mistuned threshold/gain is
                # visible on a normal run; the quiet heartbeat stays at DEBUG
                # (run with ZERO_LOG=DEBUG to watch levels continuously).
                msg = ("wake window: top score %.2f (threshold %.2f), "
                       "peak rms %.0f")
                if self._win_score >= 0.5 * self.threshold:
                    log.info(msg + " — NEAR MISS", self._win_score,
                             self.threshold, self._win_rms)
                else:
                    log.debug(msg, self._win_score, self.threshold,
                              self._win_rms)
                self._win_chunks = 0
                self._win_score = 0.0
                self._win_rms = 0.0
            if score >= self.threshold:
                fired = True
                break
        return fired

    def reset(self) -> None:
        # Drop accumulated context so the next activation starts clean.
        self._model.reset()
        self._buf = np.empty(0, dtype=np.int16)
