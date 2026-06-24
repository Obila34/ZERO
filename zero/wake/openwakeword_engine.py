"""openWakeWord engine — always-on, low CPU, supports custom wake words.

Heavy deps are imported lazily inside __init__ so the package imports fine on a
dev machine without the model installed.
"""
from __future__ import annotations

import numpy as np

from zero.utils.logging import get_logger
from zero.wake.base import WakeWord

log = get_logger("wake.oww")


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
        log.info("openWakeWord loaded (model=%s, threshold=%.2f)", model, threshold)

    def process(self, frame: np.ndarray) -> bool:
        # openWakeWord expects int16 mono samples.
        scores = self._model.predict(frame)
        score = max(scores.values()) if scores else 0.0
        return score >= self.threshold

    def reset(self) -> None:
        # Drop accumulated context so the next activation starts clean.
        self._model.reset()
