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
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5,
                 inference_framework: str = "onnx"):
        from openwakeword.model import Model  # lazy

        self.threshold = threshold
        self.model_name = model
        # ONNX by default: tflite-runtime has no wheel for Python 3.12 on ARM, so
        # we install openwakeword with --no-deps and run on onnxruntime instead.
        kwargs = {"inference_framework": inference_framework}
        # Pass a known keyword name or a path to a custom .onnx model.
        if model:
            kwargs["wakeword_models"] = [model]
        self._model = Model(**kwargs)
        log.info("openWakeWord loaded (model=%s, framework=%s, threshold=%.2f)",
                 model, inference_framework, threshold)

    def process(self, frame: np.ndarray) -> bool:
        # openWakeWord expects int16 mono samples.
        scores = self._model.predict(frame)
        score = max(scores.values()) if scores else 0.0
        return score >= self.threshold

    def reset(self) -> None:
        # Drop accumulated context so the next activation starts clean.
        self._model.reset()
