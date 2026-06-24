"""whisper.cpp STT — ARM-NEON optimized, the best speed/accuracy on Pi 5.

Uses the pywhispercpp binding. Model file (ggml-*.bin) is downloaded by
scripts/setup_pi.sh into models/whisper/.
"""
from __future__ import annotations

import numpy as np

from zero.utils.logging import get_logger
from zero.stt.base import STT

log = get_logger("stt.whispercpp")


class WhisperCppSTT(STT):
    def __init__(self, model_path: str, language: str = "en",
                 initial_prompt: str | None = None):
        from pywhispercpp.model import Model  # lazy

        self.language = language
        self.initial_prompt = initial_prompt
        # n_threads tuned for Pi 5's 4 cores; leave one for the rest of the pipeline.
        self._model = Model(model_path, n_threads=3, print_realtime=False,
                            print_progress=False)
        log.info("whisper.cpp loaded (%s)", model_path)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # whisper expects float32 mono at 16 kHz.
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        kwargs = {"language": self.language}
        if self.initial_prompt:
            # Vocabulary/style bias — trims single-phoneme mishears.
            kwargs["initial_prompt"] = self.initial_prompt
        segments = self._model.transcribe(audio, **kwargs)
        text = " ".join(seg.text for seg in segments).strip()
        log.info("heard: %r", text)
        return text
