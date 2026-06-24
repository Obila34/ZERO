"""whisper.cpp STT — ARM-NEON optimized, the best speed/accuracy on Pi 5.

Uses the pywhispercpp binding. Model file (ggml-*.bin) is downloaded by
scripts/setup_pi.sh into models/whisper/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from zero.utils.logging import get_logger
from zero.stt.base import STT

log = get_logger("stt.whispercpp")


class WhisperCppSTT(STT):
    def __init__(self, model_path: str, language: str = "en", n_threads: int = 4):
        from pywhispercpp.model import Model  # lazy

        self.language = language
        # pywhispercpp wants a model NAME (+ optional models_dir), not a full path.
        # If we were given an existing ggml-*.bin file, derive name + dir from it so
        # it loads our pre-downloaded file instead of trying to re-download.
        p = Path(model_path)
        common = dict(n_threads=n_threads, print_realtime=False, print_progress=False)
        looks_like_path = p.suffix == ".bin" or "/" in str(model_path) or "\\" in str(model_path)
        if p.is_file():
            name = p.stem.replace("ggml-", "")  # ggml-base.en.bin -> base.en
            self._model = Model(name, models_dir=str(p.parent), **common)
            log.info("whisper.cpp loaded (%s from %s)", name, p.parent)
        elif looks_like_path:
            # A path was given but the file isn't there. Fail loudly — otherwise
            # pywhispercpp loads a null model and transcribe() SEGFAULTS.
            raise FileNotFoundError(
                f"Whisper model not found: {p}\n"
                f"Download it (see scripts/setup_pi.sh) or fix stt.model_path in "
                f"config.yaml. Expected a ggml-*.bin file."
            )
        else:
            # Treat as a known model name; pywhispercpp will fetch it if missing.
            self._model = Model(model_path, **common)
            log.info("whisper.cpp loaded (%s)", model_path)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # whisper expects float32 mono at 16 kHz.
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        segments = self._model.transcribe(audio, language=self.language)
        text = " ".join(seg.text for seg in segments).strip()
        log.info("heard: %r", text)
        return text
