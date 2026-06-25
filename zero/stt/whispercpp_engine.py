"""whisper.cpp STT — ARM-NEON optimized, the best speed/accuracy on Pi 5.

Uses the pywhispercpp binding. Model file (ggml-*.bin) is downloaded by
scripts/setup_pi.sh into models/whisper/.
"""
from __future__ import annotations

import re

import numpy as np

from zero.utils.logging import get_logger
from zero.stt.base import STT

log = get_logger("stt.whispercpp")

# Whisper invents non-speech cues like "(applause)", "[BLANK_AUDIO]", "(music)"
# when fed silence or echo. Strip bracketed segments; if nothing real remains,
# treat the whole thing as empty so it never reaches the LLM.
_BRACKETS = re.compile(r"[\(\[][^\)\]]*[\)\]]")


def _strip_hallucinations(text: str) -> str:
    cleaned = _BRACKETS.sub("", text).strip(" .,!?-…")
    # Need at least one real word (2+ letters) to count as genuine speech.
    return cleaned if re.search(r"[A-Za-z]{2,}", cleaned) else ""


class WhisperCppSTT(STT):
    def __init__(self, model_path: str, language: str = "en",
                 initial_prompt: str | None = None, threads: int = 4,
                 audio_ctx: int = 0):
        from pywhispercpp.model import Model  # lazy

        self.language = language
        self.initial_prompt = initial_prompt
        # audio_ctx caps how much of the 30s encoder window is processed. Lower =
        # faster encode (the Pi's main STT cost) at some accuracy risk on long
        # speech. 0 = full window (default). ~750 suits short conversational turns.
        self.audio_ctx = audio_ctx
        self._supports_audio_ctx = True  # flipped off if the binding rejects it
        # During transcription nothing else needs the CPU, so use all 4 Pi cores.
        self._model = Model(model_path, n_threads=threads, print_realtime=False,
                            print_progress=False)
        log.info("whisper.cpp loaded (%s, threads=%d, audio_ctx=%d)",
                 model_path, threads, audio_ctx)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # whisper expects float32 mono at 16 kHz.
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        kwargs = {"language": self.language}
        if self.initial_prompt:
            # Vocabulary/style bias — trims single-phoneme mishears.
            kwargs["initial_prompt"] = self.initial_prompt
        if self.audio_ctx and self._supports_audio_ctx:
            kwargs["audio_ctx"] = self.audio_ctx
        try:
            segments = self._model.transcribe(audio, **kwargs)
        except TypeError:
            # Older/newer pywhispercpp may not accept audio_ctx — drop it once.
            self._supports_audio_ctx = False
            kwargs.pop("audio_ctx", None)
            log.warning("pywhispercpp rejected audio_ctx; disabling it")
            segments = self._model.transcribe(audio, **kwargs)
        text = " ".join(seg.text for seg in segments).strip()
        cleaned = _strip_hallucinations(text)
        if cleaned != text:
            log.info("heard: %r -> cleaned: %r", text, cleaned)
        else:
            log.info("heard: %r", text)
        return cleaned
