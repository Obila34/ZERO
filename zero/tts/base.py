"""Text-to-speech interface: text (with emotion cues) in, audio out.

Both engines accept the SAME cue-tagged text from the LLM. The orchestrator
normalizes the shared cue vocabulary into whatever syntax the chosen engine
wants, so the LLM prompt stays engine-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class TTS(ABC):
    #: Native output sample rate of the engine (Hz).
    sample_rate: int = 22050

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Render cue-tagged text to a float32 mono waveform at self.sample_rate."""
        raise NotImplementedError

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio in chunks as it's produced. Default: one chunk (the whole
        clip). Engines that can stream (e.g. remote Orpheus) override this."""
        audio = self.synthesize(text)
        if getattr(audio, "size", 0):
            yield audio

    def close(self) -> None:
        """Release resources. Optional."""
