"""Speech-to-text interface: audio in, text out."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class STT(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a complete utterance (int16 or float32 mono) to text."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources. Optional."""
