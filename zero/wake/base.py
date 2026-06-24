"""Wake-word interface. An engine consumes audio frames and reports activation."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WakeWord(ABC):
    """Always-on detector. Feed it 16 kHz mono frames; it returns True on trigger."""

    @abstractmethod
    def process(self, frame: np.ndarray) -> bool:
        """Push one audio frame (int16 mono). Return True when the wake word fires."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal buffers (call after each activation). Optional."""

    def close(self) -> None:
        """Release resources. Optional."""
