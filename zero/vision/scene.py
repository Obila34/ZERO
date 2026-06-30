"""Thread-safe holder for "what the camera sees right now".

The always-on capture->detect->color loop writes a fresh ``Snapshot`` here; the
conversation thread reads it (instantly, no blocking) when a turn needs visual
context. The snapshot carries both the named detections and the keyframe they
were measured on, so the same image can be sent to the GPU for depth and/or to a
multimodal LLM.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from zero.vision.schemas import Detection


@dataclass(frozen=True)
class Snapshot:
    detections: list[Detection] = field(default_factory=list)
    frame_rgb: Optional["object"] = None  # np.ndarray, or None if no frame yet
    timestamp: float = 0.0
    frame_index: int = 0

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.timestamp) if self.timestamp else float("inf")


class SceneState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._detections: list[Detection] = []
        self._frame = None
        self._timestamp = 0.0
        self._frame_index = 0

    def update(self, detections: list[Detection], frame_rgb=None) -> None:
        with self._lock:
            self._detections = [d.model_copy(deep=True) for d in detections]
            if frame_rgb is not None:
                self._frame = frame_rgb  # already a private copy from the loop
            self._timestamp = time.time()
            self._frame_index += 1

    def snapshot(self) -> Snapshot:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return Snapshot(
                detections=[d.model_copy(deep=True) for d in self._detections],
                frame_rgb=frame,
                timestamp=self._timestamp,
                frame_index=self._frame_index,
            )
