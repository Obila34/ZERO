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
from collections import deque
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
    def __init__(self, frame_buffer: int = 8) -> None:
        self._lock = threading.Lock()
        self._detections: list[Detection] = []
        self._frame = None
        self._timestamp = 0.0
        self._frame_index = 0
        # A short rolling buffer of recent (frame, timestamp) pairs so a visual
        # turn can sample 2-3 frames across the last ~second and SEE motion (a
        # raised hand, someone mid-turn) instead of a single frozen pose.
        self._frames: deque = deque(maxlen=max(1, int(frame_buffer)))

    def update(self, detections: list[Detection], frame_rgb=None) -> None:
        with self._lock:
            # The perception loop builds a FRESH detections list every frame and
            # never mutates it after publishing, so storing by reference is safe —
            # and deep-copying pydantic models 30x/s was real CPU on the Pi.
            self._detections = list(detections)
            self._timestamp = time.time()
            if frame_rgb is not None:
                self._frame = frame_rgb  # already a private copy from the loop
                self._frames.append((frame_rgb, self._timestamp))
            self._frame_index += 1

    def recent_frames(self, n: int = 2, within_s: float = 1.0) -> list:
        """Up to ``n`` recent frames sampled evenly across the last ``within_s``
        seconds (oldest -> newest). Returns copies; ``[]`` if nothing seen yet.
        """
        n = max(1, int(n))
        with self._lock:
            buf = list(self._frames)
        if not buf:
            return []
        now = time.time()
        windowed = [(f, t) for (f, t) in buf if now - t <= within_s] or buf[-1:]
        if len(windowed) <= n:
            chosen = windowed
        elif n == 1:
            chosen = windowed[-1:]
        else:
            # Evenly spaced indices across the window (dedup keeps it robust for
            # tiny buffers), so the frames span the motion rather than clustering.
            last = len(windowed) - 1
            idxs = sorted({round(i * last / (n - 1)) for i in range(n)})
            chosen = [windowed[i] for i in idxs]
        return [f.copy() for (f, _t) in chosen]

    def snapshot(self) -> Snapshot:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            # Shallow list copy: readers treat detections as read-only, and the
            # writer replaces (never mutates) the published list.
            return Snapshot(
                detections=list(self._detections),
                frame_rgb=frame,
                timestamp=self._timestamp,
                frame_index=self._frame_index,
            )
