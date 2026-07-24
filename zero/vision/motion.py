"""Tier 0 — frame-differencing motion detection. Runs on EVERY frame.

The cheapest possible "did anything happen?" signal: downscale to grayscale,
absdiff against the previous frame, count changed pixels. No models, no
allocation churn — a few milliseconds on a Pi. Its two jobs:

* publish a live motion level into the world state (Tier 0 write), and
* GATE Tier 1: when the scene is static, the (much costlier) detector drops
  to a keepalive cadence instead of burning CPU re-detecting a still room.
  This is the sparse-firing budget made explicit: most of the brain silent
  most of the time.
"""
from __future__ import annotations

from typing import Optional


class MotionDetector:
    """Stateful per-frame motion score in [0, 1] (fraction of pixels changed).

    ``update(frame_rgb)`` returns ``(level, active)``. ``active`` applies
    hysteresis: motion switches on above ``threshold`` and only switches off
    after ``quiet_frames`` consecutive below-threshold frames, so brief pauses
    mid-movement don't flicker the gate.
    """

    def __init__(self, downscale_w: int = 160, pixel_delta: int = 25,
                 threshold: float = 0.02, quiet_frames: int = 15):
        self.downscale_w = int(downscale_w)
        self.pixel_delta = int(pixel_delta)
        self.threshold = float(threshold)
        self.quiet_frames = max(1, int(quiet_frames))
        self._prev = None
        self._quiet_run = 0
        self._active = False
        self.level = 0.0

    def update(self, frame_rgb) -> tuple[float, bool]:
        import cv2
        import numpy as np

        h, w = frame_rgb.shape[:2]
        scale_h = max(1, int(round(h * self.downscale_w / max(1, w))))
        small = cv2.cvtColor(
            cv2.resize(frame_rgb, (self.downscale_w, scale_h),
                       interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2GRAY)
        prev, self._prev = self._prev, small
        if prev is None or prev.shape != small.shape:
            self.level = 0.0
            return 0.0, self._active
        diff = cv2.absdiff(small, prev)
        self.level = float(np.count_nonzero(diff > self.pixel_delta)) / diff.size
        if self.level >= self.threshold:
            self._quiet_run = 0
            self._active = True
        else:
            self._quiet_run += 1
            if self._quiet_run >= self.quiet_frames:
                self._active = False
        return self.level, self._active

    def reset(self) -> None:
        self._prev = None
        self._quiet_run = 0
        self._active = False
        self.level = 0.0


class DetectionGate:
    """Decides when Tier 1 (detection) may run, from Tier 0's motion signal.

    Policy: full cadence (``active_interval_s``, 0 = every frame) while motion
    is active or for ``linger_s`` after it stops; otherwise a keepalive
    detection every ``idle_interval_s`` so slow scene changes (someone placing
    an object during a still moment) are still noticed within a bounded delay.
    """

    def __init__(self, active_interval_s: float = 0.0,
                 idle_interval_s: float = 2.0, linger_s: float = 3.0):
        self.active_interval_s = float(active_interval_s)
        self.idle_interval_s = float(idle_interval_s)
        self.linger_s = float(linger_s)
        self._last_detect = 0.0
        self._last_active = 0.0

    def should_detect(self, motion_active: bool, now: float) -> bool:
        if motion_active:
            self._last_active = now
        recently_active = motion_active or (now - self._last_active
                                            <= self.linger_s)
        interval = (self.active_interval_s if recently_active
                    else self.idle_interval_s)
        if now - self._last_detect >= interval:
            self._last_detect = now
            return True
        return False
