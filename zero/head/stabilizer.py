"""Stabilizer — estimate the camera's angular rate so vision can cancel ZERO's
own motion (efference copy / VOR).

The camera rides on the head, so every neck move sweeps the whole image. This
tells the vision layer how fast that self-motion is, for three uses: suppressing
false novelty (done in Eyes), dropping motion-blurred frames mid-saccade, and
(future) subtracting the predicted pixel flow from the tracker.

Two sources, and it is IMU-READY by design:
  * EFFERENCE COPY (default, no sensor) — the commanded neck angular velocity.
    For a STATIC base this is exact: the only thing moving the camera is us, and
    we know exactly how we commanded it. Needs no hardware.
  * IMU (optional, drop-in) — attach_imu(reader) where reader() returns measured
    angular rate (deg/s). A real gyro also catches base shake / bumps that the
    efference copy cannot know about, so when present it is preferred. Wire a
    BNO055/ICM over the head Pi's I2C and pass a reader; nothing else changes.

Pure logic, no I/O — the IMU reader is injected. Unit-testable headless.
"""
from __future__ import annotations

import threading


class Stabilizer:
    def __init__(self, *, moving_eps_dps: float = 3.0,
                 saccade_dps: float = 15.0, imu=None):
        self._eps = float(moving_eps_dps)
        self._saccade = float(saccade_dps)
        self._imu = imu                     # callable() -> (pan_dps, tilt_dps) | None
        self._cmd = (0.0, 0.0)
        self._lock = threading.Lock()

    def note_commanded(self, pan_dps: float, tilt_dps: float) -> None:
        """Feed the current COMMANDED neck angular velocity (deg/s)."""
        with self._lock:
            self._cmd = (float(pan_dps), float(tilt_dps))

    def attach_imu(self, reader) -> None:
        """Provide a gyro reader() -> (pan_dps, tilt_dps) in deg/s (or None).
        Once attached it is preferred over the efference-copy estimate."""
        self._imu = reader

    def angular_rate(self) -> tuple[float, float]:
        """Best current estimate of the camera's angular rate (deg/s)."""
        if self._imu is not None:
            try:
                r = self._imu()
                if r is not None:
                    return (float(r[0]), float(r[1]))
            except Exception:
                pass
        with self._lock:
            return self._cmd

    def is_moving(self) -> bool:
        p, t = self.angular_rate()
        return abs(p) > self._eps or abs(t) > self._eps

    def is_saccading(self) -> bool:
        """Fast enough that a frame is motion-blurred / mid-flight — drop it."""
        p, t = self.angular_rate()
        return abs(p) > self._saccade or abs(t) > self._saccade

    @property
    def has_imu(self) -> bool:
        return self._imu is not None
