"""The living floor — sub-visible drift that separates a body from a prop.

A held hand that is PERFECTLY still reads as switched off. This generator
drifts each finger's closure within a small band and the wrists within a
few degrees, on slow incommensurate sine stacks (a cheap 1/f-ish spectrum:
no two joints share a period, nothing ever visibly repeats). Amplitude is
externally scalable so the scheduler can duck it against the room's ambient
noise floor — PCA servos whine, and the mic is close.

Pure function of time: frame(t) is deterministic per process (phases drawn
once at construction), thread-free, and costs microseconds.
"""
from __future__ import annotations

import math
import random

from zero.arms import hands


class MicroMotion:
    def __init__(self, *, closure_amp: float = 0.03, wrist_amp_deg: float = 1.5,
                 seed: int | None = None):
        self.closure_amp = float(closure_amp)
        self.wrist_amp = float(wrist_amp_deg)
        rng = random.Random(seed)
        # Three incommensurate periods per joint, breathing-adjacent rates.
        self._osc: dict[str, list[tuple[float, float, float]]] = {}
        for side in hands.SIDES:
            for f in hands.FINGERS:
                self._osc[f"{side}.{f}"] = [
                    (rng.uniform(0.07, 0.13), rng.uniform(0, math.tau), 0.5),
                    (rng.uniform(0.19, 0.31), rng.uniform(0, math.tau), 0.35),
                    (rng.uniform(0.43, 0.71), rng.uniform(0, math.tau), 0.15),
                ]
            self._osc[f"{side}.wrist"] = [
                (rng.uniform(0.05, 0.09), rng.uniform(0, math.tau), 0.6),
                (rng.uniform(0.17, 0.27), rng.uniform(0, math.tau), 0.4),
            ]

    def _v(self, key: str, t: float) -> float:
        return sum(w * math.sin(math.tau * hz * t + ph)
                   for hz, ph, w in self._osc[key])

    def closure_offsets(self, t: float, scale: float = 1.0) -> dict:
        """{(side, finger): closure delta} around whatever base pose holds."""
        a = self.closure_amp * max(0.0, min(1.0, scale))
        return {(s, f): a * self._v(f"{s}.{f}", t)
                for s in hands.SIDES for f in hands.FINGERS}

    def wrist_offsets(self, t: float, scale: float = 1.0) -> dict:
        """{side: degrees delta} around the wrist's base orientation."""
        a = self.wrist_amp * max(0.0, min(1.0, scale))
        return {s: a * self._v(f"{s}.wrist", t) for s in hands.SIDES}
