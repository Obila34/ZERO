"""Per-tier compute budgets — sparse firing as an ENFORCED ceiling.

Phase 2 made the tiers event-driven (motion gates detection, the narrator
self-skips); this makes the ceilings hard so a pathological scene (constant
motion, flickering lights) cannot pin the CPU/GPU:

* ``DutyBudget``  — Tier 1: the detector may consume at most ``max_duty`` of
  wall time over a sliding window. Above it, detection frames are skipped
  regardless of motion.
* ``RateBudget``  — Tier 2: at most ``max_per_min`` narrator inferences per
  minute, pokes included (a surprise storm degrades to routine cadence
  instead of melting the GPU).

Both are tiny, lock-free-enough (single consumer thread each), and observable
(``rejections`` counter) so the benchmark can prove enforcement.
"""
from __future__ import annotations

import time
from collections import deque


class DutyBudget:
    def __init__(self, max_duty: float = 0.6, window_s: float = 10.0):
        self.max_duty = float(max_duty)
        self.window_s = float(window_s)
        self._spent: deque = deque()      # (ts, duration_s)
        self.rejections = 0

    def _duty(self, now: float) -> float:
        while self._spent and now - self._spent[0][0] > self.window_s:
            self._spent.popleft()
        return sum(d for _, d in self._spent) / self.window_s

    def allowed(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if self._duty(now) < self.max_duty:
            return True
        self.rejections += 1
        return False

    def record(self, duration_s: float, now: float | None = None) -> None:
        self._spent.append((time.time() if now is None else now,
                            float(duration_s)))


class RateBudget:
    def __init__(self, max_per_min: int = 20):
        self.max_per_min = int(max_per_min)
        self._times: deque = deque()
        self.rejections = 0

    def allowed(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        while self._times and now - self._times[0] > 60.0:
            self._times.popleft()
        if len(self._times) < self.max_per_min:
            self._times.append(now)
            return True
        self.rejections += 1
        return False
