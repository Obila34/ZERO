#!/usr/bin/env python3
"""Print objective smoothness metrics for ZERO's head controller.

Runs the real HeadController (with config defaults) through representative moves
against the NullDriver — no hardware — and reports jerk, reversals, overshoot,
settling time and spectral concentration. Use it to judge phase-1 feel now, and
as the baseline to compare a real actuator against later (feed a fiducial-tracked
or optical-flow angle series into zero.head.smoothness.summarize the same way).

    .venv/bin/python scripts/head_smoothness.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.head.controller import HeadController
from zero.head.smoothness import simulate, summarize

try:
    from zero.config import load_config
    _cfg = load_config()

    def cget(k, d):
        return _cfg.get(k, d)
except Exception:
    def cget(k, d):
        return d


def _controller():
    return HeadController(
        lambda x, y: None,
        rate_hz=float(cget("head.rate_hz", 25.0)),
        max_speed_dps=float(cget("head.max_speed_dps", 36.0)),
        limit_deg=float(cget("head.limit_deg", 45.0)),
    )


def _report(name, schedule, duration, axis=0):
    c = _controller()
    t, x = simulate(c, schedule, dt=1.0 / float(cget("head.rate_hz", 25.0)),
                    duration=duration, axis=axis)
    r = summarize(t, x)
    print(f"\n== {name} ==")
    for k, v in r.as_dict().items():
        if isinstance(v, float):
            print(f"  {k:24s} {v:8.3f}")
        else:
            print(f"  {k:24s} {v:8d}")
    verdict = ("SMOOTH" if r.reversals <= 1 and (r.spectral_concentration or 0) > 0.85
               and (r.overshoot or 0) <= 0.05 else "CHECK")
    print(f"  --> {verdict}")


def main():
    print("HeadController smoothness (NullDriver, config defaults):")
    print(f"  max_speed_dps={cget('head.max_speed_dps', 36.0)} "
          f"rate_hz={cget('head.rate_hz', 25.0)}")
    _report("saccade: 0 -> 40 deg pan", [(0.1, 40.0, 0.0)], duration=3.0, axis=0)
    _report("small look: 0 -> 8 deg pan", [(0.1, 8.0, 0.0)], duration=1.5, axis=0)
    _report("return: 40 -> 0 after settle",
            [(0.0, 40.0, 0.0), (1.5, 0.0, 0.0)], duration=3.0, axis=0)
    _report("tilt nod: 0 -> -12 deg", [(0.1, 0.0, -12.0)], duration=1.5, axis=1)


if __name__ == "__main__":
    main()
