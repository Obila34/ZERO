#!/usr/bin/env python3
"""Verify ZERO's head FOLLOWS your face the right way (the one-look sign test).

Runs the real camera + face detector + head, and prints which side your face is
on vs which way the head turned. If they match, tracking is correct; if the head
turns AWAY from your face, a sign is flipped and we fix it in config.

    .venv/bin/python scripts/head_track_test.py --sim     # SAFE: no motion, just numbers
    .venv/bin/python scripts/head_track_test.py           # REAL motion (via reflex service)

Stand in front of the camera. Lean LEFT, then RIGHT, then look from high/low.
Watch the printed lines: 'face RIGHT ... head pan +' means it followed you right.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.config import load_config
from zero.factory import build_head, build_vision


class _Override:
    """Wrap Config, overriding a few keys (to force null driver in --sim)."""
    def __init__(self, cfg, over):
        self._c = cfg
        self._o = over

    def get(self, k, d=None):
        return self._o.get(k, self._c.get(k, d))

    def __getattr__(self, name):
        return getattr(self._c, name)


def _side(v, tol=0.08):
    return "RIGHT" if v > tol else "LEFT" if v < -tol else "center"


def main():
    p = argparse.ArgumentParser(description="face-tracking sign test")
    p.add_argument("--sim", action="store_true", help="no motion; watch the numbers")
    p.add_argument("--secs", type=float, default=40.0)
    a = p.parse_args()

    cfg = load_config()
    if a.sim:
        cfg = _Override(cfg, {"head.driver": "null"})

    eyes = build_vision(cfg)
    if eyes is None:
        print("vision is disabled/unavailable — can't run the tracking test.")
        return
    eyes.start()
    head = build_head(cfg, eyes=eyes)
    if head is None:
        print("head subsystem is disabled (head.enabled).")
        eyes.stop()
        return
    head.start()

    mode = "SIM (no motion)" if a.sim else "REAL motion via the reflex service"
    print(f"\ntracking test — {mode}. Stand in front of the camera.")
    print("Lean LEFT, then RIGHT, then tilt your head's position up/down.")
    print("Correct = the head turns TOWARD the side your face is on.\n")
    print(f"{'face side':>10} {'face ex':>8} {'face ey':>8} | {'head pan':>8} {'head tilt':>9}")
    t0 = time.time()
    try:
        while time.time() - t0 < a.secs:
            att = eyes.attention()
            pan, tilt = head.position
            fr = eyes.current_frame()
            d = head.status().get("dbg", {})
            dbg = (f"ticks={d.get('ticks',0)} face={d.get('face',0)} "
                   f"nowin={d.get('nowin',0)} branch={d.get('branch','-')} "
                   f"off={d.get('off',0)} aim={d.get('aim')} err={d.get('err','')}")
            if att and fr is not None:
                x, y, w, h = att
                hh, ww = fr.shape[:2]
                ex = (x + w / 2) / ww - 0.5
                ey = (y + h / 2) / hh - 0.5
                print(f"{_side(ex):>8} ex{ex:+.2f} ey{ey:+.2f} | pan{pan:+.1f} tilt{tilt:+.1f} | {dbg}")
            else:
                print(f"{'(noface)':>8} {'':>13} | pan{pan:+.1f} tilt{tilt:+.1f} | {dbg}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        head.stop()
        eyes.stop()
    print("\ndone. Tell me: did the head turn toward your face, or away from it?")


if __name__ == "__main__":
    main()
