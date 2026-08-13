#!/usr/bin/env python3
"""Run ZERO's head in HAND-CONTROL mode through the real HeadSystem.

Unlike scripts/hand_track_test.py (which wires the pieces by hand), this drives
the fully-integrated subsystem: build_head() -> HeadSystem with head.input=hand,
the real driver from config, efference-copy suppression, the lot. This is the
"finished" path — the same code a normal ZERO run uses when head.input: hand.

    .venv/bin/python scripts/head_hand.py            # hand control, real motion
    .venv/bin/python scripts/head_hand.py --head     # control with your head (nose)
    .venv/bin/python scripts/head_hand.py --secs 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.config import load_config
from zero.factory import build_head, build_vision


def main():
    ap = argparse.ArgumentParser(description="ZERO head — hand teleoperation")
    ap.add_argument("--head", action="store_true",
                    help="control with your head (nose) instead of your hand")
    ap.add_argument("--secs", type=float, default=90.0)
    a = ap.parse_args()

    cfg = load_config()
    # Force the integrated head into hand mode for this run (leaves config.yaml's
    # default untouched). raw is the merged dict behind Config.
    cfg.raw.setdefault("head", {})
    cfg.raw["head"]["enabled"] = True
    cfg.raw["head"]["input"] = "hand"
    cfg.raw["head"].setdefault("hand", {})["keypoint"] = "head" if a.head else "wrist"

    eyes = build_vision(cfg)
    if eyes is None:
        print("vision disabled — cannot hand-control"); return
    head = build_head(cfg, eyes=eyes)
    if head is None:
        print("head disabled/failed to build"); return

    eyes.start()
    head.start()
    what = "HEAD (nose)" if a.head else "HAND (wrist)"
    st = head.status()
    print(f"\n{what} control via HeadSystem — driver={st['driver']} "
          f"moves_hardware={st['moves_hardware']}")
    print("Keep your shoulders + one raised hand in frame; sweep left/right.\n")
    print(f"{'pan':>7} {'branch':>10}")
    t0 = time.time()
    try:
        while time.time() - t0 < a.secs:
            st = head.status()
            print(f"{st['pan']:+7.1f} {st['dbg'].get('branch','-'):>10}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        head.stop()
        eyes.stop()
    print("done.")


if __name__ == "__main__":
    main()
