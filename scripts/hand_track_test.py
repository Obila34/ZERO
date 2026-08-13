#!/usr/bin/env python3
"""Control the head with your HAND (or head) — direct teleoperation via YOLO-pose.

Move your hand left/right and the head follows in real time: pose gives the wrist
position, a 1€ filter smooths it, and it maps STRAIGHT to a pan angle (open loop,
no visual servo — snappy, can't wind up). Detection ~8 fps; the head runs at 25 Hz
and interpolates, so motion is smooth.

    .venv/bin/python scripts/hand_track_test.py           # REAL motion, hand control
    .venv/bin/python scripts/hand_track_test.py --sim     # no motion, watch numbers
    .venv/bin/python scripts/hand_track_test.py --head    # control with your head (nose)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.config import load_config
from zero.factory import build_vision
from zero.head.controller import HeadController
from zero.head.driver import NullDriver, make_driver
from zero.head.hand import HandPoseSource


def main():
    p = argparse.ArgumentParser(description="control the head with your hand/head")
    p.add_argument("--sim", action="store_true", help="no motion; watch the numbers")
    p.add_argument("--head", action="store_true",
                   help="control with your head (nose) instead of your hand")
    p.add_argument("--secs", type=float, default=90.0)
    a = p.parse_args()

    cfg = load_config()
    eyes = build_vision(cfg)
    if eyes is None:
        print("vision disabled"); return
    eyes.start()
    model = cfg.resolve_path("head.hand.model_path", "models/yolo11n-pose.onnx")
    src = HandPoseSource(
        model, keypoint=("head" if a.head else "wrist"),
        min_cutoff=float(cfg.get("head.hand.min_cutoff", 1.0)),
        beta=float(cfg.get("head.hand.beta", 0.3)),
        conf=float(cfg.get("head.hand.conf", 0.25)),
        kp_conf=float(cfg.get("head.hand.kp_conf", 0.15)),
        mirror=bool(cfg.get("head.hand.mirror", True)),
        gain=float(cfg.get("head.hand.gain", 2.2)))
    limit = float(cfg.get("head.limit_deg", 80.0))
    drv = NullDriver() if a.sim else make_driver(cfg)
    ctl = HeadController(drv.send, rate_hz=float(cfg.get("head.rate_hz", 33.0)),
                         max_speed_dps=float(cfg.get("head.max_speed_dps", 70.0)),
                         limit_deg=limit)
    ctl.start()
    what = "HEAD (nose)" if a.head else "HAND (wrist)"
    print(f"\n{what} control — {'SIM (no motion)' if a.sim else 'REAL motion'}. "
          f"Move {'your head' if a.head else 'your hand'} left/right; the head follows.")
    print(f"{'x_norm':>8} {'conf':>6} {'pan target':>11} {'head pan':>9}")
    t0 = time.time()
    try:
        while time.time() - t0 < a.secs:
            frame = eyes.current_frame()
            x, conf = src.update(frame)
            target = x * limit
            ctl.set_target(target, 0.0)
            px, _ = ctl.position
            print(f"{x:+8.2f} {conf:6.2f} {target:+11.1f} {px:+9.1f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctl.center()
        time.sleep(0.2)
        ctl.stop()
        eyes.stop()
    print("done.")


if __name__ == "__main__":
    main()
