#!/usr/bin/env python3
"""Perception probe: is the pose seeing a REAL hand, or hallucinating?

Reads the RAW camera (not the detection snapshot) and runs the same HandPoseSource
the head uses. Prints per frame: brightness (dead/black?), frame change (stale?),
person conf, both shoulders + the higher wrist, shoulder WIDTH (junk detections are
~6 px), and the resulting signal (None = rejected). No motion is commanded.

Move your hand left/right: the wrist x should track it and 'signal' swing +/-.
    .venv/bin/python scripts/hand_diag.py [seconds]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from zero.config import load_config
from zero.factory import build_vision
from zero.head.hand import HandPoseSource


def main():
    cfg = load_config()
    eyes = build_vision(cfg)
    if eyes is None:
        print("vision disabled"); return
    eyes.start()
    src = HandPoseSource(
        cfg.resolve_path("head.hand.model_path", "models/yolo11n-pose.onnx"),
        keypoint=str(cfg.get("head.hand.keypoint", "wrist")),
        conf=float(cfg.get("head.hand.conf", 0.5)),
        kp_conf=float(cfg.get("head.hand.kp_conf", 0.3)),
        min_shoulder_frac=float(cfg.get("head.hand.min_shoulder_frac", 0.10)),
        gain=float(cfg.get("head.hand.gain", 1.3)),
        mirror=bool(cfg.get("head.hand.mirror", True)))

    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    prev = None
    print("Move your hand left/right. wristHi.x should follow; signal should swing +/-.")
    print(f"{'bright':>6} {'chg':>4} {'person':>6} {'width':>6} "
          f"{'L_sh(x,y,c)':>16} {'wristHi(x,y,c)':>18} {'signal':>7}  pan?")
    limit = float(cfg.get("head.limit_deg", 80.0))
    gain = float(cfg.get("head.hand.gain", 1.3))
    mirror = bool(cfg.get("head.hand.mirror", True))
    t0 = time.time()
    try:
        while time.time() - t0 < secs:
            f = eyes.raw_frame()
            if f is None:
                print("frame=None (camera warming up)"); time.sleep(0.15); continue
            bright = float(f.mean())
            chg = 0.0 if prev is None else float(
                np.abs(f.astype(np.int16) - prev.astype(np.int16)).mean())
            prev = f.copy()
            d = src.debug(f)
            sig = d["signal"]
            if sig is None:
                pan = "-"
            else:
                hx = max(-1.0, min(1.0, sig * gain))
                hx = -hx if mirror else hx
                pan = f"{hx*limit:+6.1f}"
            lsh = "(%5.1f,%5.1f,%.2f)" % d["Lsh"]
            wr = "(%5.1f,%5.1f,%.2f)" % d["wrist"]
            sigs = "  none" if sig is None else f"{sig:+7.2f}"
            print(f"{bright:6.1f} {chg:4.1f} {d['person']:6.2f} {d['width']:6.1f} "
                  f"{lsh:>16} {wr:>18} {sigs:>7}  {pan}")
            time.sleep(0.12)
    except KeyboardInterrupt:
        pass
    finally:
        eyes.stop()
    print("done. (min shoulder width to accept: %.0f px)" % src._min_shoulder_px)


if __name__ == "__main__":
    main()
