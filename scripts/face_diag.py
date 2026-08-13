#!/usr/bin/env python3
"""Raw YuNet probe with a PERMISSIVE threshold — does the detector see a face at
all, and at what size/score? Reuses the running Eyes camera. No motion.

    .venv/bin/python scripts/face_diag.py
Stand in front, at your normal distance, and hold still a moment.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from zero.config import load_config
from zero.factory import build_vision

cfg = load_config()
model = cfg.resolve_path("identity.face.detector_path",
                         "models/face/yunet_2023mar.onnx")
# very permissive: score 0.3 (prod uses 0.8), no min-size filter here
det = cv2.FaceDetectorYN.create(str(model), "", (320, 320),
                                score_threshold=0.3, nms_threshold=0.3, top_k=50)

eyes = build_vision(cfg)
if eyes is None:
    print("vision disabled"); sys.exit(0)
eyes.start()
print("probing (score>=0.3, no size filter). Current prod: score>=0.8, min 60px.\n")
print(f"{'raw#':>5} {'top_size':>10} {'score':>6} {'cx_off':>7} {'cy_off':>7} {'bright':>7}")
try:
    for i in range(60):
        fr = eyes.current_frame()
        if fr is None:
            print("(no frame yet)"); time.sleep(0.4); continue
        h, w = fr.shape[:2]
        bgr = cv2.cvtColor(np.asarray(fr), cv2.COLOR_RGB2BGR)
        det.setInputSize((w, h))
        _, faces = det.detect(bgr)
        bright = float(np.asarray(fr).mean())
        if faces is not None and len(faces):
            f = faces[0]
            x, y, fw, fh, score = f[0], f[1], f[2], f[3], f[-1]
            cx = (x + fw / 2) / w - 0.5
            cy = (y + fh / 2) / h - 0.5
            print(f"{len(faces):>5} {int(fw)}x{int(fh):<6} {score:>6.2f} "
                  f"{cx:>+7.2f} {cy:>+7.2f} {bright:>7.0f}")
        else:
            print(f"{0:>5} {'-':>10} {'-':>6} {'-':>7} {'-':>7} {bright:>7.0f}")
        time.sleep(0.4)
except KeyboardInterrupt:
    pass
finally:
    eyes.stop()
print("done.")
