#!/usr/bin/env python
"""Collect a fine-tuning dataset from ZERO's own camera.

Saves a JPEG every --interval seconds until --count frames exist. Vary the
scene between shots (move objects, change lighting, different angles) — 200 to
500 varied frames of YOUR room beats any generic model.

    python scripts/capture_dataset.py --out dataset/images --count 300

Then: label (Roboflow / labelImg / CVAT, YOLO format), train, export:
see docs/FINETUNE.md.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/images")
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between saved frames")
    ap.add_argument("--index", type=int, default=1, help="camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    import cv2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = len(list(out.glob("*.jpg")))
    cap = cv2.VideoCapture(args.index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"camera {args.index} failed to open")
        return 1
    print(f"capturing to {out}/ every {args.interval}s "
          f"({existing} already there, target {args.count}) — Ctrl-C to stop")
    saved = existing
    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            name = out / f"frame_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
            print(f"  {saved}/{args.count}  {name.name}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
    print(f"done — {saved} frames in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
