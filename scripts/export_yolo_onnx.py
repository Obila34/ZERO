"""Export YOLO11n to an ONNX graph with NMS baked in.

Run this ONCE on any machine that has ultralytics + torch (your laptop or the GPU
node — NOT the Pi, which deliberately has no torch). It writes ``yolo11n.onnx``
next to the weights, which the Pi then runs through onnxruntime (see
``zero/vision/detector.py``). The checked-in ``yolo11n.onnx`` was produced this
way; regenerate it only if you change the model or image size.

    pip install ultralytics onnx
    python scripts/export_yolo_onnx.py            # -> yolo11n.onnx
    python scripts/export_yolo_onnx.py yolo11s.pt 640
"""
from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    weights = argv[0] if argv else "yolo11n.pt"
    imgsz = int(argv[1]) if len(argv) > 1 else 640
    from ultralytics import YOLO

    print(f"[export] {weights} -> ONNX (imgsz={imgsz}, nms=True, opset=12)")
    path = YOLO(weights).export(format="onnx", nms=True, imgsz=imgsz, opset=12)
    print(f"[export] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
