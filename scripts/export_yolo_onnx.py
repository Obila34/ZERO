"""Export a YOLO detector to an ONNX graph with NMS baked in.

Run this ONCE on any machine that has ultralytics + torch (your laptop or the GPU
node — NOT the Pi, which deliberately has no torch). It writes ``<model>.onnx``
next to the weights, which the Pi then runs through onnxruntime (see
``zero/vision/detector.py``). The checked-in ``yolo11n.onnx`` / ``yolo11s.onnx``
were produced this way; regenerate only if you change the model or image size.

Closed-vocabulary (COCO-80) — the default::

    pip install ultralytics onnx
    python scripts/export_yolo_onnx.py                    # -> yolo11n.onnx
    python scripts/export_yolo_onnx.py yolo11s.pt 640     # -> yolo11s.onnx

Open-vocabulary (YOLO-World) — detect ARBITRARY objects by name. Pass a
``--vocab`` file (one prompt word per line) and a YOLO-World checkpoint. The
script calls ``model.set_classes(vocab)`` so the exported graph's class ids index
into *your* vocabulary, then writes a sibling ``<model>.names.json`` that the Pi
detector auto-loads (so class ids map back to words with no torch on the Pi)::

    pip install ultralytics onnx clip
    python scripts/export_yolo_onnx.py yolov8s-worldv2.pt 640 \
        --vocab scripts/vocab_indoor.txt
    # -> yolov8s-worldv2.onnx  +  yolov8s-worldv2.names.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_vocab(path: str) -> list[str]:
    """Read a vocabulary file: one prompt word/phrase per line, '#' comments."""
    names: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        word = line.split("#", 1)[0].strip()
        if word:
            names.append(word)
    if not names:
        raise SystemExit(f"[export] vocab file {path} has no words")
    return names


def main(argv: list[str]) -> int:
    # Split off the optional "--vocab <file>" flag; keep positional args simple.
    vocab_path = None
    if "--vocab" in argv:
        i = argv.index("--vocab")
        try:
            vocab_path = argv[i + 1]
        except IndexError:
            raise SystemExit("[export] --vocab needs a file path")
        argv = argv[:i] + argv[i + 2:]

    weights = argv[0] if argv else "yolo11n.pt"
    imgsz = int(argv[1]) if len(argv) > 1 else 640

    from ultralytics import YOLO

    model = YOLO(weights)

    names: list[str] | None = None
    if vocab_path is not None:
        names = _load_vocab(vocab_path)
        print(f"[export] open-vocab: {len(names)} classes from {vocab_path}")
        # YOLO-World bakes the text prompts into the visual head at export time.
        model.set_classes(names)

    print(f"[export] {weights} -> ONNX (imgsz={imgsz}, nms=True, opset=12)")
    path = model.export(format="onnx", nms=True, imgsz=imgsz, opset=12)
    print(f"[export] wrote {path}")

    # Persist the class names next to the .onnx. The ONNX graph carries only
    # integer class ids, so the Pi needs this sidecar to turn them into words.
    names = names or [model.names[i] for i in range(len(model.names))]
    sidecar = Path(path).with_suffix("").with_suffix(".names.json")
    sidecar.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] wrote {sidecar} ({len(names)} names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
