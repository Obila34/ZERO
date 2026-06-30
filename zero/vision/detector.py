"""2D object detection with YOLO11n via ONNX Runtime (Pi-side, local).

Runs the **ONNX** export of YOLO11n on ``onnxruntime`` — the same runtime the wake
word already uses — so the Pi needs **no torch, no CUDA, no ultralytics** (which
on ARM drag in ~2 GB of unusable NVIDIA wheels). The model is exported once with
NMS baked into the graph (``scripts/export_yolo_onnx.py``), so the output is
already a clean list of [x1, y1, x2, y2, score, class] rows.

``detect(frame_rgb)`` returns the shared ``Detection`` objects; ``color`` is left
None for ``ColorNamer`` to fill. cv2/numpy/onnxruntime are imported lazily.
"""
from __future__ import annotations

from typing import Optional

from zero.utils.logging import get_logger
from zero.vision.coco import COCO_NAMES
from zero.vision.schemas import Detection

log = get_logger("vision.detect")


class Detector:
    def __init__(self, model_path: str = "yolo11n.onnx", confidence: float = 0.35,
                 iou: float = 0.45, device: str = "cpu", imgsz: int = 640,
                 classes: Optional[list[str]] = None,
                 names: Optional[list[str]] = None):
        self._model_path = str(model_path)
        self._conf = float(confidence)
        self._iou = float(iou)  # kept for API compatibility (NMS is baked in)
        self._device = str(device)
        self._imgsz = int(imgsz)
        self._wanted = {str(c).lower() for c in (classes or [])}
        self._names = list(names) if names else COCO_NAMES
        self._session = None
        self._input_name = None

    def _ensure_model(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "onnxruntime is not installed. Run "
                "`pip install -r requirements-vision.txt`."
            ) from exc

        providers = ["CPUExecutionProvider"]
        if self._device.lower() in ("cuda", "gpu"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(self._model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        log.info("YOLO ONNX loaded (%s) providers=%s", self._model_path,
                 self._session.get_providers())

    def _letterbox(self, frame_rgb):
        """Resize keeping aspect ratio into imgsz x imgsz, pad with 114 gray.

        Returns (input_nchw_float32, ratio, pad_w, pad_h).
        """
        import cv2
        import numpy as np

        h, w = frame_rgb.shape[:2]
        s = self._imgsz
        r = min(s / h, s / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        pad_w, pad_h = (s - nw) // 2, (s - nh) // 2
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
        # HWC uint8 RGB -> NCHW float32 0..1
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob), r, pad_w, pad_h

    def detect(self, frame_rgb) -> list[Detection]:
        import numpy as np

        self._ensure_model()
        H, W = frame_rgb.shape[:2]
        blob, r, pad_w, pad_h = self._letterbox(frame_rgb)
        outputs = self._session.run(None, {self._input_name: blob})

        # End-to-end (NMS-in-graph) export -> a single (1, N, 6) array of
        # [x1, y1, x2, y2, score, class] in letterboxed-imgsz pixel space.
        arr = np.asarray(outputs[0])
        if arr.ndim == 3:
            arr = arr[0]
        if arr.size == 0:
            return []

        detections: list[Detection] = []
        for row in arr:
            if row.shape[0] < 6:
                continue
            x1, y1, x2, y2, score, cls = (float(row[0]), float(row[1]),
                                          float(row[2]), float(row[3]),
                                          float(row[4]), int(round(float(row[5]))))
            if score < self._conf:
                continue
            label = self._names[cls] if 0 <= cls < len(self._names) else str(cls)
            if self._wanted and label.lower() not in self._wanted:
                continue
            # Undo letterbox: remove pad, divide by ratio -> original frame pixels.
            ox1 = (x1 - pad_w) / r
            oy1 = (y1 - pad_h) / r
            ox2 = (x2 - pad_w) / r
            oy2 = (y2 - pad_h) / r
            ox1 = max(0.0, min(ox1, W));  ox2 = max(0.0, min(ox2, W))
            oy1 = max(0.0, min(oy1, H));  oy2 = max(0.0, min(oy2, H))
            bw, bh = ox2 - ox1, oy2 - oy1
            if bw <= 1 or bh <= 1:
                continue
            detections.append(Detection(
                label=label, bbox=[ox1, oy1, bw, bh],
                confidence=score, color=None,
            ))
        return detections
