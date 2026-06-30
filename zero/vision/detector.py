"""2D object detection with Ultralytics YOLO11 Nano (Pi-side, local).

``Detector`` turns one RGB frame into a list of ``Detection`` objects (the shared
vision schema). It keeps only the configured classes + confidence threshold. The
``color`` field is left ``None`` here — ``ColorNamer`` fills it in.

ultralytics/torch are imported lazily (they are slow to import and only needed on
the camera node), so the package imports fine on a box without them.
"""
from __future__ import annotations

from typing import Optional

from zero.utils.logging import get_logger
from zero.vision.schemas import Detection

log = get_logger("vision.detect")


class Detector:
    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.35,
                 iou: float = 0.45, device: str = "cpu", imgsz: int = 640,
                 classes: Optional[list[str]] = None):
        self._model_path = str(model_path)
        self._conf = float(confidence)
        self._iou = float(iou)
        self._device = str(device)
        self._imgsz = int(imgsz)
        self._wanted = {str(c).lower() for c in (classes or [])}
        self._model = None
        self._class_ids: Optional[list[int]] = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "ultralytics is not installed. Run "
                "`pip install -r requirements-vision.txt`."
            ) from exc

        self._model = YOLO(self._model_path)
        names = self._model.names  # {idx: name}
        if self._wanted:
            self._class_ids = [idx for idx, name in names.items()
                               if str(name).lower() in self._wanted]
            missing = self._wanted - {str(n).lower() for n in names.values()}
            if missing:
                log.warning("classes not in model, ignored: %s", sorted(missing))
        else:
            self._class_ids = None  # keep all classes
        log.info("YOLO loaded (%s) on %s", self._model_path, self._device)

    def detect(self, frame_rgb) -> list[Detection]:
        """Run detection on one RGB frame and return filtered ``Detection``s."""
        import cv2

        self._ensure_model()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self._model.predict(
            frame_bgr, conf=self._conf, iou=self._iou, imgsz=self._imgsz,
            device=self._device, classes=self._class_ids, verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            detections.append(Detection(
                label=str(names[int(cls_id)]),
                bbox=[float(x1), float(y1), w, h],
                confidence=float(conf),
                color=None,
            ))
        return detections
