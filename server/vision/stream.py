"""Continuous streaming perception on the GPU.

The Pi streams video frames up (``POST /ingest``); this runs YOLO11l/x with an
object tracker on every frame — on the GPU, fast — and keeps a live, always-fresh
``SceneState``. When the user speaks, the conversation reads that pre-computed
scene instead of waiting on detection, so vision is effectively zero-latency on
the critical path: the eyes are already open.

Tracking (ByteTrack, via Ultralytics ``model.track(persist=True)``) gives objects
persistent ids across frames — the basis for object permanence and "someone just
walked in" events later.

torch/ultralytics are imported lazily so importing this module (or /health) never
pulls them in; the model loads on the first ingested frame.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

try:
    from shared.schemas import Detection
except ImportError:  # pragma: no cover - import path fallback (cwd=repo root)
    from server.vision.shared.schemas import Detection


# ── HSV colour naming (same vocabulary as the Pi's ColorNamer) ────────────────
_HUE_BANDS = [
    (0, 9, "red"), (10, 20, "orange"), (21, 33, "yellow"), (34, 85, "green"),
    (86, 95, "cyan"), (96, 130, "blue"), (131, 145, "purple"), (146, 159, "pink"),
    (160, 179, "red"),
]


def _dominant_color(frame_bgr, bbox, center_crop=0.6, min_sat=50, min_val=40,
                    white_val=200, min_ratio=0.10, min_px=50) -> Optional[str]:
    import cv2

    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * center_crop, h * center_crop
    x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    x1, y1 = int(round(cx + nw / 2)), int(round(cy + nh / 2))
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0].reshape(-1), hsv[:, :, 1].reshape(-1), hsv[:, :, 2].reshape(-1)
    if hh.size < min_px:
        return None
    chroma = (ss >= min_sat) & (vv >= min_val)
    if float(np.count_nonzero(chroma)) / float(hh.size) >= min_ratio:
        hues = hh[chroma]
        weights = (ss[chroma] / 255.0) * (vv[chroma] / 255.0)
        dom = int(np.argmax(np.bincount(hues, weights=weights, minlength=180)))
        val = float(np.mean(vv[chroma]))
        for lo, hi, name in _HUE_BANDS:
            if lo <= dom <= hi:
                if name in ("orange", "yellow", "red") and val < 130:
                    return "brown"
                return name
        return "red"
    mean_v = float(np.mean(vv))
    return "black" if mean_v < min_val else ("white" if mean_v >= white_val else "gray")


class LiveDetector:
    """YOLO + tracker on the GPU, keeping tracker state across streamed frames."""

    def __init__(self, config: dict):
        s = config.get("stream", {})
        self._model_path = str(s.get("model", "yolo11l.pt"))
        self._device = str(s.get("device", "0"))          # CUDA device id or 'cpu'
        self._conf = float(s.get("confidence", 0.35))
        self._iou = float(s.get("iou", 0.45))
        self._imgsz = int(s.get("imgsz", 640))
        self._tracker = str(s.get("tracker", "bytetrack.yaml"))
        self._color_top_n = int(s.get("color_top_n", 6))
        self._wanted = {str(c).lower() for c in (s.get("classes") or [])}
        self._model = None
        self._class_ids = None
        self._lock = threading.Lock()  # Ultralytics tracker state isn't re-entrant

    def _ensure(self):
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self._model_path)
        names = self._model.names
        if self._wanted:
            self._class_ids = [i for i, n in names.items() if str(n).lower() in self._wanted]
        print(f"[stream] YOLO tracker loaded ({self._model_path}) on device={self._device}",
              flush=True)

    def track(self, frame_bgr) -> list[Detection]:
        self._ensure()
        with self._lock:
            results = self._model.track(
                frame_bgr, persist=True, tracker=self._tracker, conf=self._conf,
                iou=self._iou, imgsz=self._imgsz, device=self._device,
                classes=self._class_ids, verbose=False,
            )
        if not results:
            return []
        r = results[0]
        boxes = r.boxes
        if boxes is None or boxes.xyxy is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = (boxes.id.cpu().numpy().astype(int) if boxes.id is not None
               else [None] * len(xyxy))
        names = r.names

        dets: list[Detection] = []
        rows = []
        for (x1, y1, x2, y2), cf, ci, tid in zip(xyxy, confs, clss, ids):
            w, h = float(x2 - x1), float(y2 - y1)
            if w <= 1 or h <= 1:
                continue
            rows.append((float(x1), float(y1), w, h, float(cf), int(ci),
                         None if tid is None else int(tid)))
        # Colour only the biggest few boxes (cost control).
        rows.sort(key=lambda r: r[2] * r[3], reverse=True)
        for i, (x, y, w, h, cf, ci, tid) in enumerate(rows):
            color = None
            if i < self._color_top_n:
                try:
                    color = _dominant_color(frame_bgr, (x, y, w, h))
                except Exception:
                    color = None
            dets.append(Detection(label=str(names[ci]), bbox=[x, y, w, h],
                                  confidence=cf, color=color, track_id=tid))
        return dets


class LiveScene:
    """Thread-safe holder of the latest streamed scene."""

    def __init__(self):
        self._lock = threading.Lock()
        self._dets: list[Detection] = []
        self._frame = None
        self._ts = 0.0
        self._idx = 0

    def update(self, detections: list[Detection], frame_bgr=None):
        with self._lock:
            self._dets = detections
            if frame_bgr is not None:
                self._frame = frame_bgr
            self._ts = time.time()
            self._idx += 1

    def snapshot(self):
        with self._lock:
            age = (time.time() - self._ts) if self._ts else float("inf")
            frame = None if self._frame is None else self._frame.copy()
            return list(self._dets), frame, self._idx, age
