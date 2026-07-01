"""ZERO's eyes — always-on perception + per-turn visual context.

``Eyes`` runs the camera -> YOLO11n -> color loop on a background thread from the
moment ZERO starts, continuously updating a ``SceneState`` with the latest
keyframe and named detections. This is the key to a human-feeling pipeline: when
the wake word fires and you ask "what's this?", the scene is *already* perceived,
so there is no camera spin-up or detection latency on the critical path.

Per turn the orchestrator calls:

* ``local_context()`` — instant, GPU-free "in view: ..." line for ambient
  awareness, safe to attach to every turn.
* ``visual_context(question)`` — snapshot + (optionally) GPU depth/bearing facts
  + the keyframe JPEG for a multimodal LLM. Used on visual turns. Degrades to the
  local detections if the GPU/tunnel is down.

cv2/numpy/onnxruntime are only touched once ``start()`` runs, so importing this
module is cheap and safe on a node without them.
"""
from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from typing import Optional

from zero.utils.logging import get_logger
from zero.vision import phrasing
from zero.vision.camera import CameraStream
from zero.vision.color_namer import ColorNamer
from zero.vision.detector import Detector
from zero.vision.gpu_client import VisionClient
from zero.vision.scene import SceneState

log = get_logger("vision.eyes")


@dataclass
class VisualContext:
    """What the orchestrator injects into a turn."""
    text: str = ""               # grounded/local description for the prompt
    image_b64: Optional[str] = None  # keyframe JPEG (only when multimodal + visual)
    grounded: bool = False       # True if GPU depth/bearing facts were used


class Eyes:
    def __init__(self, camera: CameraStream, detector: Optional[Detector],
                 color_namer: Optional[ColorNamer], client: Optional[VisionClient] = None,
                 *, color_top_n: int = 5, max_items: int = 6,
                 use_gpu: bool = True, multimodal: bool = False,
                 jpeg_quality: int = 80, detect_interval_s: float = 0.0,
                 mode: str = "local", stream_fps: float = 10.0):
        self._camera = camera
        self._detector = detector
        self._color = color_namer
        self._client = client
        self._color_top_n = int(color_top_n)
        self._max_items = int(max_items)
        self._use_gpu = bool(use_gpu)
        self._multimodal = bool(multimodal)
        self._jpeg_quality = int(jpeg_quality)
        self._detect_interval = float(detect_interval_s)
        # "local": YOLO on the Pi. "stream": push frames to the GPU which runs the
        # big YOLO+tracker continuously and returns the live scene.
        self._mode = str(mode)
        self._stream_period = 1.0 / stream_fps if stream_fps > 0 else 0.0

        self._scene = SceneState()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._gpu_ok = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Open the camera and begin the always-on perception loop."""
        self._camera.start()
        if self._client is not None and self._use_gpu:
            try:
                health = self._client.check_health()
                self._gpu_ok = True
                log.info("GPU vision server up: %s", health)
            except RuntimeError as e:
                self._gpu_ok = False
                log.warning("GPU vision server unreachable — distances disabled "
                            "(local detections still work): %s", e)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="Eyes", daemon=True)
        self._thread.start()
        log.info("eyes open — perceiving continuously")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._camera.stop()

    # ── perception loop ──────────────────────────────────────────────────────
    def _loop(self) -> None:
        self._detect_errors = 0
        while not self._stop.is_set():
            frame = self._camera.read_new(timeout=1.0)
            if frame is None:
                continue
            if self._mode == "stream":
                self._perceive_stream(frame)
                if self._stream_period > 0:
                    time.sleep(self._stream_period)
                continue
            # ── local mode: YOLO on the Pi ──
            detections: list = []
            try:
                detections = self._detector.detect(frame)
                self._fill_colors(frame, detections)
            except Exception as e:  # detection failed, but we still SAW the frame
                self._detect_errors += 1
                if self._detect_errors <= 3 or self._detect_errors % 30 == 0:
                    log.warning("detection failed (x%d): %s", self._detect_errors, e)
            # Always publish the frame + whatever detections we got, so a broken
            # detector never makes it look like the camera is blind.
            self._scene.update(detections, frame_rgb=frame)
            if self._detect_interval > 0:
                time.sleep(self._detect_interval)

    def _perceive_stream(self, frame) -> None:
        """Stream mode: push the frame to the GPU, store the tracked scene it
        returns. The GPU runs the big YOLO+tracker; the Pi just captures/uploads."""
        try:
            resp = self._client.ingest(frame, want_facts=False)
            self._scene.update(list(resp.detections), frame_rgb=frame)
            self._gpu_ok = True
        except RuntimeError as e:
            self._stream_errors = getattr(self, "_stream_errors", 0) + 1
            if self._stream_errors <= 3 or self._stream_errors % 50 == 0:
                log.warning("stream ingest failed (x%d): %s", self._stream_errors, e)
            self._gpu_ok = False
            # Fall back to local detection if we have a detector, so a dropped
            # stream doesn't blind ZERO entirely.
            if self._detector is not None:
                try:
                    dets = self._detector.detect(frame)
                    self._fill_colors(frame, dets)
                    self._scene.update(dets, frame_rgb=frame)
                    return
                except Exception:
                    pass
            self._scene.update([], frame_rgb=frame)

    def _fill_colors(self, frame_rgb, detections) -> None:
        if self._color is None:
            return
        # Only name the N largest boxes — color is the costly per-object step.
        ordered = sorted(
            detections,
            key=lambda d: d.bbox[2] * d.bbox[3], reverse=True,
        )
        for det in ordered[: self._color_top_n]:
            try:
                det.color = self._color.name(frame_rgb, det.bbox)
            except Exception:
                det.color = None

    # ── per-turn context ─────────────────────────────────────────────────────
    def local_context(self) -> str:
        """Instant, GPU-free 'in view: ...' line (or '' if nothing/never ran)."""
        snap = self._scene.snapshot()
        return phrasing.local_summary(snap.detections, self._max_items)

    def visual_context(self, question: str = "") -> VisualContext:
        """Build grounded context for a visual turn: GPU facts + keyframe.

        Falls back to the local detection summary if the GPU is down or there is
        no usable frame yet.
        """
        snap = self._scene.snapshot()
        image_b64 = None
        if self._multimodal and snap.frame_rgb is not None:
            image_b64 = self._encode(snap.frame_rgb)

        # Try grounded depth/bearing facts from the GPU.
        if self._client is not None and self._use_gpu and snap.frame_rgb is not None:
            try:
                facts = self._client.get_facts(snap.frame_rgb, snap.detections,
                                               question=question)
                self._gpu_ok = True
                line = phrasing.facts_line(facts, self._max_items)
                if line:
                    return VisualContext(text=line, image_b64=image_b64, grounded=True)
            except RuntimeError as e:
                self._gpu_ok = False
                log.warning("GPU facts failed, using local detections: %s", e)

        # Local fallback: detections + colors, no distance.
        return VisualContext(
            text=phrasing.local_summary(snap.detections, self._max_items),
            image_b64=image_b64, grounded=False,
        )

    def _encode(self, frame_rgb) -> Optional[str]:
        try:
            import cv2

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", frame_bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            if not ok:
                return None
            return base64.b64encode(buf).decode("ascii")
        except Exception as e:
            log.debug("keyframe encode failed: %s", e)
            return None
