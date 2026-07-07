"""ZERO's eyes — always-on perception + per-turn visual context.

``Eyes`` runs the camera -> YOLO11n -> color loop on a background thread from the
moment ZERO starts, continuously updating a ``SceneState`` with the latest
keyframe and named detections. This is the key to a human-feeling pipeline: when
the wake word fires and you ask "what's this?", the scene is *already* perceived,
so there is no camera spin-up or detection latency on the critical path.

Per turn the orchestrator calls:

* ``local_context()`` — instant, GPU-free hint ("3 people, a cup") for ambient
  awareness, safe to attach to every turn.
* ``visual_context(question)`` — the same detector hint plus 2-3 recent keyframes
  for the multimodal LLM to actually *see*. The model reads real colors, actions
  and positions off the frames; the hint only stops it undercounting. No depth /
  bearing / coordinate facts reach the speech path — those read as clinical.

cv2/numpy/onnxruntime are only touched once ``start()`` runs, so importing this
module is cheap and safe on a node without them.
"""
from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass, field
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
    text: str = ""                       # detector hint ("3 people, a cup")
    images: list[str] = field(default_factory=list)  # recent keyframe JPEGs (b64)


class Eyes:
    _PREVIEW_WIN = "ZERO - what the camera sees"

    def __init__(self, camera: CameraStream, detector: Detector,
                 color_namer: ColorNamer, client: Optional[VisionClient] = None,
                 *, color_top_n: int = 5, max_items: int = 6,
                 use_gpu: bool = True, multimodal: bool = False,
                 jpeg_quality: int = 80, detect_interval_s: float = 0.0,
                 frames_per_look: int = 2, look_window_s: float = 1.0,
                 vlm_fallback: bool = False,
                 preview: bool = False, preview_scale: float = 1.0,
                 preview_mode: str = "auto", preview_host: str = "127.0.0.1",
                 preview_port: int = 8008, learned=None,
                 unknown_conf: float = 0.45,
                 change_note_cooldown_s: float = 120.0):
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
        self._frames_per_look = max(1, int(frames_per_look))
        self._look_window_s = float(look_window_s)
        self._vlm_fallback = bool(vlm_fallback)
        self._preview = bool(preview)
        self._preview_scale = max(0.1, float(preview_scale))
        self._preview_mode = str(preview_mode)
        self._preview_host = str(preview_host)
        self._preview_port = int(preview_port)
        self._preview_sink = None  # built in start() if preview is on

        self._scene = SceneState()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._gpu_ok = False
        # Learned objects (Phase 4): user-taught names override COCO labels on
        # per-turn context; low-confidence, unmatched sightings feed curiosity.
        self._learned = learned
        self._unknown_conf = float(unknown_conf)
        self._unknowns: dict[str, float] = {}   # label -> last seen ts
        self._unknowns_lock = threading.Lock()
        # Scene diffing: debounced appear/disappear events -> spontaneous
        # remarks. A label must be present (or gone) STABLE_S before it counts.
        self._change_cooldown_s = float(change_note_cooldown_s)
        self._change_lock = threading.Lock()
        self._changes: list[str] = []
        self._last_change_note = 0.0
        self._started_at = 0.0
        self._first_seen: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._stable: set[str] = set()

    _STABLE_S = 2.0      # presence/absence must persist this long to count
    _SETTLE_S = 8.0      # startup grace: the initial scene isn't "new"

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
        if self._preview:
            from zero.vision.preview import build_preview

            self._preview_sink = build_preview(
                self._preview_mode, title=self._PREVIEW_WIN,
                scale=self._preview_scale, host=self._preview_host,
                port=self._preview_port, jpeg_quality=self._jpeg_quality,
            )
        self._stop.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, name="Eyes", daemon=True)
        self._thread.start()
        log.info("eyes open — perceiving continuously")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._camera.stop()
        if self._preview_sink is not None:
            self._preview_sink.close()
            self._preview_sink = None

    # ── perception loop ──────────────────────────────────────────────────────
    def _loop(self) -> None:
        self._detect_errors = 0
        while not self._stop.is_set():
            frame = self._camera.read_new(timeout=1.0)
            if frame is None:
                continue
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
            self._track_changes(detections)
            if self._preview_sink is not None and self._preview_sink.ok:
                self._preview_sink.show(frame, detections)
            if self._detect_interval > 0:
                time.sleep(self._detect_interval)

    def _track_changes(self, detections) -> None:
        """Debounced scene diffing: record labels that stably appear/disappear,
        as phrased remark candidates consumed by scene_changes()."""
        now = time.time()
        if now - self._started_at < self._SETTLE_S:
            for d in detections:  # seed the baseline silently
                lbl = d.label.lower()
                self._first_seen.setdefault(lbl, now)
                self._last_seen[lbl] = now
                self._stable.add(lbl)
            return
        labels = {d.label.lower() for d in detections}
        events: list[str] = []
        for lbl in labels:
            self._last_seen[lbl] = now
            first = self._first_seen.setdefault(lbl, now)
            if lbl not in self._stable and now - first >= self._STABLE_S:
                self._stable.add(lbl)
                events.append("someone new is in view" if lbl == "person"
                              else f"a {lbl} just came into view")
        for lbl in list(self._stable):
            if lbl in labels:
                continue
            if now - self._last_seen.get(lbl, 0.0) >= self._STABLE_S:
                self._stable.discard(lbl)
                self._first_seen.pop(lbl, None)
                if lbl != "person":  # person-left is proactive/identity's job
                    events.append(f"the {lbl} is no longer in view")
        if events:
            with self._change_lock:
                self._changes = (self._changes + events)[-4:]

    def scene_changes(self) -> list[str]:
        """Pending phrased scene-change remarks (max 2), rate-limited by the
        cooldown; draining clears them. [] when nothing new or too soon."""
        now = time.time()
        with self._change_lock:
            if not self._changes:
                return []
            if now - self._last_change_note < self._change_cooldown_s:
                self._changes.clear()  # stale by the time we may speak again
                return []
            out, self._changes = self._changes[:2], []
            self._last_change_note = now
            return out

    def _fill_colors(self, frame_rgb, detections) -> None:
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
    def visible_labels(self) -> set[str]:
        """Lowercased labels of the objects currently detected. Cheap (one
        snapshot read) — used to route a turn to the visual path when the user
        names something ZERO can literally see right now ('what color is the
        cup?'), which no fixed word list could enumerate."""
        snap = self._scene.snapshot()
        return {d.label.lower() for d in snap.detections}

    def current_frame(self):
        """Latest camera frame (RGB copy), or None if nothing has been seen yet.
        Used by identity (face matching) and learning (object crops)."""
        return self._scene.snapshot().frame_rgb

    def _with_learned(self, snap) -> list:
        """Detections with learned-name overrides applied; also logs sightings
        of low-confidence objects nobody has named yet (curiosity fodder)."""
        dets = snap.detections
        if self._learned is None or snap.frame_rgb is None or not dets:
            return dets
        try:
            annotated = self._learned.annotate(snap.frame_rgb, dets)
        except Exception as e:  # learned names must never break perception
            log.debug("learned annotate failed: %s", e)
            return dets
        now = time.time()
        with self._unknowns_lock:
            for orig, ann in zip(dets, annotated):
                if (ann.label == orig.label            # no learned override
                        and orig.confidence < self._unknown_conf
                        and orig.label.lower() != "person"):
                    self._unknowns[orig.label.lower()] = now
        return annotated

    def recent_unknowns(self, within_s: float = 3600.0) -> list[str]:
        """Labels of unfamiliar (low-confidence, unlearned) objects sighted
        recently — consumed by curiosity. Draining resets the log."""
        now = time.time()
        with self._unknowns_lock:
            out = [lbl for lbl, ts in self._unknowns.items()
                   if now - ts <= within_s]
            self._unknowns.clear()
        return out

    def teach_object(self, name: str, person_id: int | None = None) -> bool:
        """Bind ``name`` to what the camera is looking at right now: the largest
        detected object's crop, or the centre of the frame if nothing is boxed."""
        if self._learned is None:
            return False
        snap = self._scene.snapshot()
        frame = snap.frame_rgb
        if frame is None:
            return False
        H, W = frame.shape[:2]
        if snap.detections:
            big = max(snap.detections, key=lambda d: d.bbox[2] * d.bbox[3])
            x, y, w, h = (int(v) for v in big.bbox)
            crop = frame[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        else:
            my, mx = int(H * 0.2), int(W * 0.2)
            crop = frame[my:H - my, mx:W - mx]
        try:
            return self._learned.teach(name, crop, person_id=person_id)
        except Exception as e:
            log.warning("teach_object failed: %s", e)
            return False

    def local_context(self) -> str:
        """Instant, GPU-free detector hint ('3 people, a cup'), or '' if empty."""
        snap = self._scene.snapshot()
        return phrasing.detector_hint(self._with_learned(snap), self._max_items)

    def visual_context(self, question: str = "") -> VisualContext:
        """Context for a visual turn: the detector hint + a few recent keyframes.

        The frames let the multimodal LLM see real colors, actions and layout;
        the hint just keeps it from undercounting. No GPU depth/bearing facts —
        those read as clinical and are no longer on the speech path.
        """
        snap = self._scene.snapshot()
        images: list[str] = []
        if self._multimodal:
            frames = self._scene.recent_frames(self._frames_per_look,
                                               self._look_window_s)
            images = [b for b in (self._encode(f) for f in frames) if b]
        text = phrasing.detector_hint(self._with_learned(snap), self._max_items)
        # Long-tail naming: when the main LLM can't see (multimodal off) but the
        # GPU VLM is reachable, ask it to describe the frame so ZERO can still name
        # objects outside the detector's vocabulary. Never on the multimodal path
        # (the LLM already sees the frames) and never fatal to the turn.
        if not self._multimodal and self._vlm_fallback:
            described = self.describe(question)
            if described:
                text = f"{text}; {described}" if text else described
        return VisualContext(text=text, images=images)

    def describe(self, question: str = "") -> Optional[str]:
        """Ask the GPU VLM (Qwen2-VL) to describe the current keyframe in words.

        This is the open-ended fallback for objects the local detector can't name.
        Returns the VLM sentence, or None if the GPU is unavailable / it fails —
        vision must never break a turn.
        """
        if self._client is None or not self._gpu_ok:
            return None
        snap = self._scene.snapshot()
        frame = snap.frame_rgb
        if frame is None:
            return None
        try:
            reply = self._client.analyze(frame, snap.detections, question=question)
            return reply.strip() or None
        except Exception as e:
            log.debug("VLM describe failed: %s", e)
            return None

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
