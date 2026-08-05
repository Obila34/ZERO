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
                 preview_port: int = 8008, learned=None, framer=None,
                 unknown_conf: float = 0.45,
                 change_note_cooldown_s: float = 120.0,
                 world=None, motion=None, gate=None, narrator=None,
                 narration_max_age_s: float = 20.0, tier1_budget=None):
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
        self._framer = framer  # digital gaze (FaceFramer) — optional
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
        # Per-instance IoU tracks — "the SAME cup moved", not just labels
        # coming and going. Fed EVERY detection (people included) so the world
        # state gets persistent instance identities; the remark path filters
        # person-moved events out (people move constantly).
        from zero.vision.tracker import IouTracker

        self._tracker = IouTracker()
        # Learned-name annotation runs in the BACKGROUND, never per turn. Each
        # annotate() is a remote CLIP round trip per detection (or a Pi-CPU
        # embed when the tunnel is down) — measured 1.6-2.0s for 4 detections,
        # and it was being paid inline on turns that asked nothing visual
        # (`vision+recall=1775ms, visual=False`). The scene barely changes
        # between refreshes, so a turn attaches whatever annotation is already
        # cached and NEVER waits: stale-but-instant beats fresh-but-late in a
        # live conversation.
        self._ann_lock = threading.Lock()
        self._ann_dets: list | None = None
        self._ann_ts = 0.0
        self._ann_thread: Optional[threading.Thread] = None
        self._ann_interval_s = 2.0   # background refresh cadence
        self._ann_ttl_s = 8.0        # cached annotation older than this = stale
        # World state (Phase 2): Tier 0 motion every frame, Tier 1 tracks after
        # every detection pass, Tier 2 narration via the optional narrator.
        # All optional — Eyes runs exactly as before when world is None.
        self._world = world
        self._motion = motion
        self._gate = gate
        self._narrator = narrator
        self._narration_max_age_s = float(narration_max_age_s)
        self._last_dets: list = []
        # Tier 1 hard ceiling (Phase 3): even under constant motion, detection
        # may not eat more than its duty-cycle budget of wall time.
        self._tier1_budget = tier1_budget

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
        if self._learned is not None:
            self._ann_thread = threading.Thread(
                target=self._annotate_loop, name="EyesAnnotate", daemon=True)
            self._ann_thread.start()
        if self._narrator is not None:
            self._narrator.start()
        log.info("eyes open — perceiving continuously")

    def stop(self) -> None:
        if self._narrator is not None:
            self._narrator.stop()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._ann_thread is not None:
            self._ann_thread.join(timeout=2.0)
            self._ann_thread = None
        self._camera.stop()
        if self._preview_sink is not None:
            self._preview_sink.close()
            self._preview_sink = None

    @property
    def world(self):
        """The live WorldState (or None) — the read surface for main/proactive/
        memory: ``eyes.world.snapshot()`` / ``wait_for_change()``."""
        return self._world

    # ── perception loop ──────────────────────────────────────────────────────
    def _loop(self) -> None:
        self._detect_errors = 0
        while not self._stop.is_set():
            frame = self._camera.read_new(timeout=1.0)
            if frame is None:
                continue
            now = time.time()
            # Tier 0: motion on EVERY frame (cheap), published to the world and
            # used to gate Tier 1 — a still room doesn't get re-detected at
            # full cadence.
            motion_active = True
            if self._motion is not None:
                try:
                    level, motion_active = self._motion.update(frame)
                    if self._world is not None:
                        self._world.update_motion(level, motion_active, ts=now)
                except Exception as e:   # Tier 0 must never blind the camera
                    log.debug("motion detect failed: %s", e)
            run_detect = (self._gate.should_detect(motion_active, now)
                          if self._gate is not None else True)
            if (run_detect and self._tier1_budget is not None
                    and not self._tier1_budget.allowed(now)):
                run_detect = False           # hard ceiling beats motion

            if run_detect:
                detections: list = []
                t_detect = time.perf_counter()
                try:
                    detections = self._detector.detect(frame)
                    self._fill_colors(frame, detections)
                except Exception as e:  # detection failed, but we SAW the frame
                    self._detect_errors += 1
                    if self._detect_errors <= 3 or self._detect_errors % 30 == 0:
                        log.warning("detection failed (x%d): %s",
                                    self._detect_errors, e)
                if self._tier1_budget is not None:
                    self._tier1_budget.record(time.perf_counter() - t_detect,
                                              now)
                self._last_dets = detections
                # Always publish the frame + whatever detections we got, so a
                # broken detector never makes it look like the camera is blind.
                self._scene.update(detections, frame_rgb=frame)
                self._track_changes(detections)
                self._track_instances(detections, now)
                if self._framer is not None:
                    # Digital gaze rides the same motion-gated cadence: a
                    # still room means the face isn't moving either, so the
                    # frozen window stays correct. update() never raises.
                    self._framer.update(frame, now)
            else:
                # Gated frame: keep the LAST detections current (the scene has
                # not changed — that is why the gate closed) but still publish
                # the fresh frame for identity/keyframes/preview.
                detections = self._last_dets
                self._scene.update(detections, frame_rgb=frame)
            if self._preview_sink is not None and self._preview_sink.ok:
                self._preview_sink.show(
                    frame, detections,
                    attention=(self._framer.window
                               if self._framer is not None else None))
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

    def _track_instances(self, detections, now: float) -> None:
        """Per-instance tracking (Tier 1): update tracks from ALL detections,
        publish the scene graph to the world state, and queue 'the cup just
        moved' remark candidates (people excluded — they move constantly;
        identity/proactive own them)."""
        try:
            dets = [d for d in detections
                    if getattr(d, "bbox", None) is not None]
            events = self._tracker.update(dets, now)
        except Exception as e:  # tracking must never break perception
            log.debug("instance tracking failed: %s", e)
            return
        if self._world is not None:
            try:
                from zero.world.state import WorldEvent, WorldObject

                objects = [WorldObject(track_id=t.tid, label=t.label,
                                       bbox=tuple(t.bbox), confidence=t.conf,
                                       first_seen=t.first_ts, last_seen=t.last_ts)
                           for t in self._tracker.tracks()]
                kinds = {"new": "appeared", "gone": "left", "moved": "moved"}
                wevents = [WorldEvent(kind=kinds[k], label=lbl, ts=now)
                           for k, lbl in events if k in kinds]
                # Startup settle: the initial scene is baseline, not events.
                if now - self._started_at < self._SETTLE_S:
                    wevents = []
                self._world.update_objects(objects, wevents, ts=now)
            except Exception as e:  # world publish must never break perception
                log.debug("world publish failed: %s", e)
        if now - self._started_at < self._SETTLE_S:
            return  # tracker state warms up silently
        moved = [f"the {label} just moved" for kind, label in events
                 if kind == "moved" and label.lower() != "person"]
        if moved:
            with self._change_lock:
                self._changes = (self._changes + moved)[-4:]

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

    def attention(self) -> tuple | None:
        """The digital-gaze window (x, y, w, h) in frame pixels, or None when
        the framer is off / hasn't seen a frame yet. Later phases feed this
        crop to identity and the VLM keyframes."""
        return self._framer.window if self._framer is not None else None

    def current_frame(self):
        """Latest camera frame (RGB copy), or None if nothing has been seen yet.
        Used by identity (face matching) and learning (object crops)."""
        return self._scene.snapshot().frame_rgb

    def _with_learned(self, snap) -> list:
        """Detections with learned-name overrides — from the background cache,
        NEVER computed inline. annotate() costs a remote round trip (or a
        Pi-CPU embed) per detection and was stalling every conversation turn
        by 1.6-2.0s, including turns that asked nothing visual. A turn takes
        whatever the annotator last produced; if that's stale (or the loop
        hasn't run yet), the raw detector labels are the honest answer."""
        dets = snap.detections
        if self._learned is None or not dets:
            return dets
        with self._ann_lock:
            ann, ts = self._ann_dets, self._ann_ts
        if ann is not None and time.monotonic() - ts <= self._ann_ttl_s:
            return ann
        return dets

    def _annotate_loop(self) -> None:
        """Background learned-object annotation: refresh the cached overrides
        every couple of seconds so per-turn context reads are instant. Also
        owns the unknown-object (curiosity) tracking, which used to ride the
        inline annotate."""
        while not self._stop.is_set():
            self._stop.wait(self._ann_interval_s)
            if self._stop.is_set():
                return
            snap = self._scene.snapshot()
            dets = snap.detections
            if snap.frame_rgb is None or not dets:
                continue
            try:
                t0 = time.monotonic()
                annotated = self._learned.annotate(snap.frame_rgb, dets)
                dt = time.monotonic() - t0
                if dt > 1.0:
                    log.info("learned annotate slow: %.2fs for %d detections "
                             "(background — turns are not waiting on this)",
                             dt, len(dets))
            except Exception as e:  # learned names must never break perception
                log.debug("learned annotate failed: %s", e)
                continue
            with self._ann_lock:
                self._ann_dets = annotated
                self._ann_ts = time.monotonic()
            now = time.time()
            with self._unknowns_lock:
                for orig, ann in zip(dets, annotated):
                    if (ann.label == orig.label        # no learned override
                            and orig.confidence < self._unknown_conf
                            and orig.label.lower() != "person"):
                        self._unknowns[orig.label.lower()] = now

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

    def _world_note(self) -> str:
        """Fresh Tier 2 narration, or ''. Instant (lock-free snapshot read) —
        this is how the conversational brain reads the world state without
        triggering any analysis."""
        if self._world is None:
            return ""
        snap = self._world.snapshot()
        if snap.narration and (snap.narration_age_s()
                               <= self._narration_max_age_s):
            return snap.narration
        return ""

    def local_context(self) -> str:
        """Instant, GPU-free detector hint ('3 people, a cup'), or '' if empty.
        A fresh world narration (Tier 2) is appended when available — still
        instant, it was computed in the background."""
        snap = self._scene.snapshot()
        hint = phrasing.detector_hint(self._with_learned(snap), self._max_items)
        note = self._world_note()
        if note:
            return f"{hint}; {note}" if hint else note
        return hint

    def visual_context(self, question: str = "") -> VisualContext:
        """Context for a visual turn: the detector hint + a few recent keyframes.

        The frames let the multimodal LLM see real colors, actions and layout;
        the hint just keeps it from undercounting. No GPU depth/bearing facts —
        those read as clinical and are no longer on the speech path.
        """
        snap = self._scene.snapshot()
        images: list[str] = []
        if self._multimodal:
            # Attention crop: a tight crop of the largest person so the LLM
            # reads faces, expressions and held objects the wide frame blurs.
            # It REPLACES one wide frame (total stays frames_per_look) so a
            # tight shared GPU doesn't get an extra image to encode per turn.
            crop = self._person_crop(snap)
            n_wide = max(1, self._frames_per_look - (1 if crop else 0))
            frames = self._scene.recent_frames(n_wide, self._look_window_s)
            images = [b for b in (self._encode(f) for f in frames) if b]
            if crop:
                images.append(crop)
        text = phrasing.detector_hint(self._with_learned(snap), self._max_items)
        # World narration (Tier 2): already-computed scene understanding, free
        # to attach. On the multimodal path the LLM sees the frames anyway, but
        # the narration adds the temporal "what has been happening" the frozen
        # frames can't carry.
        note = self._world_note()
        if note:
            text = f"{text}; {note}" if text else note
        # Long-tail naming: when the main LLM can't see (multimodal off) but the
        # GPU VLM is reachable, ask it to describe the frame so ZERO can still name
        # objects outside the detector's vocabulary. Never on the multimodal path
        # (the LLM already sees the frames) and never fatal to the turn.
        if not self._multimodal and self._vlm_fallback:
            described = self.describe(question)
            if described:
                text = f"{text}; {described}" if text else described
        return VisualContext(text=text, images=images)

    def _person_crop(self, snap) -> Optional[str]:
        """Encoded crop (15% margin) of the largest person in frame, or None."""
        frame = snap.frame_rgb
        if frame is None:
            return None
        people = [d for d in snap.detections if d.label.lower() == "person"]
        if not people:
            return None
        big = max(people, key=lambda d: d.bbox[2] * d.bbox[3])
        x, y, w, h = (int(v) for v in big.bbox)
        H, W = frame.shape[:2]
        mx, my = int(w * 0.15), int(h * 0.15)
        crop = frame[max(0, y - my):min(H, y + h + my),
                     max(0, x - mx):min(W, x + w + mx)]
        if crop.size == 0:
            return None
        return self._encode(crop)

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
