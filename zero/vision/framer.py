"""FaceFramer — ZERO's digital gaze: a moveable attention window in the frame.

A software pan/tilt: the framer keeps an (x, y, w, h) window inside the camera
frame, recentered on the face it is following. No motors — the "gaze" is a
crop that slides around the full frame, so ZERO can attend to a face without
the robot (or the camera) physically moving, while YOLO keeps seeing the whole
room on the same stream.

Phase 1 (this file): instant recentering + edge clamping, fixed window size,
and freeze-in-place when no face is visible. Which face to follow is decided
by an internal ``IouTracker`` — the same per-instance tracker the scene uses —
so the window sticks to ONE person while another walks past, instead of
snapping to whichever face is largest each frame.

Planned next (deliberately not here yet): two-speed smoothing, zoom scaling
from face size, and re-acquisition when the face leaves the frame.

``update()`` is called from the Eyes detection tick and must never raise past
its own logging — a broken framer must not blind perception.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from zero.utils.logging import get_logger
from zero.vision.tracker import IouTracker

log = get_logger("vision.framer")


@dataclass
class _FaceDet:
    """Minimal detection shape for IouTracker (.label/.bbox/.confidence)."""
    label: str
    bbox: tuple
    confidence: float
    pts: np.ndarray | None = None


class FaceFramer:
    def __init__(self, detector, window_frac: float = 0.45,
                 confirm_s: float = 0.4, lost_s: float = 2.0):
        """``detector`` needs ``detect_faces(frame_rgb) -> [(bbox, pts)]``
        (a ``YuNetDetector``). ``window_frac`` sizes the window as a fraction
        of each frame dimension. ``confirm_s``/``lost_s`` tune the internal
        tracker: quick lock-on, and a face briefly lost (head turned away)
        keeps its track — and the window — for ``lost_s`` before going stale.
        """
        self._detector = detector
        self._frac = min(1.0, max(0.1, float(window_frac)))
        # move_cooldown 0: "moved" events are the scene's concern, not ours —
        # we read track positions directly every tick.
        self._tracker = IouTracker(iou_min=0.2, confirm_s=float(confirm_s),
                                   lost_s=float(lost_s))
        self._lock = threading.Lock()
        self._window: tuple[int, int, int, int] | None = None
        self._primary_tid: int | None = None

    @property
    def window(self) -> tuple[int, int, int, int] | None:
        """Current attention window (x, y, w, h), or None before first frame."""
        with self._lock:
            return self._window

    def crop(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """The attention window's pixels from ``frame_rgb`` (a view, not a
        copy), or None before the first update."""
        win = self.window
        if win is None:
            return None
        x, y, w, h = win
        return frame_rgb[y:y + h, x:x + w]

    # ── per-tick update ──────────────────────────────────────────────────────
    def update(self, frame_rgb: np.ndarray, now: float) -> tuple | None:
        """Detect faces, advance the tracker, recenter the window on the
        primary face. Returns the window. Never raises."""
        try:
            return self._update(frame_rgb, now)
        except Exception as e:
            log.debug("framer update failed: %s", e)
            return self.window

    def _update(self, frame_rgb: np.ndarray, now: float) -> tuple:
        H, W = frame_rgb.shape[:2]
        win_w, win_h = int(W * self._frac), int(H * self._frac)
        with self._lock:
            if self._window is None:  # first frame: rest at frame center
                self._window = self._clamp(W // 2, H // 2, win_w, win_h, W, H)

        faces = self._detector.detect_faces(frame_rgb)
        dets = [_FaceDet("face", tuple(bbox), 1.0, pts) for bbox, pts in faces]
        self._tracker.update(dets, now)
        primary = self._pick_primary()
        if primary is None:
            return self.window  # no confirmed face -> freeze in place

        # Anchor on the landmark centroid when this tick matched a detection
        # (bbox identical) — steadier than the bbox center, which wanders with
        # hair/chin extent. A coasting track (matched nothing this tick, still
        # inside lost_s) anchors on its last bbox center.
        cx, cy = primary.center()
        for d in dets:
            if d.bbox == primary.bbox and d.pts is not None:
                cx, cy = (float(v) for v in d.pts.mean(axis=0))
                break

        new_win = self._clamp(cx, cy, win_w, win_h, W, H)
        with self._lock:
            self._window = new_win
        return new_win

    def _pick_primary(self):
        """The track we were following if it's still alive, else the largest
        confirmed face — sticky, so a passer-by doesn't steal the gaze."""
        tracks = [t for t in self._tracker.tracks() if t.label == "face"]
        if not tracks:
            return None
        if self._primary_tid is not None:
            for t in tracks:
                if t.tid == self._primary_tid:
                    return t
        primary = max(tracks, key=lambda t: t.bbox[2] * t.bbox[3])
        self._primary_tid = primary.tid
        return primary

    @staticmethod
    def _clamp(cx: float, cy: float, win_w: int, win_h: int,
               W: int, H: int) -> tuple[int, int, int, int]:
        """Window of (win_w, win_h) centered as close to (cx, cy) as fits."""
        x = int(round(cx - win_w / 2))
        y = int(round(cy - win_h / 2))
        x = max(0, min(W - win_w, x))
        y = max(0, min(H - win_h, y))
        return (x, y, win_w, win_h)
