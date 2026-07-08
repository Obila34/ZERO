"""Minimal IoU tracker — per-INSTANCE identity across frames, so "the same cup
moved" is distinguishable from "a new cup appeared". Deliberately tiny (greedy
same-label IoU matching, no Kalman/embedding) — enough for remark-grade motion
events on a slow detection stream, not for dense multi-object scenes.

Events (label strings, consumed by Eyes):
  ("new",   label)  — a track survived ``confirm_s`` (stable arrival)
  ("moved", label)  — a confirmed track's center shifted by > ``move_frac`` of
                      its own size (box jitter-proof), per-track cooldown
  ("gone",  label)  — a confirmed track unseen for ``lost_s``
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    tid: int
    label: str
    bbox: tuple
    first_ts: float
    last_ts: float
    confirmed: bool = False
    ref_center: tuple = field(default=None)  # anchor for movement detection
    last_move_ts: float = 0.0

    def center(self):
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    def size(self) -> float:
        return max(1.0, (self.bbox[2] ** 2 + self.bbox[3] ** 2) ** 0.5)


class IouTracker:
    def __init__(self, iou_min: float = 0.25, confirm_s: float = 1.0,
                 lost_s: float = 1.0, move_frac: float = 0.75,
                 move_cooldown_s: float = 30.0):
        self.iou_min = float(iou_min)
        self.confirm_s = float(confirm_s)
        self.lost_s = float(lost_s)
        self.move_frac = float(move_frac)
        self.move_cooldown_s = float(move_cooldown_s)
        self._tracks: list[Track] = []
        self._ids = itertools.count(1)

    def update(self, detections, now: float | None = None) -> list[tuple[str, str]]:
        """Feed one frame's detections (need .label and .bbox=(x,y,w,h)).
        Returns this frame's events."""
        now = time.time() if now is None else now
        events: list[tuple[str, str]] = []

        # Greedy best-IoU matching, same label only.
        pairs = sorted(
            ((_iou(t.bbox, d.bbox), t, d)
             for t in self._tracks for d in detections
             if t.label == getattr(d, "label", None)),
            key=lambda p: p[0], reverse=True)
        matched_t, matched_d = set(), set()
        for iou, t, d in pairs:
            if iou < self.iou_min or id(t) in matched_t or id(d) in matched_d:
                continue
            matched_t.add(id(t))
            matched_d.add(id(d))
            t.bbox, t.last_ts = tuple(d.bbox), now
            if not t.confirmed and now - t.first_ts >= self.confirm_s:
                t.confirmed = True
                t.ref_center = t.center()
                events.append(("new", t.label))
            elif t.confirmed and t.ref_center is not None:
                dx = t.center()[0] - t.ref_center[0]
                dy = t.center()[1] - t.ref_center[1]
                if ((dx * dx + dy * dy) ** 0.5 > self.move_frac * t.size()
                        and now - t.last_move_ts >= self.move_cooldown_s):
                    t.last_move_ts = now
                    t.ref_center = t.center()  # re-anchor after the event
                    events.append(("moved", t.label))

        # Unmatched detections start tentative tracks.
        for d in detections:
            if id(d) not in matched_d and getattr(d, "bbox", None) is not None:
                self._tracks.append(Track(next(self._ids), d.label,
                                          tuple(d.bbox), now, now))

        # Expire tracks not seen for lost_s.
        alive = []
        for t in self._tracks:
            if now - t.last_ts >= self.lost_s:
                if t.confirmed:
                    events.append(("gone", t.label))
            else:
                alive.append(t)
        self._tracks = alive
        return events
