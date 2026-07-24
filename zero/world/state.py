"""WorldState — the shared live scene graph all perception tiers feed.

Concurrency model (the part that keeps the "instant answer" honest):

* The entire world is ONE immutable ``WorldSnapshot``. Writers never mutate a
  published snapshot — each update builds a new one and swaps the reference.
* Readers call ``snapshot()`` which is a bare attribute read: lock-free,
  wait-free, O(1), and always internally consistent (a snapshot can never be
  half-updated, because it never changes after publication). A reader holding
  a snapshot keeps a consistent view for as long as it likes.
* Writers (Tier 0 motion @ ~30 Hz, Tier 1 tracks @ ≤15 Hz, Tier 2 narration
  @ ~0.3 Hz, all on different threads) serialize on one lock. Each publish
  bumps a monotonic ``version`` and notifies a Condition, so event-driven
  readers (proactive, Phase 3 surprise gating) can sleep in
  ``wait_for_change()`` instead of polling.

Nothing here imports cv2/numpy/models — the world state is plain data and is
importable (and testable) anywhere.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

from zero.utils.logging import get_logger

log = get_logger("world")

_EVENT_CAP = 16          # recent events kept in every snapshot


@dataclass(frozen=True)
class WorldObject:
    """One tracked thing with a persistent identity across frames."""
    track_id: int
    label: str
    bbox: tuple            # (x, y, w, h) in frame pixels
    confidence: float
    first_seen: float      # epoch seconds — how long it has been in view
    last_seen: float

    def age_s(self, now: float | None = None) -> float:
        return max(0.0, (now or time.time()) - self.first_seen)


@dataclass(frozen=True)
class WorldEvent:
    """Something that just happened ("mug appeared", "person left")."""
    kind: str              # "appeared" | "left" | "moved"
    label: str
    ts: float
    track_id: int = 0

    def phrase(self) -> str:
        verb = {"appeared": "came into view", "left": "left the view",
                "moved": "moved"}.get(self.kind, self.kind)
        return f"the {self.label} {verb}"


@dataclass(frozen=True)
class WorldSnapshot:
    version: int = 0
    # Tier 1 — the scene graph proper
    objects: tuple = ()            # tuple[WorldObject, ...]
    objects_ts: float = 0.0
    # Tier 0 — motion
    motion_level: float = 0.0
    motion_active: bool = False
    motion_ts: float = 0.0
    # Tier 2 — narration
    narration: str = ""
    narration_ts: float = 0.0
    # Rolling recent events (newest last)
    events: tuple = ()             # tuple[WorldEvent, ...]

    def people(self) -> tuple:
        return tuple(o for o in self.objects if o.label.lower() == "person")

    def narration_age_s(self, now: float | None = None) -> float:
        if not self.narration_ts:
            return float("inf")
        return max(0.0, (now or time.time()) - self.narration_ts)


class WorldState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._snap = WorldSnapshot()
        self._events: list[WorldEvent] = []

    # ── readers (lock-free) ──────────────────────────────────────────────────
    def snapshot(self) -> WorldSnapshot:
        """The current world, immutable and internally consistent. Bare
        reference read — safe from any thread, never blocks."""
        return self._snap

    @property
    def version(self) -> int:
        return self._snap.version

    def wait_for_change(self, after_version: int,
                        timeout: float | None = None) -> WorldSnapshot:
        """Block until the world moves past ``after_version`` (or timeout);
        returns the then-current snapshot either way."""
        with self._cond:
            self._cond.wait_for(lambda: self._snap.version > after_version,
                                timeout=timeout)
            return self._snap

    # ── writers (serialized) ─────────────────────────────────────────────────
    def _publish(self, new_events: list[WorldEvent] | None = None,
                 **changes) -> WorldSnapshot:
        with self._cond:                      # the ONLY place the lock is taken
            if new_events:
                self._events = (self._events + list(new_events))[-_EVENT_CAP:]
            snap = replace(self._snap, version=self._snap.version + 1,
                           events=tuple(self._events), **changes)
            self._snap = snap
            self._cond.notify_all()
            return snap

    def update_motion(self, level: float, active: bool,
                      ts: float | None = None) -> None:
        """Tier 0 write — called every frame."""
        self._publish(motion_level=float(level), motion_active=bool(active),
                      motion_ts=ts or time.time())

    def update_objects(self, objects: list[WorldObject],
                       events: list[WorldEvent] | None = None,
                       ts: float | None = None) -> None:
        """Tier 1 write — the tracked scene graph after each detection pass."""
        self._publish(new_events=events, objects=tuple(objects),
                      objects_ts=ts or time.time())

    def update_narration(self, text: str, ts: float | None = None) -> None:
        """Tier 2 write — the narrator's latest scene understanding."""
        text = " ".join((text or "").split())
        if not text:
            return
        self._publish(narration=text, narration_ts=ts or time.time())

    # ── the instant answer ───────────────────────────────────────────────────
    def describe(self, max_items: int = 6, narration_max_age_s: float = 20.0,
                 events_max_age_s: float = 60.0) -> str:
        """One compact sentence for an LLM context: objects (with persistent
        durations), fresh narration, and very recent events. '' when the world
        is empty. Pure read — safe on the conversation hot path."""
        snap = self.snapshot()
        now = time.time()
        parts: list[str] = []

        if snap.objects:
            counted: dict[str, int] = {}
            for o in snap.objects:
                counted[o.label] = counted.get(o.label, 0) + 1
            items = []
            for label, n in list(counted.items())[:max_items]:
                if label.lower() == "person":
                    items.append(f"{n} people" if n > 1 else "a person")
                else:
                    items.append(f"{n} {label}s" if n > 1 else f"a {label}")
            parts.append(", ".join(items))

        if snap.narration and snap.narration_age_s(now) <= narration_max_age_s:
            parts.append(snap.narration)

        recent = [e for e in snap.events if now - e.ts <= events_max_age_s]
        if recent:
            parts.append("just now: " +
                         "; ".join(e.phrase() for e in recent[-3:]))
        return ". ".join(parts)
