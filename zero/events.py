"""EventBus — how background faculties get a word in.

Timers firing, reminders coming due, proactive triggers (Phase 5): anything
that wants ZERO to *speak without being spoken to* posts an Event here. The
main loop drains the bus at safe moments (idle wake-wait, turn boundaries) so
announcements never collide with an in-flight reply. Thread-safe, tiny.
"""
from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    kind: str                 # e.g. "timer", "reminder", "proactive"
    text: str                 # what to say (already phrased for speech)
    created_at: float = field(default_factory=time.time)
    person_id: int | None = None   # who it concerns, if anyone
    meta: dict = field(default_factory=dict)


class EventBus:
    def __init__(self, maxsize: int = 64):
        self._q: "queue.Queue[Event]" = queue.Queue(maxsize=maxsize)

    def post(self, event: Event) -> bool:
        """Queue an event; drops (False) when full rather than blocking a
        background thread — a lost nudge beats a wedged timer."""
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            return False

    def drain(self) -> list[Event]:
        """All pending events, oldest first. Non-blocking."""
        out: list[Event] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def peek_pending(self) -> bool:
        return not self._q.empty()
