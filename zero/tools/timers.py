"""TimerManager — timers and reminders that actually fire.

Each timer is a daemon thread that posts an Event on the bus when due; the
main loop drains the bus at safe moments and speaks the announcement. Timers
survive conversations (they belong to the app, not a turn) but not process
restarts — reminders that must survive a reboot are ALSO written to memory,
so ZERO can still honour "you asked me to remind you about X" after a crash.
"""
from __future__ import annotations

import threading
import time

from zero.events import Event, EventBus
from zero.utils.logging import get_logger

log = get_logger("tools.timer")


def humanize_s(seconds: float) -> str:
    s = int(round(seconds))
    if s >= 3600:
        h, rem = divmod(s, 3600)
        m = rem // 60
        return f"{h} hour{'s' if h != 1 else ''}" + (f" {m} minutes" if m else "")
    if s >= 60:
        m, sec = divmod(s, 60)
        return f"{m} minute{'s' if m != 1 else ''}" + (f" {sec} seconds" if sec else "")
    return f"{s} second{'s' if s != 1 else ''}"


class TimerManager:
    def __init__(self, events: EventBus):
        self._events = events
        self._lock = threading.Lock()
        self._active: dict[int, dict] = {}
        self._next_id = 1

    def set(self, seconds: float, label: str = "", *, kind: str = "timer",
            person_id: int | None = None) -> int:
        seconds = max(0.5, float(seconds))
        with self._lock:
            tid = self._next_id
            self._next_id += 1
            self._active[tid] = {
                "label": label, "due": time.time() + seconds, "kind": kind,
            }

        def fire():
            time.sleep(seconds)
            with self._lock:
                if tid not in self._active:   # cancelled
                    return
                del self._active[tid]
            text = (f"Reminder: {label}." if kind == "reminder" and label
                    else (f"Your {humanize_s(seconds)} timer is done."
                          + (f" It was for {label}." if label else "")))
            self._events.post(Event(kind=kind, text=text, person_id=person_id))
            log.info("%s fired: %s", kind, text)

        threading.Thread(target=fire, name=f"timer-{tid}", daemon=True).start()
        log.info("%s set: %s in %s", kind, label or "(unlabelled)", humanize_s(seconds))
        return tid

    def cancel(self, timer_id: int) -> bool:
        with self._lock:
            return self._active.pop(timer_id, None) is not None

    def active(self) -> list[tuple[int, str, float]]:
        """[(id, label, seconds_remaining)] for running timers."""
        now = time.time()
        with self._lock:
            return [(tid, t["label"], max(0.0, t["due"] - now))
                    for tid, t in sorted(self._active.items())]
