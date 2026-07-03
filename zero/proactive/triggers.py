"""TriggerSource — the background watcher that gives ZERO initiative.

A daemon thread that, ONLY while ZERO is idle, periodically:

  1. looks for a known face in the current frame → greeting event
     ("Hey David, welcome back") that also OPENS a conversation;
  2. when a known person lingers, surfaces one queued curiosity question;
  3. runs the silent idle-learning tick: memory consolidation and turning
     unfamiliar-object sightings into queued questions (never spoken to an
     empty room — asked later, opportunistically).

All speech candidates go through InteractionPolicy; this thread never talks.
It posts Events on the bus, which main.py drains at safe moments.
"""
from __future__ import annotations

import threading
import time

from zero.events import Event, EventBus
from zero.proactive.curiosity import CuriosityStore
from zero.proactive.policy import InteractionPolicy
from zero.utils.logging import get_logger

log = get_logger("proactive")


class TriggerSource:
    def __init__(self, *, events: EventBus, policy: InteractionPolicy,
                 eyes=None, identity=None, curiosity: CuriosityStore | None = None,
                 memory=None, is_idle=lambda: False,
                 check_interval_s: float = 3.0,
                 linger_before_question_s: float = 45.0,
                 consolidate_interval_s: float = 1800.0):
        self._events = events
        self._policy = policy
        self._eyes = eyes
        self._identity = identity
        self._curiosity = curiosity
        self._memory = memory
        self._is_idle = is_idle
        self.check_interval_s = float(check_interval_s)
        self.linger_s = float(linger_before_question_s)
        self.consolidate_interval_s = float(consolidate_interval_s)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._present_since: float | None = None
        self._present_person: tuple[int, str] | None = None
        self._last_consolidate = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="proactive",
                                        daemon=True)
        self._thread.start()
        log.info("proactive triggers running (interval %.0fs)",
                 self.check_interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ── the watcher ───────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop.wait(self.check_interval_s):
            try:
                if not self._is_idle():
                    continue  # someone is already talking with ZERO
                self._tick(time.time())
            except Exception as e:  # the watcher must never die quietly
                log.warning("proactive tick failed: %s", e)

    def _tick(self, now: float) -> None:
        person = self._look_for_person()
        if person is None:
            if self._present_person is not None:
                self._policy.person_left(self._present_person[0], now)
            self._present_person, self._present_since = None, None
            self._idle_learning(now)             # alone → learn silently
            return

        pid, name = person
        if self._present_person is None or self._present_person[0] != pid:
            self._present_person, self._present_since = person, now
            if self._policy.may_greet(pid, now):
                self._events.post(Event(
                    kind="greet", text=f"Hey {name}, welcome back.",
                    person_id=pid, meta={"open_conversation": True}))
                self._policy.spoke("greet", pid, now)
                log.info("greeting %s", name)
            return

        # Same person lingering: one curiosity question, opportunistically.
        if (self._curiosity is not None and self._present_since is not None
                and now - self._present_since >= self.linger_s
                and self._policy.may_ask_curiosity(pid, now)):
            q = self._curiosity.next_question(pid)
            if q is not None:
                qid, text = q
                self._curiosity.mark_asked(qid)
                self._events.post(Event(
                    kind="curiosity", text=text, person_id=pid,
                    meta={"open_conversation": True}))
                self._policy.spoke("curiosity", pid, now)
                log.info("asking: %s", text)

    def _look_for_person(self) -> tuple[int, str] | None:
        """A known face in the current frame, via identity's face channel."""
        if self._eyes is None or self._identity is None:
            return None
        frame = self._eyes.current_frame()
        if frame is None:
            return None
        result = self._identity.identify(frame_rgb=frame)
        if result.is_known:
            return result.person_id, result.name
        return None

    def _idle_learning(self, now: float) -> None:
        """Alone: consolidate memory and queue questions about the unfamiliar."""
        if self._curiosity is not None and self._eyes is not None:
            for label in self._eyes.recent_unknowns():
                self._curiosity.note_unknown_object(label)
        if (self._memory is not None
                and now - self._last_consolidate >= self.consolidate_interval_s):
            self._last_consolidate = now
            try:
                stats = self._memory.consolidate()   # decay pass only (no LLM)
                if stats.get("forgotten"):
                    log.info("idle consolidation: %s", stats)
            except Exception as e:
                log.debug("idle consolidation failed: %s", e)
