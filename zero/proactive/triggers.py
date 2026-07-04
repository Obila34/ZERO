"""TriggerSource — the background watcher that gives ZERO initiative.

A daemon thread that, ONLY while ZERO is idle, periodically:

  1. sees who's in front of it → a time-aware greeting that OPENS a
     conversation. A known face is greeted by name; an unrecognised person is
     greeted with an ice-breaker when engage_unknown is on.
  2. when that person lingers, surfaces a queued curiosity question, or a
     generic ambient remark if there's none — keeping the interaction alive;
  3. runs the silent idle-learning tick: memory consolidation and turning
     unfamiliar-object sightings into queued questions (never spoken to an
     empty room — asked later, opportunistically).

All speech candidates go through InteractionPolicy; this thread never talks.
It posts Events on the bus, which main.py drains at safe moments.
"""
from __future__ import annotations

import random
import threading
import time

from zero.events import Event, EventBus
from zero.proactive.curiosity import CuriosityStore
from zero.proactive.policy import InteractionPolicy
from zero.utils.logging import get_logger

log = get_logger("proactive")

# Openers for someone ZERO doesn't recognise. The conversation flows to the LLM
# on their reply; these just break the ice without needing the model.
_UNKNOWN_OPENERS = (
    "{tod}! I don't think we've met — I'm Zero. What are you working on?",
    "Oh, hey there. {tod}. What brings you by?",
    "{tod}. I'm Zero — nice to see someone around. What are you up to?",
)
# Ambient remarks to keep a lingering conversation alive when there's no queued
# question. Deliberately open-ended so the LLM can take it anywhere.
_AMBIENT_REMARKS = (
    "You've been at it a while — how's it going over there?",
    "Still busy? I'm here if you want to talk something through.",
    "Anything I can help with while you work?",
)


def _time_of_day(now: float) -> str:
    h = time.localtime(now).tm_hour
    if h < 12:
        return "Good morning"
    if h < 17:
        return "Good afternoon"
    return "Good evening"


class TriggerSource:
    def __init__(self, *, events: EventBus, policy: InteractionPolicy,
                 eyes=None, identity=None, curiosity: CuriosityStore | None = None,
                 memory=None, is_idle=lambda: False,
                 check_interval_s: float = 3.0,
                 linger_before_question_s: float = 45.0,
                 consolidate_interval_s: float = 1800.0,
                 engage_unknown: bool = False):
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
        self._engage_unknown = bool(engage_unknown)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._present_since: float | None = None
        self._present_person: tuple[int | None, str | None] | None = None
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
        prev_pid = self._present_person[0] if self._present_person else object()
        if self._present_person is None or prev_pid != pid:
            # Someone new arrived (or an unknown became a known face).
            self._present_person, self._present_since = person, now
            if self._policy.may_greet(pid, now):
                text = self._greeting_text(name, now)
                self._events.post(Event(
                    kind="greet", text=text, person_id=pid,
                    meta={"open_conversation": True}))
                self._policy.spoke("greet", pid, now)
                log.info("greeting %s", name or "someone")
            return

        # Same person lingering: a queued curiosity question first, else a
        # generic ambient remark — either way, keep the interaction alive.
        if self._present_since is None or now - self._present_since < self.linger_s:
            return
        if (self._curiosity is not None
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
                return
        if self._policy.may_remark(now):
            self._events.post(Event(
                kind="remark", text=random.choice(_AMBIENT_REMARKS),
                person_id=pid, meta={"open_conversation": True}))
            self._policy.spoke("remark", pid, now)
            log.info("ambient remark")

    def _greeting_text(self, name: str | None, now: float) -> str:
        """Time-aware greeting — named for people ZERO knows, an ice-breaker
        for those it doesn't."""
        tod = _time_of_day(now)
        if name:
            return f"{tod}, {name}. Good to see you."
        return random.choice(_UNKNOWN_OPENERS).format(tod=tod)

    def _look_for_person(self) -> tuple[int | None, str | None] | None:
        """Who (or whether someone) is in front of ZERO right now:
        (pid, name) for a known face; (None, None) for an unrecognised person
        when engage_unknown is on; None if no one is there."""
        if self._eyes is None:
            return None
        frame = self._eyes.current_frame()
        if self._identity is not None and frame is not None:
            result = self._identity.identify(frame_rgb=frame)
            if result.is_known:
                return result.person_id, result.name
        if self._engage_unknown:
            try:
                if "person" in self._eyes.visible_labels():
                    return None, None            # present, but unrecognised
            except Exception:  # a scene read must never break the watcher
                pass
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
