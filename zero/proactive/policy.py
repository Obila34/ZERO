"""InteractionPolicy — when ZERO may speak up on its own.

The hard part of proactivity is staying quiet. Every proactive utterance must
pass ALL gates:

  * a KNOWN person is present (strangers are never proactively engaged),
  * quiet hours aren't active,
  * the per-kind cooldown has elapsed (greetings don't repeat on every frame),
  * this person hasn't already been greeted this arrival (presence-session),
  * a global rate cap (at most N proactive utterances per hour).

Timers/reminders bypass presence gates — the user explicitly asked for those —
but still respect quiet hours by being deferred, not dropped.
"""
from __future__ import annotations

import time

from zero.utils.logging import get_logger

log = get_logger("proactive.policy")


class InteractionPolicy:
    def __init__(self, *, enabled: bool = True,
                 greet_cooldown_s: float = 4 * 3600.0,
                 curiosity_cooldown_s: float = 30 * 60.0,
                 max_per_hour: int = 6,
                 quiet_hours: tuple[int, int] | None = (22, 7),
                 presence_reset_s: float = 20 * 60.0):
        self.enabled = bool(enabled)
        self.greet_cooldown_s = float(greet_cooldown_s)
        self.curiosity_cooldown_s = float(curiosity_cooldown_s)
        self.max_per_hour = int(max_per_hour)
        self.quiet_hours = quiet_hours
        self.presence_reset_s = float(presence_reset_s)
        self._last_by_kind: dict[str, float] = {}
        self._greeted: dict[int, float] = {}     # person_id -> last greeted ts
        self._recent: list[float] = []           # timestamps of the last hour

    # ── gates ─────────────────────────────────────────────────────────────────
    def _in_quiet_hours(self, now: float) -> bool:
        if not self.quiet_hours:
            return False
        start, end = self.quiet_hours
        hour = time.localtime(now).tm_hour
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end       # wraps midnight, e.g. 22-7

    def _rate_ok(self, now: float) -> bool:
        self._recent = [t for t in self._recent if now - t < 3600.0]
        return len(self._recent) < self.max_per_hour

    def may_greet(self, person_id: int, now: float | None = None) -> bool:
        now = now or time.time()
        if not self.enabled or self._in_quiet_hours(now) or not self._rate_ok(now):
            return False
        last_kind = self._last_by_kind.get("greet", 0.0)
        if now - last_kind < 60.0:               # never two greetings in a minute
            return False
        last = self._greeted.get(person_id, 0.0)
        if now - last < self.greet_cooldown_s:
            return False
        return True

    def may_ask_curiosity(self, person_id: int | None,
                          now: float | None = None) -> bool:
        now = now or time.time()
        if not self.enabled or self._in_quiet_hours(now) or not self._rate_ok(now):
            return False
        if person_id is None:                    # only ask people ZERO knows
            return False
        last = self._last_by_kind.get("curiosity", 0.0)
        return now - last >= self.curiosity_cooldown_s

    def defer_announcement(self, now: float | None = None) -> bool:
        """Timers/reminders: speak unless it's quiet hours (defer, don't drop)."""
        return self._in_quiet_hours(now or time.time())

    # ── record what actually happened ─────────────────────────────────────────
    def spoke(self, kind: str, person_id: int | None = None,
              now: float | None = None) -> None:
        now = now or time.time()
        self._last_by_kind[kind] = now
        self._recent.append(now)
        if kind == "greet" and person_id is not None:
            self._greeted[person_id] = now

    def person_left(self, person_id: int, now: float | None = None) -> None:
        """After a long absence the next arrival greets again."""
        now = now or time.time()
        last = self._greeted.get(person_id)
        if last is not None and now - last >= self.presence_reset_s:
            del self._greeted[person_id]
