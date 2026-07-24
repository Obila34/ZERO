"""InteractionPolicy — when ZERO may speak up on its own.

The hard part of proactivity is staying quiet. Every proactive utterance must
pass ALL gates:

  * a person is present (KNOWN always; UNKNOWN only if engage_unknown),
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

UNKNOWN_PID = -1  # bucket key for an unrecognized (but present) person


class InteractionPolicy:
    def __init__(self, *, enabled: bool = True,
                 greet_cooldown_s: float = 4 * 3600.0,
                 curiosity_cooldown_s: float = 30 * 60.0,
                 remark_cooldown_s: float = 5 * 60.0,
                 max_per_hour: int = 6,
                 quiet_hours: tuple[int, int] | None = (22, 7),
                 presence_reset_s: float = 20 * 60.0,
                 engage_unknown: bool = False):
        self.enabled = bool(enabled)
        self.greet_cooldown_s = float(greet_cooldown_s)
        self.curiosity_cooldown_s = float(curiosity_cooldown_s)
        self.remark_cooldown_s = float(remark_cooldown_s)
        self.max_per_hour = int(max_per_hour)
        self.quiet_hours = quiet_hours
        self.presence_reset_s = float(presence_reset_s)
        # When True, ZERO also greets/engages people it doesn't recognise
        # (unknown -> bucketed under UNKNOWN_PID). Still fully rate-limited.
        self.engage_unknown = bool(engage_unknown)
        self._last_by_kind: dict[str, float] = {}
        self._greeted: dict[int, float] = {}     # person_id -> last greeted ts
        self._recent: list[float] = []           # timestamps of the last hour
        # Bandit-lite (Phase 3): per-kind EMA of how proactive utterances land
        # (reward in [-1,1] from zero/learning/reward.py). Kinds that keep
        # falling flat get LONGER cooldowns; kinds that land get shorter ones.
        self._outcome_ema: dict[str, float] = {}
        self._outcome_alpha = 0.3

    @staticmethod
    def _key(person_id: int | None) -> int:
        return UNKNOWN_PID if person_id is None else person_id

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

    def may_greet(self, person_id: int | None, now: float | None = None) -> bool:
        now = now or time.time()
        if not self.enabled or self._in_quiet_hours(now) or not self._rate_ok(now):
            return False
        if person_id is None and not self.engage_unknown:
            return False                         # strangers off unless enabled
        last_kind = self._last_by_kind.get("greet", 0.0)
        if now - last_kind < 60.0:               # never two greetings in a minute
            return False
        last = self._greeted.get(self._key(person_id), 0.0)
        if now - last < self.greet_cooldown_s * self.cooldown_scale("greet"):
            return False
        return True

    def may_ask_curiosity(self, person_id: int | None,
                          now: float | None = None) -> bool:
        now = now or time.time()
        if not self.enabled or self._in_quiet_hours(now) or not self._rate_ok(now):
            return False
        if person_id is None and not self.engage_unknown:
            return False                         # ask strangers only if enabled
        last = self._last_by_kind.get("curiosity", 0.0)
        return (now - last >= self.curiosity_cooldown_s
                * self.cooldown_scale("curiosity"))

    def may_remark(self, now: float | None = None) -> bool:
        """A generic ambient remark to whoever's around (keeps a lingering
        conversation alive when there's no queued question)."""
        now = now or time.time()
        if not self.enabled or self._in_quiet_hours(now) or not self._rate_ok(now):
            return False
        return (now - self._last_by_kind.get("remark", 0.0)
                >= self.remark_cooldown_s * self.cooldown_scale("remark"))

    def defer_announcement(self, now: float | None = None) -> bool:
        """Timers/reminders: speak unless it's quiet hours (defer, don't drop)."""
        return self._in_quiet_hours(now or time.time())

    # ── adaptive cooldowns (Phase 3 bandit-lite) ──────────────────────────────
    def record_outcome(self, kind: str, reward: float) -> None:
        """Fold one proactive outcome (reward in [-1,1]) into the kind's EMA."""
        reward = max(-1.0, min(1.0, float(reward)))
        prev = self._outcome_ema.get(kind)
        a = self._outcome_alpha
        self._outcome_ema[kind] = (reward if prev is None
                                   else (1 - a) * prev + a * reward)
        log.debug("proactive outcome %s: %.2f (ema %.2f)", kind, reward,
                  self._outcome_ema[kind])

    def cooldown_scale(self, kind: str) -> float:
        """Multiplier on the kind's base cooldown from its outcome history:
        EMA -1 -> 3.0x (back way off), 0/unknown -> 1.0x, +1 -> 0.5x."""
        ema = self._outcome_ema.get(kind, 0.0)
        if ema >= 0.0:
            return 1.0 - 0.5 * ema           # 1.0 .. 0.5
        return 1.0 - 2.0 * ema               # 1.0 .. 3.0

    # ── record what actually happened ─────────────────────────────────────────
    def spoke(self, kind: str, person_id: int | None = None,
              now: float | None = None) -> None:
        now = now or time.time()
        self._last_by_kind[kind] = now
        self._recent.append(now)
        if kind == "greet":
            self._greeted[self._key(person_id)] = now

    def person_left(self, person_id: int | None, now: float | None = None) -> None:
        """After a long absence the next arrival greets again."""
        now = now or time.time()
        key = self._key(person_id)
        last = self._greeted.get(key)
        if last is not None and now - last >= self.presence_reset_s:
            del self._greeted[key]
