"""Proactive: policy gates and the trigger watcher (with fake eyes/identity)."""
from __future__ import annotations

import time

from zero.events import EventBus
from zero.proactive.curiosity import CuriosityStore
from zero.proactive.policy import InteractionPolicy
from zero.proactive.triggers import TriggerSource


def _policy(**kw):
    kw.setdefault("quiet_hours", None)   # tests shouldn't depend on wall clock
    return InteractionPolicy(**kw)


# ── policy gates ─────────────────────────────────────────────────────────────
def test_greet_once_then_cooldown():
    p = _policy(greet_cooldown_s=3600)
    now = time.time()
    assert p.may_greet(1, now) is True
    p.spoke("greet", 1, now)
    assert p.may_greet(1, now + 300) is False          # within cooldown
    assert p.may_greet(2, now + 300) is True           # other person fine


def test_greet_returns_after_absence():
    p = _policy(greet_cooldown_s=3600, presence_reset_s=600)
    now = time.time()
    p.spoke("greet", 1, now)
    p.person_left(1, now + 700)                        # away > reset window
    assert p.may_greet(1, now + 800) is True


def test_hourly_rate_cap():
    p = _policy(max_per_hour=2, greet_cooldown_s=0)
    now = time.time()
    p.spoke("greet", 1, now)
    p.spoke("curiosity", 1, now + 120)
    assert p.may_greet(2, now + 240) is False          # cap of 2 reached
    assert p.may_greet(2, now + 3700) is True          # window rolled


def test_quiet_hours_block_proactive():
    p = InteractionPolicy(quiet_hours=(0, 24))         # always quiet
    assert p.may_greet(1) is False
    assert p.may_ask_curiosity(1) is False
    assert p.defer_announcement() is True


def test_curiosity_needs_known_person_and_cooldown():
    p = _policy(curiosity_cooldown_s=600)
    now = time.time()
    assert p.may_ask_curiosity(None, now) is False     # stranger: never
    assert p.may_ask_curiosity(1, now) is True
    p.spoke("curiosity", 1, now)
    assert p.may_ask_curiosity(1, now + 60) is False


# ── trigger watcher (single ticks, no thread) ────────────────────────────────
class FakeEyes:
    def __init__(self):
        self.frame = None
        self.unknowns: list[str] = []

    def current_frame(self):
        return self.frame

    def recent_unknowns(self, within_s=3600.0):
        out, self.unknowns = self.unknowns, []
        return out


class FakeIdentity:
    def __init__(self):
        self.result = None

    def identify(self, audio=None, frame_rgb=None):
        from zero.identity.fuser import STRANGER

        return self.result or STRANGER


class FakeMemory:
    def __init__(self):
        self.consolidations = 0

    def consolidate(self, reflect_fn=None):
        self.consolidations += 1
        return {"forgotten": 0, "insights": 0}


def _trigger(tmp_path, eyes, identity, memory=None, curiosity=None, **kw):
    return TriggerSource(
        events=EventBus(), policy=_policy(greet_cooldown_s=3600),
        eyes=eyes, identity=identity, curiosity=curiosity, memory=memory,
        is_idle=lambda: True, **kw)


def test_greeting_fires_once_for_known_face(tmp_path):
    from zero.identity.fuser import IdentityResult

    eyes, ident = FakeEyes(), FakeIdentity()
    eyes.frame = object()
    ident.result = IdentityResult(1, "David", 0.9, via="face")
    t = _trigger(tmp_path, eyes, ident)
    now = time.time()
    t._tick(now)
    events = t._events.drain()
    assert len(events) == 1
    assert events[0].kind == "greet" and "David" in events[0].text
    assert events[0].meta["open_conversation"] is True
    t._tick(now + 5)                                   # same person, no re-greet
    assert t._events.drain() == []


def test_stranger_gets_no_greeting(tmp_path):
    eyes, ident = FakeEyes(), FakeIdentity()
    eyes.frame = object()                              # face present, unknown
    t = _trigger(tmp_path, eyes, ident)
    t._tick(time.time())
    assert t._events.drain() == []


def test_lingering_person_gets_curiosity_question(tmp_path):
    from zero.identity.fuser import IdentityResult

    eyes, ident = FakeEyes(), FakeIdentity()
    eyes.frame = object()
    ident.result = IdentityResult(1, "David", 0.9, via="face")
    cur = CuriosityStore(str(tmp_path / "c.sqlite"))
    cur.add("k", "What's that red thing on the shelf?", 5.0)
    t = _trigger(tmp_path, eyes, ident, curiosity=cur,
                 linger_before_question_s=10.0)
    now = time.time()
    t._policy.spoke("greet", 1, now - 4000)            # already greeted long ago
    t._tick(now)                                       # arrival (no greet: cooldown)
    t._events.drain()
    t._tick(now + 15)                                  # lingered past threshold
    events = t._events.drain()
    assert len(events) == 1 and events[0].kind == "curiosity"
    assert "red thing" in events[0].text
    assert cur.pending_count() == 0                    # marked asked


def test_alone_learns_silently_never_speaks(tmp_path):
    eyes, ident = FakeEyes(), FakeIdentity()
    eyes.frame = None                                  # empty room
    eyes.unknowns = ["vase", "trophy"]
    mem = FakeMemory()
    cur = CuriosityStore(str(tmp_path / "c.sqlite"))
    t = _trigger(tmp_path, eyes, ident, memory=mem, curiosity=cur,
                 consolidate_interval_s=0.0)
    t._tick(time.time())
    assert t._events.drain() == []                     # silence in an empty room
    assert cur.pending_count() == 2                    # ...but it got curious
    assert mem.consolidations == 1                     # ...and consolidated


def test_not_idle_means_no_ticks(tmp_path):
    from zero.identity.fuser import IdentityResult

    eyes, ident = FakeEyes(), FakeIdentity()
    eyes.frame = object()
    ident.result = IdentityResult(1, "David", 0.9, via="face")
    t = TriggerSource(events=EventBus(), policy=_policy(), eyes=eyes,
                      identity=ident, is_idle=lambda: False,
                      check_interval_s=0.01)
    t.start()
    time.sleep(0.15)
    t.stop()
    assert t._events.drain() == []                     # busy: watcher stays out
