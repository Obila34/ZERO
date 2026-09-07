"""Sign-sense watcher: per-frame sidecar answers -> debounced words on the
event bus, with the additive-layer OFF guarantee."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.events import EventBus
from zero.sign_sense.watch import SignWatcher, build_sign_sense


class FakeCfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


class FakeEyes:
    def raw_frame(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _cfg(over=None):
    c = FakeCfg({"sign_sense.fps": 50.0,          # fast test clock
                 "sign_sense.stable_frames": 3,
                 "sign_sense.word_gap_s": 0.15,
                 "sign_sense.min_confidence": 0.5})
    c.update(over or {})
    return c


def _scripted(watcher, answers):
    """Replace the sidecar round-trip with a scripted answer sequence
    (None = no hand this frame); repeats the last answer when exhausted."""
    it = iter(answers)
    last = answers[-1]

    def fake_ask(frame):
        nonlocal last
        try:
            last = next(it)
        except StopIteration:
            pass
        return last
    watcher._ask = fake_ask


def _letter(ch, conf=0.9, known=True):
    return {"letter": ch, "confidence": conf, "known": known}


def test_stable_letters_become_a_word_on_the_bus():
    bus = EventBus()
    w = SignWatcher.__new__(SignWatcher)   # build without starting the thread
    # construct manually so _ask can be scripted BEFORE the loop starts
    answers = ([_letter("h")] * 4 + [{"no_hand": True}] * 2
               + [_letter("i")] * 4 + [{"no_hand": True}] * 30)
    _init_watcher(w, _cfg(), bus, answers)
    _drive(w, seconds=1.0)
    evs = bus.drain()
    assert len(evs) == 1, f"expected one word event, got {evs}"
    assert evs[0].kind == "sign"
    assert evs[0].meta["sign_word"] == "hi"
    assert evs[0].meta["open_conversation"] is True
    assert "HI" in evs[0].text


def test_unknown_and_lowconf_frames_never_emit():
    bus = EventBus()
    w = SignWatcher.__new__(SignWatcher)
    answers = ([_letter("x", known=False)] * 10
               + [_letter("y", conf=0.2)] * 10
               + [{"no_hand": True}] * 30)
    _init_watcher(w, _cfg(), bus, answers)
    _drive(w, seconds=0.8)
    assert bus.drain() == []


def test_stray_single_letter_is_dropped_but_i_survives():
    bus = EventBus()
    w = SignWatcher.__new__(SignWatcher)
    answers = ([_letter("q")] * 4 + [{"no_hand": True}] * 20
               + [_letter("i")] * 4 + [{"no_hand": True}] * 30)
    _init_watcher(w, _cfg(), bus, answers)
    _drive(w, seconds=1.2)
    words = [e.meta["sign_word"] for e in bus.drain()]
    assert words == ["i"], words


def test_dead_sidecar_is_silent_and_harmless():
    bus = EventBus()
    w = SignWatcher.__new__(SignWatcher)
    _init_watcher(w, _cfg(), bus, [None])
    _drive(w, seconds=0.4)
    assert bus.drain() == []


def test_off_is_bit_identical():
    assert build_sign_sense(FakeCfg({}), FakeEyes(), EventBus()) is None
    assert build_sign_sense(FakeCfg({"sign_sense.enabled": True}),
                            None, EventBus()) is None   # no camera


# ── plumbing helpers ────────────────────────────────────────────────────────

def _init_watcher(w, cfg, bus, answers):
    import threading
    g = lambda k, d: cfg.get(k, d)   # noqa: E731
    w._eyes = FakeEyes()
    w._events = bus
    w._on_word = None
    w._url = "scripted"
    w._fps = float(g("sign_sense.fps", 6.0))
    w._timeout = 0.1
    w._stable = int(g("sign_sense.stable_frames", 3))
    w._word_gap = float(g("sign_sense.word_gap_s", 1.2))
    w._min_conf = float(g("sign_sense.min_confidence", 0.55))
    w._echo_sign = True
    w._fail_streak = 0
    w._stop_evt = threading.Event()
    _scripted(w, answers)
    w._thread = threading.Thread(target=w._run, daemon=True)
    w._thread.start()


def _drive(w, seconds):
    time.sleep(seconds)
    w.stop()
    w._thread.join(timeout=2.0)
