"""Regression tests for Zero._wait_for_wake — the wake-word gate.

These focus on state hygiene, not on running the real pipeline. Zero is
too heavy to fully instantiate in a unit test, so we build a bare object with
object.__new__ and wire in fakes for just the collaborators _wait_for_wake
touches (mic, wake, events, indicator).
"""
from __future__ import annotations

import types

import pytest

from zero.main import Zero
from zero.state import State


class FakeMic:
    def __init__(self, frames):
        self._frames = list(frames)
        self.paused = True     # start PAUSED — mirrors the stop-phrase exit state
        self.resume_calls = 0
        self.drain_calls = 0

    def resume(self):
        self.paused = False
        self.resume_calls += 1

    def pause(self):
        self.paused = True

    def drain(self):
        self.drain_calls += 1

    def frames(self):
        # If the mic is paused, a real MicCapture callback drops frames and
        # frames() would block forever. Emulate that so a missing resume() in
        # _wait_for_wake would hang the test.
        for f in self._frames:
            if self.paused:
                # A real callback would just not enqueue — never yield.
                raise AssertionError("frames() yielded while paused — real code would hang")
            yield f


class FakeWake:
    def __init__(self, fire_on=None):
        self.fire_on = fire_on
        self.calls = 0
        self.resets = 0

    def process(self, frame):
        self.calls += 1
        return frame == self.fire_on

    def reset(self):
        self.resets += 1


class FakeEvents:
    def __init__(self):
        self.pending = False

    def peek_pending(self):
        return self.pending


def _bare_zero(mic, wake, events):
    """A Zero with just the attributes _wait_for_wake needs. No engines loaded."""
    z = object.__new__(Zero)
    z.mic = mic
    z.wake = wake
    z.events = events
    z.indicator = None
    z.state = State.SPEAKING   # anything but IDLE — verifies the transition to IDLE
    return z


def test_wait_for_wake_resumes_paused_mic():
    """After a stop phrase, mic is left paused. _wait_for_wake must resume it
    before iterating frames, or the wake word can never fire again."""
    mic = FakeMic(frames=["frame1", "WAKE"])
    wake = FakeWake(fire_on="WAKE")
    events = FakeEvents()
    z = _bare_zero(mic, wake, events)

    z._wait_for_wake()

    assert mic.resume_calls >= 1, "mic must be resumed at entry"
    assert mic.drain_calls >= 1, "stale queue must be drained"
    assert wake.calls == 2       # both frames processed
    assert z.state == State.IDLE


def test_wait_for_wake_resets_wake_engine():
    mic = FakeMic(frames=["WAKE"])
    wake = FakeWake(fire_on="WAKE")
    z = _bare_zero(mic, wake, FakeEvents())

    z._wait_for_wake()

    assert wake.resets >= 1, "wake engine must be reset so state is clean"


def test_wait_for_wake_returns_on_first_wake_hit():
    mic = FakeMic(frames=["a", "b", "WAKE", "c"])
    wake = FakeWake(fire_on="WAKE")
    z = _bare_zero(mic, wake, FakeEvents())

    z._wait_for_wake()

    # Stopped exactly at the WAKE frame; the trailing "c" is never inspected.
    assert wake.calls == 3
