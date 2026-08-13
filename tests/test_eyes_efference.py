"""Efference-copy suppression: commanded head motion must not manufacture
scene-change novelty. Built on a bare Eyes (no camera/cv2) exercising just the
diffing + suppression surface."""
import threading
import time

from zero.vision.eyes import Eyes


class _Det:
    def __init__(self, label):
        self.label = label
        self.bbox = (0, 0, 10, 10)


class _Snap:
    def __init__(self, dets):
        self.detections = dets


class _Scene:
    def __init__(self):
        self._dets = []

    def set(self, labels):
        self._dets = [_Det(l) for l in labels]

    def snapshot(self):
        return _Snap(list(self._dets))


def _bare_eyes():
    e = Eyes.__new__(Eyes)
    e._change_lock = threading.Lock()
    e._changes = []
    e._last_change_note = 0.0
    e._change_cooldown_s = 120.0
    e._started_at = time.time() - 100.0   # well past the startup settle grace
    e._first_seen = {}
    e._last_seen = {}
    e._stable = {"person"}
    e._suppress_until = 0.0
    e._scene = _Scene()
    e._motion = None
    e._world = None
    return e


def test_novelty_fires_normally():
    e = _bare_eyes()
    now = time.time()
    e._first_seen["cup"] = now - 3.0          # already present long enough
    e._track_changes([_Det("person"), _Det("cup")])
    assert e.scene_changes() == ["a cup just came into view"]


def test_commanded_motion_suppresses_novelty():
    e = _bare_eyes()
    now = time.time()
    e._first_seen["cup"] = now - 3.0
    e.suppress_changes(1.0)                    # the neck is moving
    e._track_changes([_Det("person"), _Det("cup")])
    assert e.scene_changes() == []            # no false "a cup came into view"
    assert "cup" in e._stable                 # but the baseline absorbed it


def test_resettle_reseeds_baseline_and_drops_queued():
    e = _bare_eyes()
    now = time.time()
    # queue a stale candidate as if it slipped in just before the move
    e._changes = ["a laptop just came into view"]
    e._scene.set(["person", "bookshelf"])     # the view the head turned to
    e.resettle()
    assert e._changes == []                    # queued candidate dropped
    assert e._stable == {"person", "bookshelf"}
    # the newly-faced objects are the baseline, so they are NOT announced as new
    e._track_changes([_Det("person"), _Det("bookshelf")])
    assert e.scene_changes() == []


def test_suppress_extends_never_shrinks():
    e = _bare_eyes()
    now = time.time()
    e.suppress_changes(until=now + 5.0)
    e.suppress_changes(0.1)                    # a shorter window must not win
    assert e._suppress_until >= now + 5.0 - 1e-6
    assert e._suppressed(now)
    assert not e._suppressed(now + 6.0)
