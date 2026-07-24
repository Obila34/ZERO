"""World state (Phase 2): versioned snapshots, motion gate, narrator, wiring."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from zero.world.state import WorldEvent, WorldObject, WorldState


def _obj(tid=1, label="mug", conf=0.9, ts=None):
    ts = time.time() if ts is None else ts
    return WorldObject(track_id=tid, label=label, bbox=(1, 2, 3, 4),
                       confidence=conf, first_seen=ts, last_seen=ts)


# ── WorldState core ──────────────────────────────────────────────────────────
def test_versions_are_monotonic_and_snapshots_immutable():
    w = WorldState()
    v0 = w.snapshot().version
    w.update_motion(0.5, True)
    w.update_objects([_obj()], [WorldEvent("appeared", "mug", time.time())])
    w.update_narration("a mug sits on the desk")
    snap = w.snapshot()
    assert snap.version == v0 + 3
    assert snap.motion_active and snap.motion_level == 0.5
    assert snap.objects[0].label == "mug"
    assert snap.narration == "a mug sits on the desk"
    with pytest.raises(Exception):        # frozen dataclass
        snap.narration = "hacked"


def test_old_snapshot_stays_consistent_after_updates():
    w = WorldState()
    w.update_objects([_obj(label="cup")])
    old = w.snapshot()
    w.update_objects([_obj(label="laptop")])
    assert old.objects[0].label == "cup"          # reader's view never mutates
    assert w.snapshot().objects[0].label == "laptop"


def test_events_are_capped():
    w = WorldState()
    for i in range(50):
        w.update_objects([], [WorldEvent("appeared", f"obj{i}", time.time())])
    assert len(w.snapshot().events) == 16
    assert w.snapshot().events[-1].label == "obj49"


def test_wait_for_change_unblocks_on_write():
    w = WorldState()
    v = w.version
    got = {}

    def reader():
        got["snap"] = w.wait_for_change(v, timeout=5.0)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    w.update_motion(0.9, True)
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert got["snap"].version > v


def test_wait_for_change_times_out_quietly():
    w = WorldState()
    t0 = time.perf_counter()
    snap = w.wait_for_change(w.version, timeout=0.05)
    assert time.perf_counter() - t0 < 1.0
    assert snap.version == w.version


def test_describe_composes_objects_narration_events():
    w = WorldState()
    now = time.time()
    w.update_objects(
        [_obj(1, "person"), _obj(2, "person"), _obj(3, "mug")],
        [WorldEvent("appeared", "mug", now)])
    w.update_narration("David is typing at the desk")
    text = w.describe()
    assert "2 people" in text
    assert "a mug" in text
    assert "David is typing" in text
    assert "the mug came into view" in text


def test_describe_drops_stale_narration():
    w = WorldState()
    w.update_narration("old news", ts=time.time() - 300)
    assert "old news" not in w.describe(narration_max_age_s=20.0)


# ── Tier 0: motion + gate ────────────────────────────────────────────────────
def test_motion_detector_static_vs_moving():
    pytest.importorskip("cv2")
    from zero.vision.motion import MotionDetector

    det = MotionDetector(threshold=0.02, quiet_frames=3)
    still = np.full((120, 160, 3), 80, dtype=np.uint8)
    level, active = det.update(still)          # first frame: no baseline
    for _ in range(5):
        level, active = det.update(still)
    assert level == 0.0 and not active

    moved = still.copy()
    moved[20:100, 20:100] = 200                # a big bright thing appears
    level, active = det.update(moved)
    assert level > 0.02 and active

    # Hysteresis: stays active for quiet_frames of stillness, then drops.
    for _ in range(2):
        _, active = det.update(moved)          # unchanged now -> quiet frames
    assert active
    _, active = det.update(moved)
    assert not active


def test_detection_gate_full_rate_on_motion_keepalive_when_idle():
    from zero.vision.motion import DetectionGate

    g = DetectionGate(active_interval_s=0.0, idle_interval_s=2.0, linger_s=0.5)
    t = 100.0
    assert g.should_detect(True, t)            # moving: every frame
    assert g.should_detect(True, t + 0.03)
    assert g.should_detect(False, t + 0.4)     # linger window: still full rate
    assert not g.should_detect(False, t + 1.0)   # idle: gated
    assert not g.should_detect(False, t + 2.0)
    assert g.should_detect(False, t + 2.5)     # keepalive after idle_interval
    assert g.should_detect(True, t + 2.6)      # motion reopens instantly


# ── Tier 2: narrator ─────────────────────────────────────────────────────────
class FakeBackend:
    def __init__(self, text="two people talking", fail=False):
        self.text = text
        self.fail = fail
        self.calls = 0

    def narrate(self, frame, hint):
        self.calls += 1
        if self.fail:
            raise RuntimeError("backend down")
        return self.text


def _wait_until(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_narrator_writes_into_world():
    from zero.world.narrator import Narrator

    w = WorldState()
    w.update_motion(0.5, True)                 # scene is "moving" -> no skip
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    n = Narrator(w, lambda: frame, FakeBackend(), interval_s=0.01)
    n.start()
    try:
        assert _wait_until(lambda: w.snapshot().narration != "")
        assert w.snapshot().narration == "two people talking"
        assert n.last_latency_s >= 0.0
    finally:
        n.stop()


def test_narrator_survives_backend_failure_and_backs_off():
    from zero.world.narrator import Narrator

    w = WorldState()
    w.update_motion(0.5, True)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    backend = FakeBackend(fail=True)
    n = Narrator(w, lambda: frame, backend, interval_s=0.01)
    n.start()
    try:
        assert _wait_until(lambda: n.failures >= 1)
        first = backend.calls
        time.sleep(0.1)                        # backoff (>=4s) blocks retries
        assert backend.calls == first
        assert w.snapshot().narration == ""
    finally:
        n.stop()


def test_narrator_skips_without_frame():
    from zero.world.narrator import Narrator

    w = WorldState()
    w.update_motion(0.5, True)
    backend = FakeBackend()
    n = Narrator(w, lambda: None, backend, interval_s=0.01)
    n.start()
    try:
        assert _wait_until(lambda: n.skips >= 3)
        assert backend.calls == 0
    finally:
        n.stop()


# ── Tier 1 -> world wiring through Eyes ──────────────────────────────────────
def test_eyes_publishes_tracks_and_events_to_world():
    pytest.importorskip("cv2")
    from zero.vision.eyes import Eyes
    from zero.vision.schemas import Detection

    w = WorldState()
    eyes = Eyes(camera=None, detector=None, color_namer=None, client=None,
                world=w)
    eyes._started_at = time.time() - 100       # past the settle window

    det = Detection(label="mug", bbox=[10, 10, 40, 40], confidence=0.8,
                    color=None)
    person = Detection(label="person", bbox=[100, 10, 80, 200],
                       confidence=0.9, color=None)
    t = time.time()
    eyes._track_instances([det, person], t)          # tentative tracks
    eyes._track_instances([det, person], t + 1.2)    # confirmed (confirm_s=1)

    snap = w.snapshot()
    labels = sorted(o.label for o in snap.objects)
    assert labels == ["mug", "person"]
    assert all(o.track_id > 0 for o in snap.objects)
    assert {e.kind for e in snap.events} == {"appeared"}
    assert len(snap.people()) == 1
    # Person "moved" events never become spoken remarks.
    moved_person = Detection(label="person", bbox=[400, 10, 80, 200],
                             confidence=0.9, color=None)
    eyes._track_instances([det, moved_person], t + 40.0)
    assert not any("person" in c for c in eyes._changes)
