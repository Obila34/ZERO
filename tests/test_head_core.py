"""Headless tests for the revived head reflex/pursuit core.

Pure logic — no cv2, no sockets, no hardware. These lock down the properties the
28ce9ee tuning war established (slew-limited = smooth, gate states, gesture
release, the one-look sign test) so a future refactor can't silently regress
them.
"""
import math

from zero.head.controller import HeadController
from zero.head.driver import NullDriver
from zero.head.tracker import FaceTracker


# ── HeadController ──────────────────────────────────────────────────────────

def test_slew_limit_never_exceeded_and_converges():
    """Every tick moves at most max_speed_dps*dt per axis; the head reaches a far
    target monotonically without overshoot (the 'smooth' guarantee)."""
    c = HeadController(lambda x, y: None, rate_hz=25.0, max_speed_dps=90.0, limit_deg=45.0)
    max_step = 90.0 / 25.0
    c.set_target(40.0, -30.0)
    prev_x, prev_y = 0.0, 0.0
    t = 100.0
    reached = False
    for _ in range(200):
        t += 1.0 / 25.0
        should, x, y = c._step(t, "track")
        assert should
        assert abs(x - prev_x) <= max_step + 1e-9
        assert abs(y - prev_y) <= max_step + 1e-9
        # monotonic approach, no overshoot past the target
        assert -1e-9 <= x <= 40.0 + 1e-9
        assert -30.0 - 1e-9 <= y <= 1e-9
        prev_x, prev_y = x, y
        if abs(x - 40.0) < 1e-6 and abs(y + 30.0) < 1e-6:
            reached = True
            break
    assert reached


def test_soft_limit_clamps_target():
    c = HeadController(lambda x, y: None, limit_deg=45.0)
    c.set_target(200.0, -200.0)
    t = 100.0
    for _ in range(400):
        t += 0.04
        _, x, y = c._step(t, "track")
    assert x <= 45.0 + 1e-9 and y >= -45.0 - 1e-9


def test_gate_freeze_and_yield_emit_nothing():
    c = HeadController(lambda x, y: None)
    c.set_target(30.0, 0.0)
    for state in ("freeze", "yield"):
        should, x, y = c._step(100.0, state)
        assert should is False
        assert (x, y) == (0.0, 0.0)  # never moved


def test_gate_home_eases_to_calibrated_home():
    c = HeadController(lambda x, y: None, home_x=5.0, home_y=-3.0, max_speed_dps=90.0)
    c.set_target(40.0, 40.0)  # tracker wants elsewhere...
    t = 100.0
    for _ in range(300):
        t += 0.04
        _, x, y = c._step(t, "home")  # ...but the gate says home
    assert abs(x - 5.0) < 1e-6 and abs(y + 3.0) < 1e-6


def test_gesture_overlay_adds_then_releases():
    c = HeadController(lambda x, y: None)
    c.set_target(0.0, 0.0)
    c.gesture("nod", now=100.0)          # nod = up/down tilt keyframes
    # mid-gesture: tilt offset is non-zero
    dx, dy = c._gesture_offset(100.05)
    assert (dx, dy) != (0.0, 0.0)
    # after the keyframes' total hold, the overlay releases to zero
    dx2, dy2 = c._gesture_offset(100.0 + 10.0)
    assert (dx2, dy2) == (0.0, 0.0)


def test_unknown_gesture_is_ignored():
    c = HeadController(lambda x, y: None)
    c.gesture("not-a-real-gesture", now=100.0)
    assert c._gesture_offset(100.1) == (0.0, 0.0)


# ── NullDriver ──────────────────────────────────────────────────────────────

def test_null_driver_records_last_and_counts():
    d = NullDriver()
    assert d.moves_hardware is False
    d.send(12.0, -4.0)
    d.send(13.5, -4.5)
    assert d.last == (13.5, -4.5)
    assert d.count == 2


def test_controller_pushes_to_driver():
    d = NullDriver()
    c = HeadController(d.send, max_speed_dps=90.0)
    c.set_target(10.0, 0.0)
    t = 100.0
    for _ in range(50):
        t += 0.04
        should, x, y = c._step(t, "track")
        if should:
            d.send(x, y)
    assert abs(d.last[0] - 10.0) < 1e-6


# ── FaceTracker: the one-look sign test ─────────────────────────────────────

class _FakeCtl:
    """Stand-in for HeadController: records aims, reports a settable position."""
    def __init__(self):
        self.aims = []
        self._pos = (0.0, 0.0)

    def set_target(self, x, y):
        self.aims.append((x, y))
        self._pos = (x, y)   # simulate an instantaneous neck for the sign test

    @property
    def position(self):
        return self._pos


def _drive(tracker, face, n=3, w=640, h=480):
    for _ in range(n):
        tracker.update(face, w, h)


def test_face_on_right_turns_right_with_positive_pan_sign():
    ctl = _FakeCtl()
    tr = FaceTracker(ctl, pan_sign=1.0, deadband=0.02)
    # face well to the right of frame centre (cx > w/2)
    _drive(tr, (640 * 0.85, 240, 60, 60))
    assert ctl.aims, "tracker should have issued an aim"
    assert ctl.aims[-1][0] > 0.0   # positive pan = turn right


def test_pan_sign_inversion_flips_direction():
    ctl = _FakeCtl()
    tr = FaceTracker(ctl, pan_sign=-1.0, deadband=0.02)
    _drive(tr, (640 * 0.85, 240, 60, 60))
    assert ctl.aims[-1][0] < 0.0   # flipped: same face now drives the other way


def test_centred_face_latches_and_holds_steady():
    ctl = _FakeCtl()
    tr = FaceTracker(ctl, deadband=0.06)
    # a face dead-centre should settle (latch) and not keep issuing new aims
    _drive(tr, (320, 240, 60, 60), n=5)
    n_before = len(ctl.aims)
    _drive(tr, (320, 240, 60, 60), n=5)
    # once latched, a still centred face produces no further corrective aims
    assert len(ctl.aims) - n_before <= 1
