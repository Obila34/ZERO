"""FaceFramer: recentering, clamping, freeze-on-miss, primary stickiness.

Detection is faked (scripted boxes); the tracker inside the framer is the real
IouTracker, so confirm/lost timing is exercised for real. YuNet itself is
covered by test_face_align.py / scripts/smoke_face.py.
"""
from __future__ import annotations

import numpy as np

from zero.vision.framer import FaceFramer

W, H = 640, 480
FRAME = np.zeros((H, W, 3), dtype=np.uint8)


def _pts(cx: float, cy: float) -> np.ndarray:
    """Five landmark points whose centroid is exactly (cx, cy)."""
    off = np.float32([[-10, -5], [10, -5], [0, 0], [-8, 10], [8, 10]])
    return np.float32([cx, cy]) + (off - off.mean(axis=0))


class ScriptedDetector:
    """detect_faces() returns whatever the test put in ``faces``."""

    def __init__(self):
        self.faces: list = []

    def detect_faces(self, _frame):
        return list(self.faces)


class BrokenDetector:
    def detect_faces(self, _frame):
        raise RuntimeError("camera gremlins")


def _framer(det, frac=0.5):
    return FaceFramer(det, window_frac=frac, confirm_s=0.4, lost_s=1.0)


def test_initial_window_rests_at_center():
    fr = _framer(ScriptedDetector())
    win = fr.update(FRAME, now=0.0)
    assert win == (160, 120, 320, 240)  # centered 50% window in 640x480


def test_recenters_on_confirmed_face_and_clamps():
    det = ScriptedDetector()
    fr = _framer(det)
    # Face near the top-left corner; landmark centroid at (125, 125).
    det.faces = [((100, 100, 50, 50), _pts(125, 125))]
    fr.update(FRAME, now=0.0)                   # tentative — no lock yet
    assert fr.window == (160, 120, 320, 240)    # still at rest
    win = fr.update(FRAME, now=0.5)             # past confirm_s — locked on
    # Ideal x = 125-160 = -35 -> clamped to 0; y = 125-120 = 5.
    assert win == (0, 5, 320, 240)
    # Face WALKS right (overlapping steps — a teleport would rightly be
    # treated as a different face); window follows every step.
    t = 0.5
    for i in range(1, 16):
        t += 0.1
        x, y = 100 + i * 20, 100 + i * 20 // 3
        det.faces = [((x, y, 50, 50), _pts(x + 25, y + 25))]
        win = fr.update(FRAME, now=t)
    assert win == (265, 105, 320, 240)  # centered on final centroid (425, 225)


def test_freezes_when_face_disappears():
    det = ScriptedDetector()
    fr = _framer(det)
    det.faces = [((300, 200, 60, 60), _pts(330, 230))]
    fr.update(FRAME, now=0.0)
    locked = fr.update(FRAME, now=0.5)
    det.faces = []                               # face gone
    assert fr.update(FRAME, now=0.6) == locked   # coasting inside lost_s
    assert fr.update(FRAME, now=5.0) == locked   # long gone — frozen in place
    assert fr.update(FRAME, now=9.0) == locked


def test_primary_face_is_sticky_until_lost():
    det = ScriptedDetector()
    fr = _framer(det)
    a = ((80, 200, 50, 50), _pts(105, 225))          # left face, first
    det.faces = [a]
    fr.update(FRAME, now=0.0)
    fr.update(FRAME, now=0.5)
    b = ((500, 200, 90, 90), _pts(545, 245))         # bigger face, right
    det.faces = [a, b]
    fr.update(FRAME, now=1.0)
    win = fr.update(FRAME, now=1.5)                  # b confirmed by now
    assert win[0] == 0                               # still framing a (left)
    det.faces = [b]                                  # a leaves
    fr.update(FRAME, now=2.0)
    win = fr.update(FRAME, now=3.0)                  # a expired (lost_s=1.0)
    assert win[0] == 320                             # gaze moved to b (clamped)


def test_broken_detector_never_raises_and_holds_window():
    fr = _framer(BrokenDetector())
    # The window is placed (at rest, centered) BEFORE detection runs, so even
    # a broken detector leaves ZERO with a usable gaze window.
    assert fr.update(FRAME, now=0.0) == (160, 120, 320, 240)
    fr2 = _framer(ScriptedDetector())
    first = fr2.update(FRAME, now=0.0)
    fr2._detector = BrokenDetector()
    assert fr2.update(FRAME, now=1.0) == first    # holds, no exception


def test_crop_matches_window():
    det = ScriptedDetector()
    fr = _framer(det)
    frame = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)
    fr.update(frame, now=0.0)
    x, y, w, h = fr.window
    crop = fr.crop(frame)
    assert crop.shape == (h, w, 3)
    assert np.shares_memory(crop, frame)          # a view, not a copy
