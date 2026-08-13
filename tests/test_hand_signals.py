"""Pure-signal tests for HandPoseSource — the junk-rejection guard and the
body-relative x/y math (audit L3: this module had zero tests). Objects are
built via __new__ so no ONNX model or onnxruntime is needed."""
import numpy as np

from zero.head.hand import (HandPoseSource, OneEuro, L_EAR, R_EAR, L_EYE,
                            R_EYE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST,
                            NOSE)


def _src(keypoint="wrist"):
    s = HandPoseSource.__new__(HandPoseSource)
    s._kp_conf = 0.3
    s._min_shoulder_px = 32.0     # 0.10 * 320
    s._sz = 320
    s._keypoint = keypoint
    s._deadzone_y = 0.15
    s._gain_y = 1.0
    s._filt_y = OneEuro()
    s._y0_samples = []
    s._y0 = None
    s._y = 0.0
    return s


def _kps():
    k = np.zeros((17, 3), np.float32)
    return k


def test_hand_offset_x_and_y_signs():
    s = _src()
    k = _kps()
    k[L_SHOULDER] = (100, 200, 0.9)
    k[R_SHOULDER] = (220, 200, 0.9)     # width 120, midline x=160, line y=200
    k[L_WRIST] = (250, 140, 0.9)        # right of midline, above the shoulders
    x, y = s._signal(k)
    assert x == (250 - 160) / 120       # +0.75: right is positive
    assert y == (200 - 140) / 120       # +0.5: up is positive


def test_degenerate_shoulders_rejected():
    s = _src()
    k = _kps()
    k[L_SHOULDER] = (3, 3, 0.9)
    k[R_SHOULDER] = (9, 3, 0.9)         # 6 px "person" = model hallucination
    k[L_WRIST] = (6, 1, 0.9)
    assert s._signal(k) is None


def test_head_pose_yaw_and_pitch():
    s = _src("head")
    k = _kps()
    k[L_EAR] = (100, 100, 0.9)
    k[R_EAR] = (180, 100, 0.9)          # span 80, midline x=140, line y=100
    k[NOSE] = (160, 130, 0.9)           # right of midline, below the ear line
    x, y = s._signal(k)
    assert x == (160 - 140) / 80        # yaw right positive
    assert y == (100 - 130) / 80        # below the line -> negative (up positive)


def test_head_pose_falls_back_to_eyes():
    s = _src("head")
    k = _kps()
    k[NOSE] = (160, 130, 0.9)
    k[L_EYE] = (140, 100, 0.9)
    k[R_EYE] = (180, 100, 0.9)          # ears missing; eye span 40 (>= 0.02*320)
    x, y = s._signal(k)
    assert x == (160 - 160) / 40
    assert y == (100 - 130) / 40


def test_vertical_neutral_then_deviation():
    s = _src()
    # first 10 valid samples define neutral; output stays 0 while learning
    for _ in range(10):
        assert s._vertical(0.30, None) == 0.0
    assert s._y0 == 0.30
    # small wiggle inside the deadzone -> 0
    assert s._vertical(0.40, None) == 0.0
    # a real move above the deadzone -> positive, bounded
    out = s._vertical(0.80, None)
    assert 0.0 < out <= 1.0
