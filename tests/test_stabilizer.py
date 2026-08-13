"""Stabilizer (efference copy + IMU-ready) and the Eyes ego-motion gate."""
import threading
import time

from zero.head.stabilizer import Stabilizer
from zero.vision.eyes import Eyes


def test_efference_copy_default_from_commanded():
    s = Stabilizer(moving_eps_dps=3.0, saccade_dps=15.0)
    assert not s.has_imu
    s.note_commanded(0.0, 0.0)
    assert not s.is_moving()
    s.note_commanded(8.0, 0.0)
    assert s.is_moving() and not s.is_saccading()
    s.note_commanded(30.0, 0.0)
    assert s.is_saccading()
    assert s.angular_rate() == (30.0, 0.0)


def test_imu_preferred_when_attached():
    s = Stabilizer()
    s.note_commanded(5.0, 5.0)
    s.attach_imu(lambda: (40.0, -2.0))     # a real gyro sees fast motion
    assert s.has_imu
    assert s.angular_rate() == (40.0, -2.0)
    assert s.is_saccading()


def test_imu_reader_failure_falls_back_to_efference():
    s = Stabilizer()
    s.note_commanded(9.0, 0.0)
    def broken():
        raise RuntimeError("gyro offline")
    s.attach_imu(broken)
    assert s.angular_rate() == (9.0, 0.0)   # falls back, never raises


def _bare_eyes():
    e = Eyes.__new__(Eyes)
    e._ego_rate = (0.0, 0.0)
    e._saccade_dps = 15.0
    return e


def test_eyes_ego_motion_flag():
    e = _bare_eyes()
    assert not e.is_ego_moving()
    e.set_ego_motion(20.0, 0.0)
    assert e.is_ego_moving()                 # a saccade → drop the blurred frame
    e.set_ego_motion(5.0, 0.0)
    assert not e.is_ego_moving()             # slow pursuit → keep detecting
