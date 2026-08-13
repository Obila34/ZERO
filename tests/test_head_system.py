"""HeadSystem composition — headless, NullDriver, fake Eyes. No hardware, no
threads: the pure source/efference methods are driven directly."""
import numpy as np

from zero.head.driver import NullDriver
from zero.head.system import HeadSystem


class FakeCfg:
    def __init__(self, over=None):
        self._o = over or {}

    def get(self, key, default=None):
        return self._o.get(key, default)


class FakeEyes:
    def __init__(self, win=None, shape=(480, 640)):
        self._win = win
        self._shape = shape
        self.suppressed = 0
        self.resettled = 0

    def set_win(self, win):
        self._win = win

    def attention(self):
        return self._win

    def current_frame(self):
        if self._shape is None:
            return None
        return np.zeros((self._shape[0], self._shape[1], 3), np.uint8)

    def suppress_changes(self, duration_s=0.6, until=None):
        self.suppressed += 1

    def resettle(self):
        self.resettled += 1


def _sys(eyes=None, over=None):
    return HeadSystem(FakeCfg(over), eyes=eyes)


def test_defaults_to_null_driver_no_hardware():
    hs = _sys()
    assert isinstance(hs._driver, NullDriver)
    assert hs.status()["moves_hardware"] is False


def test_estop_gate_freezes_then_resumes():
    hs = _sys()
    assert hs._gate() == "track"
    hs.estop()
    assert hs._gate() == "freeze"
    hs.resume()
    assert hs._gate() == "track"


def test_centered_face_holds_the_neck():
    # a face centred in frame → the digital crop covers it, neck stays home
    eyes = FakeEyes(win=(640 / 2 - 40, 480 / 2 - 40, 80, 80))
    hs = _sys(eyes)
    hs.set_state("listening")
    for i in range(10):
        hs._source_tick(1000.0 + i * 0.07)
    ax, ay = hs._last_aim
    assert abs(ax) < 3.0 and abs(ay) < 3.0    # tracker aims ~home; neck holds


def test_offcenter_face_engages_the_neck():
    # a face pushed to the right edge → past the engage threshold → neck turns
    eyes = FakeEyes(win=(640 * 0.85, 240 - 40, 80, 80))
    hs = _sys(eyes)
    hs.set_state("listening")
    for i in range(8):
        hs._source_tick(1000.0 + i * 0.07)
    assert abs(hs._last_aim[0]) > 1.0          # the tracker issued a real pan


def test_thinking_looks_away_open_loop():
    eyes = FakeEyes(win=(640 / 2 - 40, 480 / 2 - 40, 80, 80))
    hs = _sys(eyes)
    hs._last_aim = (0.0, 0.0)
    hs.set_state("thinking")
    hs._source_tick(1000.0)
    # thinking averts up: the target carries a positive tilt offset. Advance one
    # slew step of the (otherwise unspun) controller to see it commit upward.
    _, _x, ty = hs._controller._step(1000.1, "track")
    assert ty > 0.0


def test_efference_copy_suppresses_on_motion_then_resettles():
    eyes = FakeEyes()
    hs = _sys(eyes)
    hs._prev_pos = (0.0, 0.0)
    # simulate the controller having moved
    hs._controller._cur_x = 8.0
    hs._efference_copy(2000.0)
    assert eyes.suppressed >= 1
    assert hs._moved is True
    # now hold still; after the dwell, one resettle fires
    hs._efference_copy(2000.05)               # first still tick — start dwell
    hs._efference_copy(2000.05 + 1.0)         # past resettle_dwell_s
    assert eyes.resettled == 1
    assert hs._moved is False
