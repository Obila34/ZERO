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
    hs._scheduler.set_state("listening", 999.0)
    for i in range(10):
        hs._source_tick(1000.0 + i * 0.07)
    ax, ay = hs._last_aim
    assert abs(ax) < 3.0 and abs(ay) < 3.0    # tracker aims ~home; neck holds


def test_offcenter_face_engages_the_neck():
    # a face pushed to the right edge → past the engage threshold → neck turns
    eyes = FakeEyes(win=(640 * 0.85, 240 - 40, 80, 80))
    hs = _sys(eyes)
    hs._scheduler.set_state("listening", 999.0)
    for i in range(8):
        hs._source_tick(1000.0 + i * 0.07)
    assert abs(hs._last_aim[0]) > 1.0          # the tracker issued a real pan


def test_thinking_looks_away_open_loop():
    eyes = FakeEyes(win=(640 / 2 - 40, 480 / 2 - 40, 80, 80))
    hs = _sys(eyes)
    hs._last_aim = (0.0, 0.0)
    hs._scheduler.set_state("thinking", 999.0)
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


def test_tilt_envelope_from_config_reaches_controller():
    # audit H5: the calibrated tilt window must exist in BELIEF space too
    hs = _sys(over={"head.tilt_min_deg": -10.0, "head.tilt_max_deg": 25.0})
    assert hs._controller.clamp_to_envelope(0.0, 90.0) == (0.0, 25.0)
    assert hs._controller.clamp_to_envelope(0.0, -90.0) == (0.0, -10.0)
    # pan window untouched
    assert hs._controller.clamp_to_envelope(120.0, 0.0)[0] == 45.0


def test_command_sentinel_is_clamped_into_the_envelope():
    # audit M3: FULL_DEG=999 must never leak into _cmd_target
    hs = _sys(over={"head.tilt_min_deg": -10.0, "head.tilt_max_deg": 25.0})
    hs.look_direction("tilt", 999.0)
    assert hs._cmd_target == (0.0, 25.0)
    hs.look_direction("pan", -999.0)
    assert hs._cmd_target[0] == -45.0     # default limit_deg


def test_command_deadline_is_set_when_target_appears():
    # audit M2: the deadline is written BEFORE the target so a source tick
    # can never see a fresh target with an expired deadline
    import time as _t
    hs = _sys()
    hs.look_direction("pan", 20.0)
    assert hs._cmd_target is not None
    assert hs._cmd_until > _t.monotonic()


def test_hand_tick_vertical_channel_gated_and_clamped():
    class FakeHand:
        value = (0.5, 1.0, 0.9)          # x, y, conf

        def reset_filters(self):
            pass

    hs = _sys(over={"head.tilt_min_deg": -10.0, "head.tilt_max_deg": 25.0})
    hs._hand = FakeHand()
    hs._hand_tilt = False                 # kill switch (default): tilt stays 0
    hs._hand_tick(now=0.0)
    assert hs._last_aim[1] == 0.0
    hs._hand_tilt = True                  # enabled: y maps into the envelope
    hs._hand_tilt_deg = 15.0
    hs._hand_tick(now=0.0)
    assert hs._last_aim[1] == 15.0
    hs._hand = FakeHand()
    hs._hand.value = (0.0, 100.0, 0.9)    # absurd y still clamped to the window
    hs._hand_tick(now=0.0)
    assert hs._last_aim[1] == 25.0


def test_stale_face_window_is_not_chased():
    """A frozen attention window must not read as a live target (2026-08-17:
    the head wound to its ±80° limit chasing a face that had gone)."""
    eyes = FakeEyes(win=(600, 400, 80, 80))     # far off-centre -> would engage
    hs = _sys(eyes, over={"head.face_stale_s": 0.7})
    hs._scheduler.set_state("listening", 999.0)

    eyes.attention_age = lambda: 0.1            # face seen 100 ms ago = live
    for i in range(6):
        hs._source_tick(1000.0 + i * 0.07)
    assert hs._dbg["branch"] == "engage"
    live_aim = hs._last_aim[0]
    assert abs(live_aim) > 1.0                  # it did try to follow

    eyes.attention_age = lambda: 3.0            # no detection for 3 s = ghost
    for i in range(6):
        hs._source_tick(1001.0 + i * 0.07)
    assert hs._dbg["branch"] == "stale-face"    # stops chasing


def test_missing_attention_age_falls_back_to_live():
    """Eyes without the freshness signal must still track (old behaviour)."""
    eyes = FakeEyes(win=(600, 400, 80, 80))
    hs = _sys(eyes)
    hs._scheduler.set_state("listening", 999.0)
    for i in range(6):
        hs._source_tick(1000.0 + i * 0.07)
    assert hs._dbg["branch"] == "engage"


class FakeMirror:
    """Stands in for the head-yaw pose source: (x, y, conf)."""
    def __init__(self, x=0.0, conf=1.0):
        self.value = (x, 0.0, conf)

    def set(self, x, conf=1.0):
        self.value = (x, 0.0, conf)

    def reset_filters(self):
        pass


def _mirror_sys(**over):
    base = {"head.mirror.deadzone": 0.2, "head.mirror.range_deg": 80.0,
            "head.mirror.hold_s": 2.5}
    base.update(over)
    hs = _sys(FakeEyes(win=(300, 200, 80, 80)), over=base)
    hs._mirror = FakeMirror()
    return hs


def test_small_head_movements_are_ignored():
    hs = _mirror_sys()
    hs._mirror.set(0.15)                       # inside the deadzone
    assert hs._mirror_target(1000.0) is None


def test_big_turn_maps_proportionally_across_the_range():
    hs = _mirror_sys()
    hs._mirror.set(1.0)                        # full turn -> full range
    assert hs._mirror_target(1000.0) == 80.0
    hs._mirror.set(-1.0)                       # and symmetric the other way
    assert hs._mirror_target(1000.0) == -80.0
    # just past the deadzone starts from ~0, not a step to a big angle
    hs._mirror.set(0.21)
    assert 0.0 < hs._mirror_target(1000.0) < 5.0
    # and it is monotonic: further turn, bigger angle
    hs._mirror.set(0.6)
    mid = hs._mirror_target(1000.0)
    hs._mirror.set(0.8)
    assert hs._mirror_target(1000.0) > mid


def test_turn_is_held_when_you_leave_the_frame_then_released():
    hs = _mirror_sys()
    hs._mirror.set(1.0)
    assert hs._mirror_target(1000.0) == 80.0
    hs._mirror.set(0.0, conf=0.0)              # turned away -> signal lost
    assert hs._mirror_target(1001.0) == 80.0   # held, not snapped back
    assert hs._mirror_target(1004.0) is None   # past hold_s -> face tracking


def test_facing_forward_hands_the_neck_back_to_attention():
    hs = _mirror_sys()
    hs._mirror.set(1.0)
    hs._mirror_target(1000.0)
    hs._mirror.set(0.0)                        # squared up again
    assert hs._mirror_target(1000.5) is None
    # and the source tick then runs the face-tracking branch
    hs._scheduler.set_state("listening", 999.0)
    hs._source_tick(1000.6)
    assert hs._dbg["branch"] != "mirror"


def test_mirror_drives_the_neck_and_respects_the_envelope():
    hs = _mirror_sys()
    hs._mirror.set(1.0)
    hs._scheduler.set_state("listening", 999.0)
    hs._source_tick(1000.0)
    assert hs._dbg["branch"] == "mirror"
    assert hs._last_aim[0] == 45.0             # clamped to default limit_deg
