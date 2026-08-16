"""Arm subsystem: command parsing, envelope safety, gesture playback, driver
honesty — all headless (NullArmDriver / in-process fake gateway)."""
import time

from zero.arms.commands import parse_arm_command as P
from zero.arms.driver import JointSpec, NullArmDriver, load_joints
from zero.arms.system import ArmSystem


class FakeCfg:
    def __init__(self, over=None):
        self._o = over or {}

    def get(self, key, default=None):
        return self._o.get(key, default)


WRIST = {"right_wrist_joint": {"min": -35, "max": 35, "home": 0}}


# ── command parsing ─────────────────────────────────────────────────────────

def test_wave_and_sides():
    assert P("wave") == {"kind": "gesture", "name": "wave_right"}
    assert P("wave your left hand")["name"] == "wave_left"
    assert P("can you wave")["name"] == "wave_right"


def test_raise_and_lower():
    assert P("raise your right arm")["name"] == "raise_right"
    assert P("lift your left hand")["name"] == "raise_left"
    assert P("put your arms down")["name"] == "rest"
    assert P("lower your arm")["name"] == "rest"
    assert P("arms down")["name"] == "rest"


def test_open_close_hand():
    assert P("open your hand")["name"] == "open_right_hand"
    assert P("close your left hand")["name"] == "close_left_hand"


def test_ordinary_speech_does_not_move_arms():
    for phrase in ["we should wave goodbye to everyone at the door tomorrow",
                   "the tide will raise the boat by two feet",
                   "open the window please", "close the door",
                   "hand me the cup", "put the cup down on the table"]:
        assert P(phrase) is None, phrase


# ── envelope safety ─────────────────────────────────────────────────────────

def test_uncalibrated_joints_are_inert():
    joints = load_joints(FakeCfg({"arms.joints": {}}))
    assert joints == {}


def test_stepper_gated_behind_allow_steppers():
    cfg = {"arms.joints": {"right_bicep_joint": {"min": -10, "max": 10},
                           **WRIST}}
    joints = load_joints(FakeCfg(cfg))
    assert "right_bicep_joint" not in joints        # gated
    assert "right_wrist_joint" in joints
    cfg["arms.allow_steppers"] = True
    joints = load_joints(FakeCfg(cfg))
    assert "right_bicep_joint" in joints


def test_jointspec_clamps_and_home_inside_envelope():
    s = JointSpec("j", -10, 20, 100)                # bogus home gets clamped
    assert s.home_deg == 20
    assert s.clamp(999) == 20 and s.clamp(-999) == -10


# ── system / playback ───────────────────────────────────────────────────────

def _system(over=None):
    base = {"arms.joints": WRIST, "arms.rate_hz": 200.0,
            "arms.max_speed_dps": 2000.0}
    base.update(over or {})
    sys_ = ArmSystem(FakeCfg(base))
    sys_._driver = NullArmDriver()                  # capture, never post
    return sys_


def test_gesture_plays_clamped_and_returns_home():
    s = _system({"arms.gestures": {
        "poke": [{"joints": {"right_wrist_joint": 999}, "s": 0.05},
                 {"joints": {"right_wrist_joint": "home"}, "s": 0.05}]}})
    assert s.play("poke")
    s._player.join(timeout=2.0)
    sent = s._driver.last
    assert sent["right_wrist_joint"] == 0.0         # ended at home
    assert s._pose["right_wrist_joint"] == 0.0
    # and the 999 never survived the envelope
    assert all(v <= 35.0 for v in [s._pose["right_wrist_joint"]])


def test_unknown_gesture_refused_and_estop_blocks():
    s = _system()
    assert not s.play("backflip")
    s.estop()
    assert not s.play("wave_right")
    s.resume()
    assert s.play("wave_right")
    s._player.join(timeout=3.0)


def test_rest_targets_every_calibrated_joint():
    s = _system()
    s._pose["right_wrist_joint"] = 20.0
    assert s.rest()
    s._player.join(timeout=2.0)
    assert s._pose["right_wrist_joint"] == 0.0


def test_uncalibrated_joint_in_gesture_is_skipped():
    s = _system({"arms.gestures": {
        "bad": [{"joints": {"left_elbow_joint": 30,
                            "right_wrist_joint": 10}, "s": 0.05}]}})
    assert s.play("bad")
    s._player.join(timeout=2.0)
    assert "left_elbow_joint" not in s._driver.last  # never commanded
    assert s._driver.last["right_wrist_joint"] == 10.0


def test_preemption_stops_previous_gesture():
    s = _system({"arms.rate_hz": 50.0, "arms.gestures": {
        "slow": [{"joints": {"right_wrist_joint": 30}, "s": 5.0}]}})
    assert s.play("slow")
    time.sleep(0.1)
    t1 = s._player
    assert s.rest()                                  # preempts
    t1.join(timeout=1.0)
    assert not t1.is_alive()


# ── tool dispatch ───────────────────────────────────────────────────────────

def test_arm_tool_paths():
    from zero.tools.arms import ArmTool
    from zero.tools.base import ToolContext

    tool = ArmTool()
    # no arms wired -> honest refusal
    assert "can't move my arms" in tool.run({}, ToolContext())
    s = _system()
    ctx = ToolContext(extras={"arms": s})
    # fast path: raw utterance
    assert tool.run({"text": "wave your hand"}, ctx) == "Okay, waving."
    s._player.join(timeout=3.0)
    # LLM path: named gesture, unknown -> honest refusal about calibration
    assert "isn't calibrated" in tool.run({"gesture": "open_right_hand"}, ctx)
    assert "didn't catch" in tool.run({"text": "tell me a story"}, ctx)


def test_gesture_with_no_calibrated_joints_refused():
    s = _system({"arms.joints": {}})                 # nothing calibrated
    assert not s.play("wave_right")                  # not a silent success
    assert s.rest()                                  # rest is always allowed
