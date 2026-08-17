"""Arm subsystem: command parsing, envelope safety, gesture playback, driver
honesty — all headless (NullArmDriver / in-process fake gateway)."""
import time

from zero.arms.commands import SMALL_DEG, parse_arm_command as P
from zero.arms.driver import JointSpec, NullArmDriver, load_joints
from zero.arms.system import ArmSystem


class FakeCfg:
    def __init__(self, over=None):
        self._o = over or {}

    def get(self, key, default=None):
        return self._o.get(key, default)


# Wrists are excluded from the gesture layer, so the fixtures calibrate the
# joints gestures actually use (both steppers -> allow_steppers below).
WRIST = {"right_up_down_joint": {"min": -40, "max": 40, "home": 0},
         "right_elbow_joint": {"min": -35, "max": 35, "home": 0}}


# ── command parsing ─────────────────────────────────────────────────────────

def test_wave_and_sides():
    assert P("wave") == {"kind": "gesture", "name": "wave_right"}
    assert P("wave your left hand")["name"] == "wave_left"
    assert P("can you wave")["name"] == "wave_right"


def test_raise_and_lower():
    # "raise your arm/hand" drives the shoulder-lift joint directly — there is
    # no raise_* gesture, so a joint move is what can actually happen.
    assert P("raise your right arm")["joints"] == ["right_up_down_joint"]
    assert P("lift your left hand")["joints"] == ["left_up_down_joint"]
    # but a limb sent down with no amount means "back to rest"
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
    # a finger is a servo joint: neither a stepper nor excluded
    cfg = {"arms.joints": {"right_bicep_joint": {"min": -10, "max": 10},
                           "right_indexp1_joint": {"min": 0, "max": 60}}}
    joints = load_joints(FakeCfg(cfg))
    assert "right_bicep_joint" not in joints        # gated
    assert "right_indexp1_joint" in joints
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
            "arms.max_speed_dps": 2000.0, "arms.allow_steppers": True,
            "arms.min_gesture_gap_s": 0.0}
    base.update(over or {})
    sys_ = ArmSystem(FakeCfg(base))
    sys_._driver = NullArmDriver()                  # capture, never post
    return sys_


def test_gesture_plays_clamped_and_returns_home():
    s = _system({"arms.gestures": {
        "poke": [{"joints": {"right_elbow_joint": 999}, "s": 0.05},
                 {"joints": {"right_elbow_joint": "home"}, "s": 0.05}]}})
    assert s.play("poke")
    s._player.join(timeout=2.0)
    sent = s._driver.last
    assert sent["right_elbow_joint"] == 0.0         # ended at home
    assert s._pose["right_elbow_joint"] == 0.0
    # and the 999 never survived the envelope
    assert all(v <= 35.0 for v in [s._pose["right_elbow_joint"]])


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
    s._pose["right_elbow_joint"] = 20.0
    assert s.rest()
    s._player.join(timeout=2.0)
    assert s._pose["right_elbow_joint"] == 0.0


def test_uncalibrated_joint_in_gesture_is_skipped():
    s = _system({"arms.gestures": {
        "bad": [{"joints": {"left_elbow_joint": 30,
                            "right_elbow_joint": 10}, "s": 0.05}]}})
    assert s.play("bad")
    s._player.join(timeout=2.0)
    assert "left_elbow_joint" not in s._driver.last  # never commanded
    assert s._driver.last["right_elbow_joint"] == 10.0


def test_preemption_stops_previous_gesture():
    s = _system({"arms.rate_hz": 50.0, "arms.gestures": {
        "slow": [{"joints": {"right_elbow_joint": 30}, "s": 5.0}]}})
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


# ── the intelligence layer: cues, grounding, pacing ─────────────────────────

from zero.arms.cues import find_cues, prompt_block, strip_cues


ARMS = {"right_up_down_joint": {"min": -40, "max": 40, "home": 0},
        "left_up_down_joint": {"min": -40, "max": 40, "home": 0},
        "right_elbow_joint": {"min": -40, "max": 40, "home": 0},
        "left_elbow_joint": {"min": -40, "max": 40, "home": 0}}


def _armsys(over=None):
    base = {"arms.joints": ARMS, "arms.allow_steppers": True,
            "arms.rate_hz": 200.0, "arms.max_speed_dps": 2000.0,
            "arms.min_gesture_gap_s": 0.0}
    base.update(over or {})
    s = ArmSystem(FakeCfg(base))
    s._driver = NullArmDriver()
    return s


def test_cues_are_found_and_stripped_from_speech():
    text = "Yeah [beat] the big one, over there."
    assert find_cues(text) == ["[beat]"]
    # load-bearing: an unstripped cue is SPOKEN as a word by the Piper path
    assert strip_cues(text) == "Yeah the big one, over there."
    assert "[" not in strip_cues("Hi there [wave] good to see you")


def test_excluded_joints_never_load():
    cfg = {"arms.allow_steppers": True, "arms.joints": {
        "right_in_out_joint": {"min": -10, "max": 10},
        "right_wrist_joint": {"min": -10, "max": 10},
        **ARMS}}
    joints = load_joints(FakeCfg(cfg))
    assert "right_in_out_joint" not in joints    # operator-excluded
    assert "right_wrist_joint" not in joints     # and the dead PCA wrists
    assert "right_elbow_joint" in joints         # gesture joints still load


def test_express_plays_the_cued_gesture():
    s = _armsys()
    assert s.express("Hi there [wave] good to see you") == "wave_right"
    s._player.join(timeout=3.0)


def test_no_cue_means_no_movement_when_speech_beats_are_off():
    s = _armsys({"arms.speech_beat": False})
    assert s.express("Just an ordinary sentence with no cue at all") is None


def test_pacing_keeps_most_turns_idle():
    s = _armsys({"arms.min_gesture_gap_s": 30.0})
    assert s.express("First [beat] point", now=1000.0) == "beat_right"
    s._player.join(timeout=2.0)
    # a second gesture inside the floor is dropped, not queued
    assert s.express("Second [beat] point", now=1005.0) is None
    assert s.express("Much later [beat] point", now=1040.0) == "beat_right"
    s._player.join(timeout=2.0)


def test_occupied_hand_is_never_used():
    s = _armsys()
    s.set_hand_state(right="occupied")
    assert s.express("Hi [wave] there") is None
    s.set_hand_state(right="free")
    assert s.express("Hi [wave] there") == "wave_right"
    s._player.join(timeout=3.0)


def test_deictic_refused_until_perception_can_ground_it():
    s = _armsys()
    assert s.express("It's over [point_left] there") is None   # ungrounded
    s.set_pointing_allowed(True)
    assert s.express("It's over [point_left] there") == "point_left"
    s._player.join(timeout=3.0)


def test_suppression_blocks_all_gestures():
    s = _armsys()
    s.suppress(60.0)                       # someone is close to the arms
    assert s.express("Hi [wave] there") is None


def test_only_one_gesture_per_sentence():
    s = _armsys()
    assert s.express("[wave] and also [shrug]") == "wave_right"
    s._player.join(timeout=3.0)


def test_prompt_only_teaches_performable_gestures():
    assert prompt_block([]) == ""                 # nothing calibrated: silent
    block = prompt_block(["beat_right", "wave_right"])
    assert "[beat]" in block and "[wave]" in block
    assert "[point_left]" not in block            # not available -> not taught


def test_driver_subtracts_gateway_offsets_and_waits_for_them():
    """The gateway ADDS its stored offset before converting to steps, so a raw
    0 on a joint offset by -150 commands a 150-degree move. The driver works
    in effective degrees and must refuse to post until it knows the offsets."""
    import json as _json
    import threading as _th
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    posts, offsets_served = [], {"right_elbow_joint": 50.0}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = _json.dumps(offsets_served).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            posts.append(_json.loads(self.rfile.read(n)))
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    _th.Thread(target=srv.serve_forever, daemon=True).start()
    from zero.arms.driver import HttpArmDriver, JointSpec

    d = HttpArmDriver(base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                      joints={"right_elbow_joint": JointSpec(
                          "right_elbow_joint", -40, 40, 0)},
                      max_hz=200.0, deadband_deg=0.1, timeout_s=1.0)
    try:
        d.send({"right_elbow_joint": 10.0})     # effective +10
        t0 = time.monotonic()
        while not posts and time.monotonic() - t0 < 3.0:
            time.sleep(0.01)
        assert posts, "nothing posted"
        # 10 effective - 50 offset = -40 raw, so the gateway re-adds to +10
        assert abs(posts[-1]["angle_deg"] - (-40.0)) < 1e-6
    finally:
        d.close(); srv.shutdown(); srv.server_close()


def test_limit_frac_shrinks_the_envelope_around_home():
    cfg = {"arms.allow_steppers": True, "arms.limit_frac": 0.5,
           "arms.joints": {"right_elbow_joint": {"min": -90.0, "max": 20.0,
                                                 "home": 0.0}}}
    j = load_joints(FakeCfg(cfg))["right_elbow_joint"]
    assert j.min_deg == -45.0 and j.max_deg == 10.0
    assert j.home_deg == 0.0


# ── direct joint control by voice ───────────────────────────────────────────

def test_joint_commands_parse_with_side_direction_and_amount():
    r = P("raise your right elbow")
    assert r["kind"] == "joint" and r["joints"] == ["right_elbow_joint"]
    assert r["degrees"] > 0
    assert P("bend your left elbow 20 degrees")["degrees"] == -20.0
    assert P("lower your arm a bit")["degrees"] == -SMALL_DEG
    assert P("raise both shoulders")["joints"] == ["right_up_down_joint",
                                                   "left_up_down_joint"]
    # ordinary speech still never moves an arm
    for phrase in ["raise the boat by two feet", "open the window",
                   "hand me the cup", "close the door"]:
        assert P(phrase) is None, phrase


def test_move_joint_is_relative_clamped_and_reports_what_moved():
    s = _armsys()
    moved = s.move_joint("right_elbow_joint", 10.0)
    assert moved == ["right_elbow_joint"]
    s._player.join(timeout=2.0)
    assert s.joint_pose()["right_elbow_joint"] == 10.0
    s.move_joint("right_elbow_joint", 10.0)      # relative: adds to the pose
    s._player.join(timeout=2.0)
    assert s.joint_pose()["right_elbow_joint"] == 20.0
    s.move_joint("right_elbow_joint", 9999.0)    # clamped to the envelope
    s._player.join(timeout=2.0)
    assert s.joint_pose()["right_elbow_joint"] == 40.0


def test_uncalibrated_joint_is_reported_not_silently_dropped():
    s = _armsys()
    assert s.move_joint("left_bicep_joint", 10.0) == []   # not in the fixture


def test_joint_sign_flips_direction_without_code_change():
    s = _armsys({"arms.joint_sign": {"right_elbow_joint": -1.0}})
    s.move_joint("right_elbow_joint", 10.0)
    s._player.join(timeout=2.0)
    assert s.joint_pose()["right_elbow_joint"] == -10.0


def test_tool_moves_a_named_joint():
    from zero.tools.arms import ArmTool
    from zero.tools.base import ToolContext

    s = _armsys()
    ctx = ToolContext(extras={"arms": s})
    tool = ArmTool()
    assert "right elbow up" in tool.run({"text": "raise your right elbow"}, ctx)
    s._player.join(timeout=2.0)
    # the LLM's structured form
    assert "elbow down" in tool.run(
        {"part": "elbow", "side": "right", "degrees": -10}, ctx)
    s._player.join(timeout=2.0)
    assert "don't have a" in tool.run({"part": "tail"}, ctx)


def test_speech_beat_moves_the_arms_while_talking():
    """The model rarely writes a cue, so an uncued sentence still gets an
    occasional small beat — otherwise the arms never move during a reply."""
    s = _armsys({"arms.speech_beat": True, "arms.beat_gap_s": 5.0})
    got = s.express("That is a reasonably long spoken sentence", now=1000.0)
    assert got in ("beat_right", "beat_both")
    s._player.join(timeout=2.0)
    # paced: the very next sentence does not also beat
    assert s.express("And another sentence right behind it", now=1001.0) is None
    # but a later one does
    assert s.express("And one more after the gap has passed", now=1010.0)


def test_speech_beat_skips_short_sentences_and_can_be_disabled():
    s = _armsys({"arms.speech_beat": True, "arms.beat_gap_s": 0.0})
    assert s.express("Yeah.", now=1000.0) is None          # too short to beat
    off = _armsys({"arms.speech_beat": False, "arms.beat_gap_s": 0.0})
    assert off.express("A perfectly long sentence here", now=1000.0) is None


def test_speech_beat_still_obeys_estop_and_suppression():
    s = _armsys({"arms.speech_beat": True, "arms.beat_gap_s": 0.0})
    # set the window directly: suppress() stamps the REAL monotonic clock,
    # which the fake `now` below would sail straight past.
    s._suppress_until = 2000.0
    assert s.express("A perfectly long sentence here", now=1000.0) is None
    s._suppress_until = 0.0
    s.estop()
    assert s.express("A perfectly long sentence here", now=1000.0) is None


def test_arm_commands_are_not_stolen_by_the_gaze_parser():
    """2026-08-17: 'rotate right bicep' turned the HEAD, because the gaze
    parser saw a gaze verb plus 'right'. A typo ('righ') was what made it work,
    which is how the hijack was spotted."""
    from zero.head.commands import parse_gaze_command as G

    for phrase in ["rotate right bicep", "rotate left bicep", "raise left arm",
                   "turn your right shoulder up", "lower right elbow"]:
        assert G(phrase) is None, phrase          # gaze declines it
        assert P(phrase) is not None, phrase      # arms take it
    # and real gaze commands are untouched
    assert G("turn right")["axis"] == "pan"
    assert G("look left")["sign"] == -1.0
    assert G("face forward")["kind"] == "center"


def test_bare_part_direction_phrasings():
    """'arms up' / 'left elbow down' — no verb, the way people actually talk.
    These parsed as nothing and fell through to the LLM."""
    assert P("arms up")["joints"] == ["right_up_down_joint",
                                      "left_up_down_joint"]
    assert P("left elbow down")["degrees"] < 0
    assert P("elbow up")["joints"] == ["right_elbow_joint"]
    assert P("both arms up")["side"] == "both"
    # a limb sent down still means rest, however it is phrased
    assert P("arms down")["name"] == "rest"
    assert P("drop left arm")["name"] == "rest"
