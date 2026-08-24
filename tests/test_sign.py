"""Sign engine + hand model: asymmetry safety, spelling, honesty, lexicon
validation — all against NullTransport (nothing moves)."""
import time

from zero.arms import hands
from zero.motion.bus import BusJoint, MotionBus
from zero.motion.transport import NullTransport
from zero.sign.engine import SignEngine, sign_prompt_block
from zero.sign.handshapes import HANDSHAPES, capabilities


class FakeCfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


_BUSES = []


def teardown_module():
    # Park no leaked bus threads into pytest's exit-time GC (mediapipe's
    # FaceLandmarker teardown is already flaky there — keep our side clean).
    for b in _BUSES:
        b.close()
    _BUSES.clear()


def _engine(over=None):
    t = NullTransport()
    bus = MotionBus(t, rate_hz=500.0)
    _BUSES.append(bus)
    for name, s in hands.hand_joint_specs().items():
        bus.register(BusJoint(name, min_deg=s["min"], max_deg=s["max"],
                              home_deg=s["home"], batch=True))
    cfg = FakeCfg({"sign.letter_s": 0.06, "sign.transition_s": 0.03,
                   "sign.single_hold_s": 0.05, "sign.rate_hz": 500,
                   "sign.max_speed_dps": 100000.0,
                   "sign.lexicon_path": "/nonexistent.yaml"})
    cfg.update(over or {})
    return SignEngine(cfg, bus), bus, t


def _wait_released(bus, timeout=5.0):
    # Wait for the sign playback to CLAIM the hands and then RELEASE them.
    # (Just checking for release races the thread startup: right after
    # spell() the claim may not exist yet and release looks instant.)
    end = time.monotonic() + timeout
    seen = False
    while time.monotonic() < end:
        owning = (bus.owner("left_indexp1_joint") == "sign" or
                  bus.owner("right_indexp1_joint") == "sign")
        if owning:
            seen = True
        elif seen:
            return True
        time.sleep(0.005)
    return False


# ── the hand model: per-side truth ──────────────────────────────────────────

def test_hands_are_asymmetric_and_closure_is_portable():
    # The bug that motivated all of this: a closed thumb is 140 on the left
    # but only 70 on the right. Closure space hides that from the sign.
    assert hands.finger_deg("left", "thumb", 1.0) == 140.0
    assert hands.finger_deg("right", "thumb", 1.0) == 70.0
    # ...and open differs too (right pinky opens at 83, left at 99)
    assert hands.finger_deg("left", "pinky", 0.0) == 99.0
    assert hands.finger_deg("right", "pinky", 0.0) == 83.0
    # wrists mount mirrored: palm-forward is 180 left, 0 right
    assert hands.wrist_deg("left", "forward") == 180.0
    assert hands.wrist_deg("right", "forward") == 0.0


def test_hand_joint_specs_match_firmware_travel():
    specs = hands.hand_joint_specs()
    assert specs["right_thumbp1_joint"]["max"] == 70.0    # NOT 180
    assert specs["left_thumbp1_joint"]["max"] == 140.0
    assert specs["left_indexp1_joint"]["max"] == 90.0
    assert set(specs) == set(hands.HAND_JOINTS)
    assert len(specs) == 12


# ── the alphabet ────────────────────────────────────────────────────────────

def test_all_26_letters_defined_and_ambiguities_documented():
    assert len(HANDSHAPES) == 26
    caps = capabilities()
    # the five mechanical compromises, and ONLY those, are flagged
    assert caps["approx"] == ["P", "Q", "R", "V", "Z"]
    # the flat-table collisions are resolved for the pairs that CAN differ:
    # L vs Q by orientation, K vs P documented (P needs the dark forearm)
    assert HANDSHAPES["L"]["orient"] != HANDSHAPES["Q"]["orient"]


def test_spelling_speaks_paces_and_releases():
    eng, bus, t = _engine()
    spoken = eng.spell("COW")
    assert spoken == "Spelling COW: C - O - W."
    assert bus.owner("left_indexp1_joint") == "sign"     # signing claims
    assert _wait_released(bus)
    # eased back to open at the end
    time.sleep(0.05)
    assert abs(t.posted["left_indexp1_joint"] - 90.0) < 2.0


def test_right_thumb_never_exceeds_its_stop_during_a_spell():
    eng, bus, t = _engine()
    eng.spell("SAM")                       # S = closed fist, thumb 1.0
    assert _wait_released(bus)
    posts = [p["right_thumbp1_joint"] for p in t.posts
             if "right_thumbp1_joint" in p]
    assert posts and max(posts) <= 70.01


def test_sign_outranks_a_gesture_on_the_same_joints():
    eng, bus, t = _engine()
    bus.write("gesture", {"left_indexp1_joint": 33.0})   # a speech beat
    eng.spell("A")                          # A closes the index (deg 0)
    assert _wait_released(bus)
    idx = [p["left_indexp1_joint"] for p in t.posts
           if "left_indexp1_joint" in p]
    # the sign took the joint and eased it to closed (within the bus's
    # deadband of exactly 0); the beat's 33 never re-lands mid-sign
    assert min(idx) < 0.5
    closed_at = idx.index(min(idx))
    assert 33.0 not in idx[closed_at:]


def test_unknown_sign_refused_lexicon_empty():
    eng, _bus, _t = _engine()
    assert eng.sign("hello") is None
    assert eng.sign_names() == []


def test_letter_refusals_and_estop():
    eng, bus, _t = _engine()
    assert eng.letter("?") is None
    bus.estop()
    assert eng.spell("HI") is None
    assert eng.letter("A") is None
    bus.resume()
    assert eng.letter("A") is not None


def test_prompt_block_is_generated_and_honest():
    eng, _bus, _t = _engine()
    block = sign_prompt_block(eng)
    assert "Kenyan Sign Language" in block
    assert "P, Q, R, V, Z" in block          # the honest caveat
    assert "don't know full KSL signs yet" in block   # empty lexicon
    assert sign_prompt_block(None) == ""


# ── the lexicon validator ───────────────────────────────────────────────────

def test_lexicon_validator_refuses_malformed_and_accepts_valid():
    from zero.sign.lexicon import validate_sign

    assert validate_sign("x", {}) == "no segments"
    assert "moves nothing" in validate_sign(
        "x", {"segments": [{"hold_s": 1}]})
    assert "unknown handshape" in validate_sign(
        "x", {"segments": [{"hands": {"left": {"shape": "TRIANGLE"}}}]})
    assert "unknown orientation" in validate_sign(
        "x", {"segments": [{"hands": {"left": {"shape": "B",
                                               "orient": "sideways"}}}]})
    assert "outside 0..1" in validate_sign(
        "x", {"segments": [{"hands": {"left": {"closure": {"thumb": 3}}}}]})
    ok = {"segments": [
        {"hands": {"right": {"shape": "B", "orient": "forward"}},
         "arm": {"right_up_down_joint": 40}, "move_s": 0.5, "hold_s": 0.3},
        {"hands": {"right": {"shape": "B", "orient": "in"}}, "move_s": 0.4},
    ]}
    assert validate_sign("hello", ok) is None


def test_lexicon_sign_plays_and_drops_dark_arm_joints():
    import yaml

    eng, bus, t = _engine()
    entry = {"segments": [
        {"hands": {"right": {"shape": "B", "orient": "forward"}},
         "arm": {"right_up_down_joint": 40.0},    # a stepper: not registered
         "move_s": 0.05, "hold_s": 0.05},
    ]}
    eng._lexicon = {"hello": entry}          # inject: loader is tested above
    spoken = eng.sign("hello")
    assert spoken == "Signing hello."
    assert _wait_released(bus)
    # the hand moved; the dark arm joint never reached the wire
    assert "right_up_down_joint" not in t.posted
    assert any("right_indexp1_joint" in p for p in t.posts)


# ── the finger-gesture layer routes through ArmSystem, not the driver ──────

def test_hand_gestures_play_through_the_safe_pipeline():
    from zero.arms.system import ArmSystem

    s = ArmSystem(FakeCfg({"arms.rate_hz": 500.0,
                           "arms.hand_speed_dps": 100000.0,
                           "arms.max_speed_dps": 100000.0}))
    spoken = s.hand_gesture("peace")
    assert spoken == "Peace sign!"
    s._player.join(timeout=5.0)
    # both hands, each at ITS OWN calibration: ring closed = 0 both sides,
    # thumb closed = 140 left / 70 right
    assert s._pose["left_thumbp1_joint"] <= 140.0
    assert s._pose["right_thumbp1_joint"] <= 70.0
    assert s.hand_gesture("backflip") is None            # honest refusal
    out = s.move_finger("index", "left", closure=1.0)
    assert out == "Curling my left index."
    s._player.join(timeout=5.0)
    assert s._pose["left_indexp1_joint"] == 0.0          # closed


def test_arm_tool_sign_paths_and_no_fake_success():
    from zero.arms.system import ArmSystem
    from zero.tools.arms import ArmTool
    from zero.tools.base import ToolContext

    eng, _bus, _t = _engine()
    arms = ArmSystem(FakeCfg({"arms.rate_hz": 500.0,
                              "arms.hand_speed_dps": 100000.0,
                              "arms.max_speed_dps": 100000.0}))
    tool = ArmTool()
    ctx = ToolContext(extras={"arms": arms, "sign": eng})
    assert tool.run({"text": "spell cow"}, ctx) == "Spelling COW: C - O - W."
    assert "letter K" in tool.run({"text": "sign the letter K"}, ctx)
    # lexicon miss -> HONEST fingerspell fallback, not an invented sign
    out = tool.run({"text": "sign hello"}, ctx)
    assert "don't know the full sign for hello" in out
    assert "H - E - L - L - O" in out
    # "spell my name" with no recognised speaker -> honest ask
    assert "don't actually know your name" in tool.run(
        {"text": "spell my name"}, ctx)
    # ...and with one, it spells it
    ctx2 = ToolContext(person_name="Sam", extras={"arms": arms, "sign": eng})
    assert "S - A - M" in tool.run({"text": "spell my name"}, ctx2)
    # sign engine missing -> says so, no fake success
    ctx3 = ToolContext(extras={"arms": arms})
    assert "sign system isn't running" in tool.run(
        {"text": "spell cow"}, ctx3)


# ── the signing stance (Phase 5: in/out + shoulders live) ───────────────────

STANCE_CFG = {
    "sign.stance.enabled": True,
    "sign.stance.move_s": 0.05,
    "sign.stance_speed_dps": 100000.0,
    "sign.stance.joints": {
        "left": {"left_in_out_joint": -40.0, "left_up_down_joint": -45.0},
        "right": {"right_in_out_joint": 40.0, "right_up_down_joint": 45.0},
    },
}


def _stance_engine():
    t = NullTransport()
    bus = MotionBus(t, rate_hz=500.0)
    _BUSES.append(bus)
    for name, s in hands.hand_joint_specs().items():
        bus.register(BusJoint(name, min_deg=s["min"], max_deg=s["max"],
                              home_deg=s["home"], batch=True))
    for name, lo, hi in (("left_in_out_joint", -136.5, 0.0),
                         ("right_in_out_joint", 0.0, 136.5),
                         ("left_up_down_joint", -106.0, 68.8),
                         ("right_up_down_joint", -68.8, 106.0)):
        bus.register(BusJoint(name, min_deg=lo, max_deg=hi, home_deg=0.0))
    cfg = FakeCfg({"sign.letter_s": 0.06, "sign.transition_s": 0.03,
                   "sign.rate_hz": 500, "sign.max_speed_dps": 100000.0,
                   "sign.lexicon_path": "/nonexistent.yaml", **STANCE_CFG})
    return SignEngine(cfg, bus), bus, t


def test_stance_rises_before_letters_and_lowers_to_rest_after():
    eng, bus, t = _stance_engine()
    eng.spell("HI")
    assert _wait_released(bus)
    io = [p["right_in_out_joint"] for p in t.posts
          if "right_in_out_joint" in p]
    assert io, "stance joints must move during a spell"
    assert max(io) > 39.0                        # rose to the stance
    assert abs(io[-1]) < 1.0                     # ...and came back to rest
    # the first HANDSHAPE write comes only after the stance target appears
    first_hand = next(i for i, p in enumerate(t.posts)
                      if "left_indexp1_joint" in p)
    first_stance = next(i for i, p in enumerate(t.posts)
                        if "right_in_out_joint" in p)
    assert first_stance < first_hand
    # left side is mirrored: it abducted NEGATIVE
    lio = [p["left_in_out_joint"] for p in t.posts
           if "left_in_out_joint" in p]
    assert min(lio) < -39.0 and abs(lio[-1]) < 1.0


def test_stance_degrades_to_hands_only_when_steppers_unregistered():
    # Same config, but a hands-only bus (arms.driver: null world): the
    # stance is filtered out and spelling works exactly as before.
    t = NullTransport()
    bus = MotionBus(t, rate_hz=500.0)
    _BUSES.append(bus)
    for name, s in hands.hand_joint_specs().items():
        bus.register(BusJoint(name, min_deg=s["min"], max_deg=s["max"],
                              home_deg=s["home"], batch=True))
    cfg = FakeCfg({"sign.letter_s": 0.06, "sign.transition_s": 0.03,
                   "sign.rate_hz": 500, "sign.max_speed_dps": 100000.0,
                   "sign.lexicon_path": "/nonexistent.yaml", **STANCE_CFG})
    eng = SignEngine(cfg, bus)
    assert eng.spell("HI") is not None
    assert _wait_released(bus)
    assert not any("right_in_out_joint" in p for p in t.posts)
    assert any("left_indexp1_joint" in p for p in t.posts)


def test_stance_steppers_move_at_their_own_slower_cap():
    import time as _t

    t = NullTransport()
    bus = MotionBus(t, rate_hz=500.0)
    _BUSES.append(bus)
    for name, s in hands.hand_joint_specs().items():
        bus.register(BusJoint(name, min_deg=s["min"], max_deg=s["max"],
                              home_deg=s["home"], batch=True))
    bus.register(BusJoint("right_in_out_joint", min_deg=0.0, max_deg=136.5))
    cfg = FakeCfg({"sign.letter_s": 0.02, "sign.transition_s": 0.02,
                   "sign.rate_hz": 500, "sign.max_speed_dps": 100000.0,
                   "sign.lexicon_path": "/nonexistent.yaml",
                   "sign.stance.enabled": True,
                   "sign.stance.move_s": 0.02,        # asks for instant...
                   "sign.stance_speed_dps": 90.0,      # ...but the cap rules
                   "sign.stance.joints": {
                       "right": {"right_in_out_joint": 40.0}}})
    eng = SignEngine(cfg, bus)
    t0 = _t.monotonic()
    eng.spell("A", side="right")
    assert _wait_released(bus)
    took = _t.monotonic() - t0
    # 40 deg at min-jerk under a 90 dps cap needs >= 1.875*40/90 = 0.83 s
    # each way; well over the ~0.1 s the letters themselves need.
    assert took >= 1.5, f"stance moved too fast: {took:.2f}s"


def test_service_stop_eases_everything_home_not_abandons_it():
    """The shutdown guarantee: stop() mid-spell EASES raised joints back to
    rest — open hands, stance down — instead of freezing the letter in the
    air. This is what makes `systemctl stop zero` leave the robot at its
    original angles."""
    eng, bus, t = _stance_engine()
    eng.spell("CALIBRATION")               # long word — plenty of mid-air
    # wait until the stance is up AND a letter is actually on the hand
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        if (t.posted.get("right_in_out_joint", 0.0) > 20.0
                and t.posted.get("left_indexp1_joint", 90.0) < 60.0):
            break
        time.sleep(0.01)
    assert t.posted.get("right_in_out_joint", 0.0) > 20.0
    assert t.posted.get("left_indexp1_joint", 90.0) < 60.0   # mid-letter
    n_before = len(t.posts)
    eng.stop()                             # blocks until lowered
    time.sleep(0.1)
    assert bus.owner("right_in_out_joint") != "sign"
    assert abs(t.posted["right_in_out_joint"]) < 1.0        # stance down
    assert abs(t.posted["left_in_out_joint"]) < 1.0
    assert abs(t.posted["left_indexp1_joint"] - 90.0) < 2.0  # hand open
    # ...and it EASED there: several intermediate posts, not one hop
    lowering = [p["right_in_out_joint"] for p in t.posts[n_before:]
                if "right_in_out_joint" in p]
    assert len(lowering) >= 5


def test_estop_stop_releases_without_moving():
    """Under e-stop, stop() must NOT drive anything home — an e-stop means
    do not move, including to go to rest."""
    eng, bus, t = _stance_engine()
    eng.spell("HI")
    end = time.monotonic() + 5.0
    while time.monotonic() < end and \
            t.posted.get("right_in_out_joint", 0.0) < 20.0:
        time.sleep(0.01)
    bus.estop()
    frozen = dict(t.posted)
    eng.stop()
    time.sleep(0.1)
    assert t.posted == frozen              # nothing moved after the e-stop
    assert bus.owner("right_in_out_joint") != "sign"
