"""Motion sign dictionary: compile-time handedness/caps, playback keyframes,
engine precedence, and the ships-dark guarantee."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sign_dict_compile import compile_dictionary  # noqa: E402
from zero.sign.dictionary import SignDictionary, load_dictionary  # noqa: E402


class FakeCfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


def _synthetic_vault(tmp_path):
    """Two signs; the 'dominant' motion lives in the DATA-LEFT block, the
    way the real recordings do (unmirrored camera)."""
    T = 48
    frames = np.zeros((2, T, 144), dtype=np.float16)
    t = np.linspace(0, 2 * np.pi, T)
    # sign 0 'wave': data-left hand (-> robot RIGHT) raised and moving
    lh = np.zeros((T, 21, 3), dtype=np.float32)
    lh[:, :, 0] = 0.5
    lh[:, 9, 1] = 0.1                      # middle MCP: span reference
    lh[:, 8, 1] = 0.05 + 0.1 * np.sin(t)   # index tip curls rhythmically
    lh[:, 6, 1] = 0.08
    frames[0, :, 18:81] = lh.reshape(T, 63)
    arm = np.zeros((T, 6, 3), dtype=np.float32)
    arm[:, 0] = (0.4, 0.0, 0.0)            # data-left shoulder
    arm[:, 1] = (0.6, 0.0, 0.0)
    arm[:, 2] = (0.38, 0.25, 0.0)          # elbow below shoulder
    arm[:, 4] = (0.36, -0.30, 0.0)         # data-left wrist ABOVE shoulder
    # the wrist BOBS: compile keeps motion (centered dynamics), not posture
    arm[:, 4, 1] = -0.30 + 0.15 * np.sin(t)
    frames[0, :, :18] = arm.reshape(T, 18)
    # sign 1 'still': nothing visible at all
    np.savez_compressed(tmp_path / "sign_motion.npz",
                        glosses=np.array(["wave", "still"]),
                        frames=frames)
    vault = tmp_path
    (vault / "generation" / "sign-dictionary").mkdir(parents=True)
    os.replace(tmp_path / "sign_motion.npz",
               vault / "generation" / "sign-dictionary" / "sign_motion.npz")
    return str(vault)


def test_compile_swaps_handedness_and_caps_arms(tmp_path):
    out = compile_dictionary(_synthetic_vault(tmp_path))
    assert list(out["glosses"]) == ["wave", "still"]
    # data-left motion must land on the robot's RIGHT side
    assert out["present"][0, :, 1].all(), "robot-right hand not present"
    assert not out["present"][0, :, 0].any(), "phantom left hand"
    r_cl = np.float32(out["closures"][0, :, 5:])
    assert r_cl.std() > 0.01, "finger motion lost"
    arms = np.float32(out["arms"][0])
    # motion-centered: the bobbing wrist survives as raise DYNAMICS
    assert arms[:, 1].std() > 2.0, "arm motion lost"          # R raise
    assert abs(arms[:, 1].mean()) < 2.0, "posture not centered out"
    assert np.abs(arms).max() <= 45.0 + 1e-3, "arm cap broken"
    # invisible sign compiles to stillness, not garbage
    assert np.float32(out["closures"][1]).max() == 0.0


def test_dictionary_frames_and_engine_playback(tmp_path):
    out = compile_dictionary(_synthetic_vault(tmp_path))
    np.savez_compressed(tmp_path / "dict.npz", **out)
    d = SignDictionary(str(tmp_path / "dict.npz"))
    assert "wave" in d and "WAVE " in [g.upper() + " " for g in ["wave"]]
    # play-time contract: the stance carries height, deltas ride on top
    stance = {"right_up_down_joint": 45.0, "right_elbow_joint": -30.0}
    got = d.frames("wave", stance=stance)
    assert got is not None
    frames, arm_joints, sides = got
    assert sides == ("right",)
    assert "right_up_down_joint" in arm_joints
    ud = [p["right_up_down_joint"] for p, _m, _h in frames
          if "right_up_down_joint" in p]
    assert max(ud) - min(ud) > 3.0, "arm dynamics lost at play time"
    assert 30.0 < np.mean(ud) <= 65.0, "stance base missing from arm"
    assert len(frames) > 8
    for pose, mv, hold in frames:
        for j, v in pose.items():
            if "elbow" in j or "up_down" in j or "in_out" in j:
                assert abs(v) <= 45.0 + 1e-3
    assert d.frames("nonsense") is None
    # through the real engine on a Null bus: lexicon empty -> dictionary
    from zero.motion.bus import BusJoint, MotionBus
    from zero.motion.transport import NullTransport
    from zero.arms.hands import hand_joint_specs
    from zero.sign.engine import SignEngine
    t = NullTransport()
    bus = MotionBus(t, rate_hz=200.0)
    for name, s in hand_joint_specs().items():
        bus.register(BusJoint(name, min_deg=s["min"], max_deg=s["max"],
                              home_deg=s["home"], batch=True))
    cfg = FakeCfg({"sign.dictionary.enabled": True,
                   "sign.dictionary.path": str(tmp_path / "dict.npz"),
                   "sign.stance.enabled": False})
    eng = SignEngine(cfg, bus)
    assert eng.knows_sign("wave")
    said = eng.sign("wave")
    assert said == "Signing wave."
    import time
    time.sleep(1.2)
    eng.stop()
    bus.close()
    moved = [p for p in t.posts if "right_indexp1_joint" in p]
    assert moved, "dictionary sign never reached the bus"


def test_ships_dark_and_missing_asset_is_harmless(tmp_path):
    assert load_dictionary(FakeCfg({})) is None
    assert load_dictionary(FakeCfg({
        "sign.dictionary.enabled": True,
        "sign.dictionary.path": str(tmp_path / "absent.npz")})) is None
