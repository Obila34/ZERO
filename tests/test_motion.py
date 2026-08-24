"""MotionBus: priority arbitration, clamping, walking, batching, hold-on-
release, offsets, e-stop — all against NullTransport (nothing moves)."""
import time

from zero.motion.bus import BusJoint, MotionBus
from zero.motion.transport import NullTransport


def _bus(**joints):
    t = NullTransport()
    bus = MotionBus(t, rate_hz=500.0)
    for name, kw in joints.items():
        bus.register(BusJoint(name, **kw))
    return bus, t


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_higher_priority_track_wins_and_release_holds():
    bus, t = _bus(j=dict(min_deg=0, max_deg=90))
    bus.write("gesture", {"j": 10.0})
    bus.write("sign", {"j": 80.0})
    assert _wait(lambda: t.posted.get("j") == 80.0)
    assert bus.owner("j") == "sign"
    # release: the bus HOLDS the last posted value — the stale gesture
    # setpoint must not resurface on its own...
    bus.release("sign")
    time.sleep(0.05)
    assert t.posted["j"] == 80.0
    # the sign's writes PREEMPTED the gesture's standing claim — a finished
    # beat must not snap the joint back when the sign releases. The joint is
    # unclaimed until some track writes it again.
    assert bus.owner("j") is None
    bus.write("gesture", {"j": 5.0})         # a fresh write reclaims it
    assert _wait(lambda: t.posted.get("j") == 5.0)
    bus.close()


def test_envelope_clamps_at_the_bus():
    bus, t = _bus(j=dict(min_deg=0, max_deg=90))
    bus.write("sign", {"j": 500.0})
    assert _wait(lambda: t.posted.get("j") == 90.0)
    bus.close()


def test_unregistered_joint_never_reaches_the_wire():
    bus, t = _bus(j=dict(min_deg=0, max_deg=90))
    accepted = bus.write("sign", {"ghost_joint": 10.0, "j": 1.0})
    assert accepted == ["j"]
    assert _wait(lambda: "j" in t.posted)
    assert "ghost_joint" not in t.posted
    bus.close()


def test_max_jump_walks_after_outage_swing():
    bus, t = _bus(j=dict(min_deg=-80, max_deg=80, max_jump_deg=16))
    bus.write("gaze", {"j": 40.0})
    assert _wait(lambda: t.posted.get("j") == 40.0)
    bus.write("gaze", {"j": -40.0})          # 80-degree swing
    assert _wait(lambda: t.posted.get("j") == -40.0)
    hops = [p["j"] for p in t.posts if "j" in p]
    assert hops == [40.0, 24.0, 8.0, -8.0, -24.0, -40.0]
    bus.close()


def test_hand_joints_batch_into_one_pose_cmd():
    bus, t = _bus(a=dict(min_deg=0, max_deg=90, batch=True),
                  b=dict(min_deg=0, max_deg=90, batch=True))
    bus.write("sign", {"a": 10.0, "b": 20.0})
    assert _wait(lambda: t.posted.get("a") == 10.0 and t.posted.get("b") == 20.0)
    assert any(len(p) == 2 for p in t.posts)   # ONE post carried both
    bus.close()


def test_offset_joint_is_mute_until_offsets_known_then_subtracts():
    class NoOffsets(NullTransport):
        def __init__(self):
            super().__init__()
            self.offsets = None

        def fetch_offsets(self):
            return self.offsets

    t = NoOffsets()
    bus = MotionBus(t, rate_hz=500.0)
    bus.register(BusJoint("s", min_deg=-90, max_deg=90, use_offset=True))
    bus.write("command", {"s": 10.0})
    time.sleep(0.1)
    assert "s" not in t.posted               # mute: offsets unknown
    t.offsets = {"s": -150.0}                # the real 2026-08-17 case
    # the WIRE value subtracted the stored offset (effective -> raw): posting
    # a raw 10 would have commanded the shoulder 150 deg off. The bus's own
    # belief stays in effective degrees.
    assert _wait(lambda: t.posted.get("s") == 160.0)
    assert bus.last["s"] == 10.0
    bus.close()


def test_estop_freezes_every_track_and_resume_recovers():
    bus, t = _bus(j=dict(min_deg=0, max_deg=90))
    bus.write("gaze", {"j": 10.0})
    assert _wait(lambda: t.posted.get("j") == 10.0)
    bus.estop()
    assert t.stops == 1
    bus.write("sign", {"j": 50.0})           # even sign is frozen
    time.sleep(0.1)
    assert t.posted["j"] == 10.0
    bus.resume()
    assert _wait(lambda: t.posted.get("j") == 50.0)
    bus.close()


def test_failed_posts_retry_and_flip_healthy():
    class Flaky(NullTransport):
        def __init__(self):
            super().__init__()
            self.down = True

        def post_joint(self, name, deg):
            if self.down:
                return False
            return super().post_joint(name, deg)

    t = Flaky()
    bus = MotionBus(t, rate_hz=500.0)
    bus.register(BusJoint("j", min_deg=0, max_deg=90))
    bus.write("gaze", {"j": 30.0})
    assert _wait(lambda: not bus.healthy, timeout=3.0)
    assert "j" not in t.posted
    t.down = False                           # link restored
    assert _wait(lambda: t.posted.get("j") == 30.0, timeout=3.0)
    assert bus.healthy
    bus.close()


def test_bus_head_driver_ports_the_nod_mapping():
    from zero.motion.drivers import BusHeadDriver

    t = NullTransport()
    bus = MotionBus(t, rate_hz=500.0)
    drv = BusHeadDriver(bus, pan_joint="head_tilt_joint",
                        tilt_joint="head_nod_joint", limit_deg=80,
                        nod_sign=-1.0, nod_offset_deg=43.4,
                        nod_min_deg=43.4, nod_max_deg=71.4,
                        max_jump_deg=0.0)
    # rest: tilt 0 -> the calibrated nod home (servo 133.4 = angle 43.4)
    drv.send(10.0, 0.0)
    assert _wait(lambda: t.posted.get("head_nod_joint") == 43.4)
    assert t.posted["head_tilt_joint"] == 10.0
    # +tilt = up, but rest IS the up stop -> clamped at 43.4
    drv.send(10.0, 20.0)
    time.sleep(0.05)
    assert t.posted["head_nod_joint"] == 43.4
    # look down 10 -> 53.4, inside the window
    drv.send(10.0, -10.0)
    assert _wait(lambda: t.posted.get("head_nod_joint") == 53.4)
    bus.close()


# ── the joint-angle black box ───────────────────────────────────────────────

def test_blackbox_records_acked_posts_with_owning_track(tmp_path):
    from zero.motion.blackbox import JointAngleLog

    box = JointAngleLog(str(tmp_path / "joints.sqlite"))
    t = NullTransport()
    bus = MotionBus(t, rate_hz=500.0, blackbox=box)
    bus.register(BusJoint("j", min_deg=0, max_deg=90))
    bus.write("sign", {"j": 40.0})
    assert _wait(lambda: t.posted.get("j") == 40.0)
    assert _wait(lambda: "j" in box.last_angles())
    deg, _ts, src = box.last_angles()["j"]
    assert deg == 40.0 and src == "sign"
    bus.close()


def test_blackbox_throttles_noise_but_keeps_big_hops(tmp_path):
    import sqlite3

    from zero.motion.blackbox import JointAngleLog

    db = str(tmp_path / "joints.sqlite")
    box = JointAngleLog(db)
    box.log("j", 10.0, "gaze")
    box.log("j", 10.05, "gaze")       # sub-deadband jitter — dropped
    box.log("j", 10.1, "gaze")        # still noise — dropped
    box.log("j", 40.0, "gaze")        # big hop — always recorded
    n = sqlite3.connect(db).execute(
        "SELECT COUNT(*) FROM joint_angles").fetchone()[0]
    assert n == 2
    box.close()


def test_blackbox_snapshot_and_broken_disk_never_raise(tmp_path):
    from zero.motion.blackbox import JointAngleLog

    box = JointAngleLog(str(tmp_path / "joints.sqlite"))
    assert box.snapshot({"a": 1.0, "b": 2.0}, "telemetry") == 2
    assert box.last_angles()["b"][0] == 2.0
    box.close()
    # a dead DB records nothing and raises nothing — motion must not care
    dead = JointAngleLog("/nonexistent-dir/x/joints.sqlite")
    dead.log("j", 1.0, "gaze")
    assert dead.snapshot({"j": 1.0}, "t") == 0
    assert dead.last_angles() == {}
