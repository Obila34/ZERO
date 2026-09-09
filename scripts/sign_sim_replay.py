#!/usr/bin/env python3
"""Sign playback simulator — commanded vs PHYSICAL, before any metal moves.

Runs a real sign sequence through the REAL SignEngine + dictionary +
MotionBus (recording transport, nothing moves), then simulates what the
stepper firmware actually does with those commands: AccelStepper-style
trapezoidal motion at the Nano's configured acceleration (2500 steps/s^2
= ~28 deg/s^2 at the assumed 88.9 steps/deg — glacial for direction
changes). Servo hand joints are modeled as fast first-order tracking.

Outputs, per joint: tracking lag, RMS error, and the keyframe-stretch
distribution that quantifies the shared-duration flaw (audit finding 1:
one slow arm joint time-warps every finger in the frame). Also dumps
the full command + simulated-physical streams to JSON for the Blender
rig / Isaac stage on zerolabs0.

    .venv/bin/python scripts/sign_sim_replay.py hello please yes
    .venv/bin/python scripts/sign_sim_replay.py --vmax-steps 4000 thankyou
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

STEPS_PER_DEG = 88.8889          # gateway global (per-joint truth pending T1.3)
ACCEL_STEPS = 2500.0             # nano1_steppers.ino setAcceleration
ARM_KEYWORDS = ("elbow", "up_down", "in_out", "tilt")


class RecordingTransport:
    """NullTransport with timestamps — the command stream, as the gateway
    would have received it."""

    def __init__(self):
        self.rows: list[tuple[float, str, float]] = []
        self.moves_hardware = False

    def post_joint(self, name, deg):
        self.rows.append((time.monotonic(), name, float(deg)))
        return True

    def post_pose(self, pose):
        t = time.monotonic()
        for n, v in pose.items():
            self.rows.append((t, n, float(v)))
        return True

    def stop(self):
        return True

    def resume(self):
        return True

    def fetch_offsets(self):
        return {}                 # sim: offsets are the gateway's business


def capture(glosses):
    from zero.arms.hands import hand_joint_specs
    from zero.motion.bus import BusJoint, MotionBus
    from zero.sign.engine import SignEngine

    class Cfg(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    cfg = Cfg({"sign.dictionary.enabled": True,
               "sign.dictionary.path": "data/sign_dictionary.npz"})
    t = RecordingTransport()
    bus = MotionBus(t, rate_hz=100.0)
    for name, s in hand_joint_specs().items():
        bus.register(BusJoint(name, min_deg=s["min"], max_deg=s["max"],
                              home_deg=s["home"], batch=True))
    for name, lo, hi in [("right_elbow_joint", -73.5, 73.5),
                         ("left_elbow_joint", -73.5, 73.5),
                         ("right_up_down_joint", -68.8, 106.0),
                         ("left_up_down_joint", -106.0, 68.8),
                         ("right_in_out_joint", 0.0, 136.5),
                         ("left_in_out_joint", -136.5, 0.0)]:
        bus.register(BusJoint(name, min_deg=lo, max_deg=hi, home_deg=0.0,
                              use_offset=True, max_jump_deg=16.0))
    eng = SignEngine(cfg, bus)
    t0 = time.monotonic()
    played = eng.sign_sequence(list(glosses))
    while eng.busy:
        time.sleep(0.05)
    time.sleep(0.3)
    eng.stop()
    bus.close()
    rows = [(ts - t0, n, v) for ts, n, v in t.rows]
    return played, rows


def simulate_stepper(times, targets, vmax_steps):
    """AccelStepper-ish: trapezoid toward the LATEST target, re-planned
    whenever the target changes. Returns physical position at `times`."""
    a = ACCEL_STEPS / STEPS_PER_DEG               # deg/s^2
    vmax = vmax_steps / STEPS_PER_DEG             # deg/s
    pos, vel = targets[0], 0.0
    out = [pos]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        tgt = targets[i - 1]                      # target active during step
        # decelerate if we'd overshoot given stopping distance, else chase
        n = max(1, int(dt / 0.002))
        h = dt / n
        for _ in range(n):
            d = tgt - pos
            stop_d = (vel * vel) / (2 * a) if a > 0 else 0.0
            if abs(d) < 1e-6 and abs(vel) < 1e-3:
                vel = 0.0
            else:
                want = np.sign(d)
                if np.sign(vel) == want and stop_d >= abs(d):
                    vel -= np.sign(vel) * a * h   # braking zone
                else:
                    vel += want * a * h
                vel = float(np.clip(vel, -vmax, vmax))
            pos += vel * h
        out.append(pos)
    return np.array(out)


def servo_track(times, targets, tau=0.05):
    """Hand servo: fast first-order lag (50 ms)."""
    pos = targets[0]
    out = [pos]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        pos += (targets[i - 1] - pos) * min(1.0, dt / tau)
        out.append(pos)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("glosses", nargs="*", default=["hello", "please", "yes"])
    ap.add_argument("--vmax-steps", type=float, default=4000.0,
                    help="firmware setMaxSpeed guess, steps/s (sweep me — "
                         "the .ino value is unverified)")
    ap.add_argument("--out", default="/tmp/sign_sim_replay.json")
    a = ap.parse_args()

    played, rows = capture(a.glosses or ["hello"])
    if not played:
        print("nothing played — dictionary missing?")
        return 1
    print(f"captured {len(rows)} joint commands for {played}")

    by_joint: dict[str, list] = {}
    for ts, n, v in rows:
        by_joint.setdefault(n, []).append((ts, v))

    report = {}
    dump = {"played": played, "vmax_steps": a.vmax_steps, "joints": {}}
    for name, seq in sorted(by_joint.items()):
        t = np.array([x[0] for x in seq])
        v = np.array([x[1] for x in seq])
        if len(t) < 3:
            continue
        is_arm = any(k in name for k in ARM_KEYWORDS)
        phys = (simulate_stepper(t, v, a.vmax_steps) if is_arm
                else servo_track(t, v))
        err = np.abs(phys - v)
        # lag: shift of best correlation between command and physical
        lag_ms = 0.0
        vc, pc = v - v.mean(), phys - phys.mean()
        if vc.std() > 1e-3 and pc.std() > 1e-3 and len(v) > 8:
            xc = np.correlate(pc, vc, mode="full")
            lag_i = int(np.argmax(xc)) - (len(v) - 1)
            med_dt = float(np.median(np.diff(t))) or 0.01
            lag_ms = lag_i * med_dt * 1000.0
        report[name] = {"cmd_range_deg": round(float(v.max() - v.min()), 1),
                        "rms_err_deg": round(float(np.sqrt((err**2).mean())), 2),
                        "max_err_deg": round(float(err.max()), 2),
                        "lag_ms": round(lag_ms, 0),
                        "n": len(v), "arm": is_arm}
        dump["joints"][name] = {"t": t.round(4).tolist(),
                                "cmd": v.round(3).tolist(),
                                "phys": phys.round(3).tolist()}

    print(f"\n{'joint':24s} {'range':>6s} {'rms_err':>8s} {'max_err':>8s} "
          f"{'lag':>7s}")
    for n, r in sorted(report.items(), key=lambda kv: -kv[1]["max_err_deg"]):
        tag = "ARM " if r["arm"] else "hand"
        print(f"{n:24s} {r['cmd_range_deg']:6.1f} {r['rms_err_deg']:8.2f} "
              f"{r['max_err_deg']:8.2f} {r['lag_ms']:6.0f}ms  {tag}")
    arm_max = max((r["max_err_deg"] for r in report.values() if r["arm"]),
                  default=0.0)
    print(f"\nworst ARM tracking error: {arm_max:.1f} deg "
          f"(the gap between what the engine believes and what metal does)")
    with open(a.out, "w") as f:
        json.dump(dump, f)
    print(f"streams -> {a.out} (feed to the zl0 rig / Isaac)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
