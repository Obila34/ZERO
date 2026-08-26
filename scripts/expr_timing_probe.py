#!/usr/bin/env python3
"""Living Hands timing probe — ON REAL HARDWARE.

Drives the actual HandScheduler through the actual MotionBus/HTTP gateway
with a synthetic stressed 'sentence' played out at real time, then reads
the joint black box back to measure what the plan only estimated:

  * gateway ack latency (bus write -> HTTP 200), sampled directly
  * beat apex error: when the index-finger excursion peaked (black box,
    source 'idle') vs when the accent was scheduled to land

MOVES THE FINGERS (small beat pulses + one count shape). Have eyes on the
robot. Run on the Pi:

    .venv/bin/python scripts/expr_timing_probe.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from zero.config import load_config  # noqa: E402
from zero.expr.schedule import HandScheduler  # noqa: E402
from zero.motion.drivers import get_bus, reset_bus  # noqa: E402

SR = 24000


def stress_sentence(accents=(0.8, 1.8, 2.8), dur=3.6):
    x = np.zeros(int(dur * SR), dtype=np.float32)

    def place(t, f0, amp, blen):
        n = int(blen * SR)
        tt = np.arange(n) / SR
        f = f0 * (1 + 0.3 * np.sin(np.pi * tt / blen))
        sig = np.hanning(n) * amp * np.sin(2 * np.pi * np.cumsum(f) / SR)
        i = int(t * SR)
        x[i:i + n] += sig[:len(x) - i]

    t = 0.15
    for a in accents:
        while t < a - 0.25:
            place(t, 140, 0.15, 0.14)
            t += 0.33
        place(a - 0.09, 185, 0.5, 0.18)
        t = a + 0.3
    place(t, 140, 0.15, 0.14)
    return x + 0.003 * np.random.randn(len(x)).astype(np.float32)


def main() -> int:
    state = subprocess.run(
        ["systemctl", "is-active", "zero.service"],
        capture_output=True, text=True).stdout.strip()
    if state == "active":
        print("NOTE: zero.service is running — its gaze loop shares the "
              "gateway. Hands are probe-exclusive; head chatter is fine.")

    cfg = load_config()
    if str(cfg.get("motion.driver")) != "http":
        print("motion.driver is not http — this probe is pointless. Abort.")
        return 1
    bus = get_bus(cfg)
    # ensure hand joints are registered (same firmware-truth specs)
    from zero.arms.hands import hand_joint_specs
    from zero.motion.bus import BusJoint

    for name, sp in hand_joint_specs().items():
        if bus.spec(name) is None:
            bus.register(BusJoint(name, min_deg=sp["min"], max_deg=sp["max"],
                                  home_deg=sp["home"], batch=True))

    # 1) raw gateway ack latency, 10 samples on a real joint at its rest
    from zero.motion.transport import HttpTransport

    tr = HttpTransport(base_url=cfg.get("motion.gateway.base_url"),
                       timeout_s=2.0)
    lat = []
    for _ in range(10):
        t0 = time.perf_counter()
        ok = tr.post_joint("left_wrist_joint", 90.0)
        if ok:
            lat.append((time.perf_counter() - t0) * 1000.0)
        time.sleep(0.05)
    lat.sort()
    print(f"\ngateway ack latency ({len(lat)}/10 ok): "
          f"median {lat[len(lat)//2]:.0f} ms, p90 {lat[int(len(lat)*0.9)-1]:.0f} ms"
          if lat else "gateway unreachable!")

    # 2) the scheduler, for real
    sched = HandScheduler(cfg, bus)
    playout_delay = float(cfg.get("expression.hands.playout_delay_ms", 320.0))
    latency_ms = float(cfg.get("expression.hands.latency_ms", 60.0))
    accents = (0.8, 1.8, 2.8)
    audio = stress_sentence(accents)
    print(f"\nWATCH THE HANDS: 3 beat pulses over ~4 s "
          f"(planned accents at {accents} s + {playout_delay:.0f} ms "
          f"playout delay - {latency_ms:.0f} ms wire latency)")
    sched.on_audio(0, "well THAT was quite the THING to SEE", audio, SR)
    hop = int(0.05 * SR)
    t_anchor = time.time()
    for i in range(0, len(audio), hop):
        sched.on_playout(0, hop)
        time.sleep(0.05)
    time.sleep(2.5)
    sched.stop()

    # 3) read the black box back: idle-track index-finger excursions
    import sqlite3

    db = cfg.get("motion.blackbox.db_path", "zero_joints.sqlite")
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT ts, angle_deg FROM joint_angles WHERE joint = ? AND "
        "source = 'idle' AND ts >= ? ORDER BY ts",
        ("left_indexp1_joint", t_anchor - 1)).fetchall()
    conn.close()
    if not rows:
        print("\nNo idle-track rows in the black box — did the hands move?")
        return 1
    print(f"\nblack box: {len(rows)} index-finger rows recorded")
    # find local minima (deepest curl) per accent window
    planned = [a + playout_delay / 1000.0 - latency_ms / 1000.0
               for a in accents]
    print(f"{'accent':>8s} {'planned':>9s} {'measured':>9s} {'error':>8s}")
    errs = []
    for a, p in zip(accents, planned):
        win = [(ts - t_anchor, deg) for ts, deg in rows
               if abs((ts - t_anchor) - p) < 0.5]
        if not win:
            print(f"{a:8.2f} {p:9.2f} {'—':>9s}  (no excursion recorded)")
            continue
        t_meas, deg = min(win, key=lambda r: r[1])
        err = (t_meas - p) * 1000.0
        errs.append(err)
        print(f"{a:8.2f} {p:9.2f} {t_meas:9.2f} {err:+7.0f}ms  (curl to "
              f"{deg:.1f} deg)")
    if errs:
        med = sorted(abs(e) for e in errs)[len(errs) // 2]
        print(f"\nmedian |apex error| = {med:.0f} ms "
              f"(budget: <=50 ms planned, <=100 ms lag hard ceiling)")
        print("Positive error = hand LATE vs plan -> raise latency_ms by "
              "that amount.\nConsistent negative = lower playout_delay_ms.")
    reset_bus()
    return 0


if __name__ == "__main__":
    sys.exit(main())
