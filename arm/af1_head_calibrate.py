#!/usr/bin/env python3
"""AF-1 head calibration — SUPERVISED. Measures the head numbers the ZERO-side
plan flagged as estimates. RUN THIS WITH A HUMAN WATCHING THE ROBOT.

It answers, safely and incrementally:
  * command-path latency  — POST round-trip stats to /api/joint_cmd (the network
                            + gateway + serial-write share; the MECHANICAL
                            latency needs a camera — see --film below).
  * the SIGN test          — commands a small +pan then +tilt so you can read off
                            which way the head actually turns and set
                            head.tracker.pan_sign / tilt_sign on the ZERO side.
  * range / limits         — steps to +/- the soft limit in small increments and
                            back, so you can confirm the mechanical range and
                            watch for binding.
  * a filmable smoothness sweep — a slow point-to-point move you record at
                            120/240 fps; track a fiducial on the head, then feed
                            the angle-vs-time series into
                            zero.head.smoothness.summarize for objective jerk /
                            reversal / overshoot / settling numbers on the REAL
                            actuator (compare against the NullDriver baseline).

SAFE BY DEFAULT: prints the plan and moves NOTHING unless you pass --arm. Small
increments, dwell between steps, always starts and ends at home, and Ctrl-C
forwards /api/stop. Never exceeds --limit.

    python3 af1_head_calibrate.py                 # dry-run: show the plan
    python3 af1_head_calibrate.py --arm --sign    # supervised sign test
    python3 af1_head_calibrate.py --arm --sweep --film   # slow sweep to record
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request

STEP = 5.0          # deg per increment on ranged moves (gentle)


def post(gateway, joint, deg, *, arm):
    body = json.dumps({"name": joint, "angle_deg": deg,
                       "angle_rad": deg * math.pi / 180.0}).encode()
    if not arm:
        print(f"    [dry] {joint} -> {deg:+.1f} deg")
        return 0.0
    req = urllib.request.Request(gateway + "/api/joint_cmd", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.perf_counter()
    urllib.request.urlopen(req, timeout=1.0).close()
    return (time.perf_counter() - t0) * 1000.0   # ms


def stop(gateway, *, arm):
    if not arm:
        print("    [dry] /api/stop")
        return
    try:
        urllib.request.urlopen(urllib.request.Request(
            gateway + "/api/stop", data=b"", method="POST"), timeout=1.0).close()
    except Exception as e:
        print(f"    stop failed: {e}")


def ramp(gateway, joint, lo, hi, *, arm, dwell, latencies):
    """Step joint from lo to hi and back in STEP increments, dwelling each step."""
    seq = list(_frange(lo, hi, STEP)) + list(_frange(hi, lo, -STEP))
    for deg in seq:
        latencies.append(post(gateway, joint, deg, arm=arm))
        time.sleep(dwell)


def _frange(a, b, step):
    n = int(abs((b - a) / step)) + 1
    return [round(a + i * step * (1 if b >= a else 1), 3) if False else
            round(a + (b - a) * i / max(1, n - 1), 3) for i in range(n)]


def do_sign(gw, a):
    print("\n== SIGN TEST == watch the head; note the direction of each move")
    lat = []
    post(gw, a.pan_joint, 0.0, arm=a.arm); time.sleep(0.5)
    print("  commanding pan +10 deg — the head should turn to ITS OWN RIGHT")
    lat.append(post(gw, a.pan_joint, 10.0, arm=a.arm)); time.sleep(1.2)
    post(gw, a.pan_joint, 0.0, arm=a.arm); time.sleep(0.8)
    print("  commanding tilt +10 deg — the head should look UP")
    lat.append(post(gw, a.tilt_joint, 10.0, arm=a.arm)); time.sleep(1.2)
    post(gw, a.tilt_joint, 0.0, arm=a.arm); time.sleep(0.5)
    print("  If a move went the OTHER way, flip that axis' sign on the ZERO side")
    print("  (head.tracker.pan_sign / tilt_sign).")
    _report_latency(lat)


def do_sweep(gw, a):
    print("\n== SMOOTHNESS SWEEP == "
          + ("RECORD at 120/240 fps now. " if a.film else "")
          + f"slow {a.limit:.0f} deg point-to-point, both axes")
    lat = []
    post(gw, a.pan_joint, 0.0, arm=a.arm); time.sleep(0.8)
    print("  pan: 0 -> +lim -> -lim -> 0")
    ramp(gw, a.pan_joint, 0.0, a.limit, arm=a.arm, dwell=a.dwell, latencies=lat)
    ramp(gw, a.pan_joint, 0.0, -a.limit, arm=a.arm, dwell=a.dwell, latencies=lat)
    print("  tilt: 0 -> +lim -> -lim -> 0")
    ramp(gw, a.tilt_joint, 0.0, a.limit, arm=a.arm, dwell=a.dwell, latencies=lat)
    ramp(gw, a.tilt_joint, 0.0, -a.limit, arm=a.arm, dwell=a.dwell, latencies=lat)
    _report_latency(lat)
    print("  Now feed your fiducial-tracked angle series into "
          "zero.head.smoothness.summarize(t, angle).")


def _report_latency(lat):
    lat = [x for x in lat if x > 0]
    if not lat:
        print("  (dry-run: no timing)")
        return
    print(f"  command-path latency ms: n={len(lat)} "
          f"min={min(lat):.1f} median={statistics.median(lat):.1f} "
          f"max={max(lat):.1f}")
    print("  NOTE: this is POST+gateway+serial-write, NOT mechanical latency. "
          "Time visible motion from the recorded video for that.")


def main():
    p = argparse.ArgumentParser(description="AF-1 supervised head calibration")
    p.add_argument("--gateway", default="http://127.0.0.1:5000")
    p.add_argument("--pan-joint", default="head_tilt_joint")
    p.add_argument("--tilt-joint", default="head_nod_joint")
    p.add_argument("--limit", type=float, default=20.0,
                   help="max angle to command (gentle default; raise carefully)")
    p.add_argument("--dwell", type=float, default=0.6, help="seconds per step")
    p.add_argument("--sign", action="store_true", help="run the sign test")
    p.add_argument("--sweep", action="store_true", help="run the smoothness sweep")
    p.add_argument("--film", action="store_true", help="print record cues")
    p.add_argument("--arm", action="store_true",
                   help="ACTUALLY MOVE THE HEAD (default: dry-run, no motion)")
    a = p.parse_args()
    if not (a.sign or a.sweep):
        a.sign = a.sweep = True
    print(f"AF-1 head calibration  gateway={a.gateway}  arm={a.arm}  "
          f"limit=+/-{a.limit}")
    if not a.arm:
        print("DRY-RUN: nothing will move. Re-run with --arm (human watching).")
    try:
        if a.sign:
            do_sign(a.gateway, a)
        if a.sweep:
            do_sweep(a.gateway, a)
        # always end at home
        post(a.gateway, a.pan_joint, 0.0, arm=a.arm)
        post(a.gateway, a.tilt_joint, 0.0, arm=a.arm)
        print("\nDone. Head returned to home.")
    except KeyboardInterrupt:
        print("\ninterrupted — forwarding /api/stop")
        stop(a.gateway, arm=a.arm)


if __name__ == "__main__":
    main()
