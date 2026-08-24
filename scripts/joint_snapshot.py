#!/usr/bin/env python3
"""Read the AF-1 gateway's joint telemetry and record it in the black box.

    .venv/bin/python scripts/joint_snapshot.py            # snapshot + print
    .venv/bin/python scripts/joint_snapshot.py --last     # print latest DB
                                                          # row per joint,
                                                          # no gateway call

Read-only against the robot: telemetry + stored offsets, no commands. Rows
land in zero_joints.sqlite (source='telemetry') next to the MotionBus's own
per-command records, so "where was every joint, when, and who moved it" is
one SQL query.

NOTE: gateway telemetry echoes the last COMMANDED angle in raw command
space; the true position of an encoderless stepper is not measurable. The
'effective' column adds the stored offset back so rows share the frame the
config envelopes use.

Caveat: head_nod_joint's driver posts raw WITHOUT offset compensation (its
2026-08-17 window was calibrated in raw terms), so its 'effective' row here
is shifted by the gateway's stored nod offset relative to what the bus
records — compare head_nod across sources in raw, not effective.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.motion.blackbox import JointAngleLog, gateway_effective  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://100.67.233.65:5000")
    ap.add_argument("--db", default="zero_joints.sqlite")
    ap.add_argument("--last", action="store_true",
                    help="print the latest recorded row per joint and exit")
    a = ap.parse_args()

    box = JointAngleLog(a.db)
    if a.last:
        rows = box.last_angles()
        if not rows:
            print("black box is empty")
            return 0
        print(f"{'joint':28s} {'angle':>8s} {'source':>10s}  recorded")
        for j, (deg, ts, src) in sorted(rows.items()):
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            print(f"{j:28s} {deg:8.1f} {src:>10s}  {when}")
        return 0

    base = a.base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/telemetry", timeout=5.0) as r:
            tel = json.loads(r.read().decode())
        with urllib.request.urlopen(f"{base}/api/calibration", timeout=5.0) as r:
            off = {k: float(v) for k, v in json.loads(r.read().decode()).items()}
    except Exception as e:
        print(f"gateway unreachable ({e})")
        return 1

    now = time.time()
    # Boot-default exclusion + effective conversion live in ONE place
    # (zero.motion.blackbox.gateway_effective), shared with the continuous
    # TelemetrySampler so the rule can never drift between the two.
    effective = gateway_effective(tel, off)
    print(f"{'joint':28s} {'raw':>8s} {'offset':>7s} {'effective':>9s}  last commanded")
    for j, v in sorted(tel.items()):
        if j == "null" or not isinstance(v, dict):
            continue
        raw = float(v.get("angle_deg", 0.0))
        o = off.get(j, 0.0)
        age = now - float(v.get("timestamp", now))
        when = (f"{age/60:.0f} min ago" if age < 5400
                else f"{age/3600:.1f} h ago")
        if j not in effective:
            print(f"{j:28s} {raw:8.1f} {o:7.1f} {'—':>9s}  boot default "
                  f"(no command since restart, {when})")
            continue
        print(f"{j:28s} {raw:8.1f} {o:7.1f} {effective[j]:9.1f}  {when}")
    n = box.snapshot(effective, "telemetry")
    print(f"\n{n} row(s) recorded to {a.db} "
          f"({len(tel) - 1 - n} boot-default row(s) excluded)")
    box.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
