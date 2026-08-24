#!/usr/bin/env python3
"""SUPERVISED probe: does /api/pose_cmd apply stepper offsets like
/api/joint_cmd? — the open question gating batched stepper posts (Phase 5).

Protocol (one calibrated STEPPER joint, operator watching, hand on e-stop):
  1. joint_cmd parks the joint at a mid-range effective pose (raw = eff -
     offset). This is the known-good reference path.
  2. pose_cmd then sends the IDENTICAL raw value.
       * joint does NOT move  -> pose_cmd == joint_cmd (offsets applied the
         same way): batching steppers is safe.
       * joint steps by ~|offset| deg -> pose_cmd is offset-blind: steppers
         must STAY on joint_cmd (the MotionBus already does this).
  3. joint_cmd immediately re-parks either way.

Pick a joint whose |offset| is visible but safe mid-range (left_bicep,
offset ~-16, is ideal). NEVER probe with a large-offset joint like
right_bicep (-108).

    .venv/bin/python scripts/pose_probe.py left_bicep_joint --effective 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.arms.driver import STEPPER_JOINTS  # noqa: E402

MAX_ABS_OFFSET = 35.0     # refuse joints whose offset would jerk further


def _post(base: str, path: str, body: dict) -> bool:
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=2.0).close()
        return True
    except Exception as e:
        print(f"  !! POST {path} failed: {e}")
        return False


def joint_cmd(base: str, joint: str, raw: float) -> bool:
    return _post(base, "/api/joint_cmd",
                 {"name": joint, "angle_deg": raw,
                  "angle_rad": raw * math.pi / 180.0})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("joint")
    ap.add_argument("--base-url", default="http://100.67.233.65:5000")
    ap.add_argument("--effective", type=float, default=10.0,
                    help="mid-range effective pose to park at first")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")

    if a.joint not in STEPPER_JOINTS:
        print(f"{a.joint} is not a stepper — the probe only means anything "
              "on an offset-bearing stepper.")
        return 1
    try:
        with urllib.request.urlopen(f"{base}/api/calibration", timeout=3.0) as r:
            offsets = {k: float(v) for k, v in json.loads(r.read().decode()).items()}
    except Exception as e:
        print(f"cannot read /api/calibration ({e}) — refusing to move blind")
        return 1
    offset = offsets.get(a.joint, 0.0)
    if abs(offset) < 2.0:
        print(f"{a.joint} offset is {offset:+.1f} — too small to tell the "
              "two endpoints apart. Pick a joint with |offset| 5..35.")
        return 1
    if abs(offset) > MAX_ABS_OFFSET:
        print(f"{a.joint} offset is {offset:+.1f} — a wrong answer would "
              f"jerk that far. Refusing; probe with a smaller-offset joint.")
        return 1

    eff = float(a.effective)
    raw = eff - offset
    print("=" * 72)
    print(f"POSE_CMD OFFSET PROBE — {a.joint} (stored offset {offset:+.1f})")
    print("=" * 72)
    print(f"Step 1  joint_cmd  raw {raw:+.1f} (= effective {eff:+.1f}) — parks")
    print(f"Step 2  pose_cmd   raw {raw:+.1f} — WATCH: no move = offsets "
          f"applied; a ~{abs(offset):.0f} deg step = offset-blind")
    print("Step 3  joint_cmd re-parks either way")
    print("\nThe joint MUST already be calibrated and the arm's boot pose "
          "must equal its rest pose (same rule as arm_calibrate.py).")
    ack = input("Type exactly 'I am watching the robot' to begin: ").strip()
    if ack.lower() != "i am watching the robot":
        print("Not confirmed — nothing was moved.")
        return 1

    # Step 1: walk to the reference pose gently (3-deg hops).
    cur = 0.0
    while abs(cur - eff) > 0.5:
        cur += math.copysign(min(3.0, abs(eff - cur)), eff - cur)
        if not joint_cmd(base, a.joint, cur - offset):
            return 1
        time.sleep(0.35)
    input(f"\nParked at effective {eff:+.1f}. Eyes on the joint — press "
          "Enter to fire pose_cmd with the identical raw value...")
    if not _post(base, "/api/pose_cmd", {"joints": {a.joint: raw}}):
        return 1
    time.sleep(1.5)
    moved = input("\nDid the joint MOVE? [y/n] ").strip().lower()
    # Step 3: re-park through the trusted path regardless of the answer.
    joint_cmd(base, a.joint, eff - offset)
    time.sleep(0.5)
    print()
    if moved == "y":
        print("VERDICT: pose_cmd is OFFSET-BLIND (raw, no stored offset).")
        print("Steppers must stay on joint_cmd — the MotionBus already "
              "routes them there; do NOT add steppers to the batch set.")
    else:
        print("VERDICT: pose_cmd applies stored offsets like joint_cmd.")
        print("Batching steppers via pose_cmd is safe. Record this in "
              "docs/SIGN_LANGUAGE.md (Phase 5 checklist) with today's date.")
    print("\nEase the joint back to rest (park) with arm_calibrate.py if "
          "you moved it from rest to run this probe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
