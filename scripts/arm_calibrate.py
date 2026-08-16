#!/usr/bin/env python3
"""SUPERVISED arm-joint calibration — one joint at a time, small steps.

Same discipline as nod_calibrate.py, generalised to any gateway joint. You
name a joint, the script steps it 2 degrees per keypress while you watch and
listen, you mark min/max/home, and it prints the arms.joints YAML entry.

    .venv/bin/python scripts/arm_calibrate.py right_wrist_joint --start 0

SERVO joints (wrists/fingers): command space is angle_deg; the gateway adds
its saved calibration offset before writing the servo, so 0 is the gateway's
neutral. Start at 0 unless you know better.

STEPPER joints (elbows/biceps/shoulders): DANGEROUS — 160:1 gearing, no
encoders, and position zero is WHEREVER THE ARM WAS WHEN THE NANO BOOTED.
The script refuses steppers unless you pass --stepper, and then insists the
arm is at its natural rest pose (= zero) before it moves anything.

STOP ('b', then 'park') at the FIRST buzz, strain, bind or hard stop. The
gateway cannot feel any of that — you are the only sensor.
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

STEP_HARD_CAP = 3.0
ANGLE_RAIL = 150.0          # |angle_deg| this script will never cross
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "calibration_arm_log.txt")


def post(base: str, joint: str, deg: float, log) -> bool:
    body = json.dumps({"name": joint, "angle_deg": deg,
                       "angle_rad": deg * math.pi / 180.0}).encode()
    req = urllib.request.Request(f"{base}/api/joint_cmd", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        urllib.request.urlopen(req, timeout=2.0).close()
        log.write(f"{time.time():.2f} {joint} angle={deg:.1f}\n")
        log.flush()
        return True
    except Exception as e:
        print(f"  !! POST failed: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("joint", help="gateway joint name, e.g. right_wrist_joint")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--start", type=float, default=0.0,
                    help="starting angle_deg (default 0 = gateway neutral)")
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--stepper", action="store_true",
                    help="required acknowledgement to touch a stepper joint")
    a = ap.parse_args()

    if a.joint in STEPPER_JOINTS and not a.stepper:
        print(f"{a.joint} is a STEPPER (geared, no encoders, boot-relative "
              "zero). Re-run with --stepper only if the arm is at its rest "
              "pose and you understand the risk.")
        return 1

    base = a.base_url
    if base is None:
        try:
            from zero.config import load_config
            cfg = load_config()
            base = cfg.get("arms.gateway.base_url",
                           cfg.get("head.gateway.base_url"))
        except Exception:
            pass
    if not base:
        print("No gateway base URL (pass --base-url).")
        return 1
    base = base.rstrip("/")
    step = min(abs(a.step), STEP_HARD_CAP)

    print("=" * 72)
    print(f"SUPERVISED CALIBRATION — {a.joint}")
    print("=" * 72)
    print(f"Gateway {base}; {step:.1f} deg per keypress; hard rails "
          f"+/-{ANGLE_RAIL:.0f}. Watch and LISTEN to the joint, not the screen.")
    if a.joint in STEPPER_JOINTS:
        print("STEPPER: confirm the arm is at its natural rest pose NOW —")
        print("this pose is zero, and every command is relative to it.")
    ack = input("Type exactly 'I am watching the robot' to begin: ").strip()
    if ack.lower() != "i am watching the robot":
        print("Not confirmed — nothing was moved.")
        return 1

    cur = float(a.start)
    log = open(LOG, "a")
    log.write(f"# session {time.strftime('%Y-%m-%d %H:%M:%S')} joint={a.joint}\n")
    print(f"\nFirst command asserts angle {cur:.1f} — watch for movement.")
    if input("Send it? [y/N] ").strip().lower() != "y":
        return 1
    if not post(base, a.joint, cur, log):
        return 1

    marks: dict[str, float] = {}
    hist = [cur]
    while True:
        c = input(f"[{a.joint} {cur:+.1f}] +/-/b/min/max/home/park/q > ").strip().lower()
        if c in ("", "+"):
            nxt = cur + step
        elif c == "-":
            nxt = cur - step
        elif c == "b":
            if len(hist) < 2:
                print("  nothing to undo"); continue
            hist.pop(); nxt = hist[-1]; hist.pop()
        elif c in ("min", "max", "home"):
            marks[c] = cur
            print(f"  marked {c} = {cur:+.1f}")
            continue
        elif c == "park":
            tgt = marks.get("home", hist[0])
            while abs(cur - tgt) > 0.5:
                cur += math.copysign(min(step, abs(tgt - cur)), tgt - cur)
                post(base, a.joint, cur, log)
                time.sleep(0.35)
            break
        elif c == "q":
            print("  leaving the joint where it is."); break
        else:
            print("  ?"); continue
        if abs(nxt) > ANGLE_RAIL:
            print(f"  refused: outside the +/-{ANGLE_RAIL:.0f} rail"); continue
        lo, hi = marks.get("min"), marks.get("max")
        if lo is not None and nxt < lo - 0.01:
            print(f"  refused: below your marked min ({lo:+.1f})"); continue
        if hi is not None and nxt > hi + 0.01:
            print(f"  refused: above your marked max ({hi:+.1f})"); continue
        if post(base, a.joint, nxt, log):
            cur = nxt
            hist.append(cur)

    log.close()
    if {"min", "max", "home"} <= marks.keys():
        print("\nPaste into config.yaml under arms.joints:\n")
        print(f"    {a.joint}: {{min: {marks['min']:.1f}, "
              f"max: {marks['max']:.1f}, home: {marks['home']:.1f}}}")
        if a.joint in STEPPER_JOINTS:
            print("\n(stepper: also set arms.allow_steppers: true — and "
                  "remember these numbers are only valid while boot pose = "
                  "the rest pose you calibrated from)")
    else:
        print("\nIncomplete (need min, max AND home) — nothing emitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
