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


def fetch_offsets(base: str) -> dict:
    """The gateway's stored zero-offsets (/api/calibration). It ADDS these to
    every command, so raw angle 0 is NOT 'no motion' for offset joints —
    right_elbow_joint carries offset 304, a 27,000-step landmine."""
    try:
        with urllib.request.urlopen(f"{base}/api/calibration", timeout=2.0) as r:
            return {k: float(v) for k, v in json.loads(r.read().decode()).items()}
    except Exception:
        return {}


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
    ap.add_argument("--start", type=float, default=None,
                    help="starting EFFECTIVE degrees (default: servo neutral, "
                         "or the stepper's boot-pose zero)")
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
    offset = fetch_offsets(base).get(a.joint, 0.0)
    is_stepper = a.joint in STEPPER_JOINTS
    # Work in EFFECTIVE degrees (what the hardware sees = angle + offset).
    # Steppers: effective 0 == the step counter's zero == the pose at the last
    # Nano reset, so starting there is guaranteed no-motion. Servos: the
    # offset IS the neutral servo angle, so effective `offset` is the neutral.
    if a.start is not None:
        eff = float(a.start)
    else:
        eff = 0.0 if is_stepper else offset
    if offset:
        print(f"NOTE: gateway adds a stored offset of {offset:+.1f} to this "
              f"joint; the script compensates (commands are shown in "
              f"effective degrees, hardware-true).")

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

    cur = eff
    log = open(LOG, "a")
    log.write(f"# session {time.strftime('%Y-%m-%d %H:%M:%S')} joint={a.joint}\n")
    if is_stepper:
        print("\nSTEPPER ZERO CHECK: effective 0 = the arm's pose at the last "
              "gateway restart. If the arm has been moved (cockpit or by hand) "
              "since, RESTART THE GATEWAY first so zero = the current pose.")
    print(f"\nFirst command asserts effective {cur:+.1f} — watch for movement.")
    if input("Send it? [y/N] ").strip().lower() != "y":
        return 1
    if not post(base, a.joint, cur - offset, log):
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
                post(base, a.joint, cur - offset, log)
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
        if post(base, a.joint, nxt - offset, log):
            cur = nxt
            hist.append(cur)

    log.close()
    if {"min", "max", "home"} <= marks.keys():
        # EFFECTIVE degrees, verbatim — the same space the driver clamps in
        # before subtracting the gateway offset at the wire. (This used to
        # emit raw command space, marks - offset: pasted into config, the
        # driver would subtract the offset a SECOND time and every command
        # landed `offset` degrees off — 108 deg on right_bicep. Never
        # convert here.)
        print("\nPaste into config.yaml under arms.joints (EFFECTIVE "
              "degrees — the driver handles the gateway offset):\n")
        print(f"    {a.joint}: {{min: {marks['min']:.1f}, "
              f"max: {marks['max']:.1f}, "
              f"home: {marks['home']:.1f}}}")
        # Direction: the gesture layer's convention is + = the direction the
        # spoken verb "raise"/"lift" means (URDF-positive). A mirrored motor
        # needs arms.joint_sign, not hand-edited envelopes.
        ans = input("\nDid '+' steps move the joint the way RAISE/LIFT "
                    "should? [y/n/skip] ").strip().lower()
        if ans == "n":
            print("Mirrored motor — ALSO paste under arms.joint_sign:\n")
            print(f"    {a.joint}: -1")
            print("(keep the envelope above as printed; joint_sign mirrors "
                  "the window for you)")
        if a.joint in STEPPER_JOINTS:
            print("\n(stepper: also set arms.allow_steppers: true — and "
                  "remember these numbers are only valid while boot pose = "
                  "the rest pose you calibrated from)")
    else:
        print("\nIncomplete (need min, max AND home) — nothing emitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
