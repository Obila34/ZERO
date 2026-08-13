#!/usr/bin/env python3
"""SUPERVISED nod (tilt) calibration — find the servo's true safe envelope.

The nod is a hobby PWM servo behind an unverified linkage: it can buzz, draw
stall current, strip gears or slam a hard stop. This script therefore refuses
to do anything clever. It moves in SMALL RELATIVE STEPS ONLY (default 2 servo
degrees, hard-capped at 3), one per keypress, with a human watching the robot
and answering questions. There is no sweep mode. Run it while sitting next to
ZERO, hand on the power switch:

    .venv/bin/python scripts/nod_calibrate.py            # uses config base_url
    .venv/bin/python scripts/nod_calibrate.py --start 50 # if telemetry is empty

What it produces: the servo-degree min / max / home you observed, the direction
sign, and the exact YAML to paste into config.yaml (both the driver's servo
clamp and the controller's belief-space tilt envelope).

Commands at the prompt:
    +  / <enter>  step up   by --step servo degrees
    -             step down by --step servo degrees
    b             go back to the previous position (undo the last step)
    min / max / home   mark the current position as that calibration point
    where         re-print the current commanded position
    park          return to the marked home (or start) in small steps and exit
    q             quit WITHOUT parking (servo holds where it is)

STOP IMMEDIATELY (press 'b', then 'park') at the FIRST sign of buzzing,
straining, binding, or the linkage reaching a mechanical stop.

Every command is an absolute servo target derived from small relative steps, so
a lost keystroke can never cause a large move. All posts are logged to
calibration_nod_log.txt with timestamps.
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

STEP_HARD_CAP = 3.0          # servo degrees per keypress, never more
SERVO_RAIL_LO = 15.0         # absolute rails this script will never cross,
SERVO_RAIL_HI = 165.0        # even if you ask it to


def post_servo(base: str, servo_deg: float, log) -> bool:
    """Command the nod to an absolute servo angle via the gateway's mapping
    (servo = 90 + angle_deg with a zero offset)."""
    angle = servo_deg - 90.0
    body = json.dumps({"name": "head_nod_joint", "angle_deg": angle,
                       "angle_rad": angle * math.pi / 180.0}).encode()
    req = urllib.request.Request(f"{base}/api/joint_cmd", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        urllib.request.urlopen(req, timeout=2.0).close()
        log.write(f"{time.time():.2f} servo={servo_deg:.1f}\n")
        log.flush()
        return True
    except Exception as e:
        print(f"  !! POST failed: {e}")
        return False


def telemetry(base: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base}/api/telemetry", timeout=2.0) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=None, help="gateway base URL")
    ap.add_argument("--step", type=float, default=2.0,
                    help="servo degrees per step (max %.0f)" % STEP_HARD_CAP)
    ap.add_argument("--start", type=float, default=None,
                    help="starting servo position if telemetry has none")
    a = ap.parse_args()

    base = a.base_url
    if base is None:
        try:
            from zero.config import load_config
            base = load_config().get("head.gateway.base_url")
        except Exception:
            pass
    if not base:
        print("No gateway base URL (pass --base-url).")
        return 1
    base = base.rstrip("/")
    step = min(abs(a.step), STEP_HARD_CAP)

    print("=" * 72)
    print("SUPERVISED NOD CALIBRATION — read this before touching anything")
    print("=" * 72)
    print(f"Gateway: {base}")
    print("* The robot's head WILL move, a couple of degrees per keypress.")
    print("* You must be WATCHING THE ROBOT, not this terminal.")
    print("* Stop ('b' then 'park') at the FIRST buzz/strain/bind/hard stop.")
    print("* Note: the gateway only echoes commands — it cannot feel strain.")
    print("  Your eyes and ears are the only sensors in this loop.")
    ack = input("\nType exactly 'I am watching the robot' to begin: ").strip()
    if ack.lower() != "i am watching the robot":
        print("Not confirmed — nothing was moved.")
        return 1

    tel = telemetry(base)
    if tel is None:
        print("Gateway did not answer /api/telemetry — is it running?")
        return 1
    last = tel.get("head_nod_joint")
    if last is not None and a.start is None:
        cur = 90.0 + float(last.get("angle_deg", 0.0))
        print(f"\nTelemetry says the last COMMANDED nod was servo {cur:.1f} "
              "(command echo, not a sensor).")
    elif a.start is not None:
        cur = float(a.start)
        print(f"\nStarting from your --start value: servo {cur:.1f}.")
    else:
        print("\nTelemetry has no nod entry (gateway restarted since the last "
              "command). If the head was parked at the old park (servo 50), "
              "rerun with --start 50. If you don't know where the servo is, "
              "look at the head and estimate before commanding anything.")
        return 1
    cur = max(SERVO_RAIL_LO, min(SERVO_RAIL_HI, cur))

    print(f"\nFirst command re-asserts servo {cur:.1f} (should cause NO or "
          "minimal movement). Watch the head.")
    if input("Send it? [y/N] ").strip().lower() != "y":
        return 1

    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "calibration_nod_log.txt"), "a")
    log.write(f"# session {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if not post_servo(base, cur, log):
        return 1
    moved = input("Did the head move more than a twitch? [y/N] ").strip().lower()
    if moved == "y":
        print("Then the assumed position was wrong. The current commanded "
              f"value ({cur:.1f}) is now truth; continue from here, carefully.")

    marks: dict[str, float] = {}
    hist: list[float] = [cur]
    print(f"\nStepping {step:.1f} servo-deg per command. Marks: min/max/home. "
          "'park' to finish.")
    while True:
        c = input(f"[servo {cur:.1f}] +/-/b/min/max/home/where/park/q > ").strip().lower()
        if c in ("", "+"):
            nxt = cur + step
        elif c == "-":
            nxt = cur - step
        elif c == "b":
            if len(hist) < 2:
                print("  nothing to undo"); continue
            hist.pop(); nxt = hist[-1]
            hist.pop()          # will be re-appended below
        elif c in ("min", "max", "home"):
            marks[c] = cur
            print(f"  marked {c} = servo {cur:.1f}")
            continue
        elif c == "where":
            print(f"  commanded servo {cur:.1f}; marks: " +
                  (", ".join(f"{k}={v:.1f}" for k, v in marks.items()) or "none"))
            continue
        elif c == "park":
            tgt = marks.get("home", hist[0])
            print(f"  easing to {tgt:.1f} in {step:.1f}-deg steps…")
            while abs(cur - tgt) > 0.5:
                cur += math.copysign(min(step, abs(tgt - cur)), tgt - cur)
                post_servo(base, cur, log)
                time.sleep(0.35)
            break
        elif c == "q":
            print("  leaving the servo where it is.")
            break
        else:
            print("  ?"); continue
        if not SERVO_RAIL_LO <= nxt <= SERVO_RAIL_HI:
            print(f"  refused: {nxt:.1f} is outside the script's hard rails "
                  f"[{SERVO_RAIL_LO:.0f}, {SERVO_RAIL_HI:.0f}]")
            continue
        lo = marks.get("min"); hi = marks.get("max")
        if lo is not None and nxt < lo - 0.01:
            print(f"  refused: below your marked min ({lo:.1f}). Unmark by "
                  "re-marking if you really mean it."); continue
        if hi is not None and nxt > hi + 0.01:
            print(f"  refused: above your marked max ({hi:.1f})."); continue
        if post_servo(base, nxt, log):
            cur = nxt
            hist.append(cur)

    log.close()
    if not {"min", "max", "home"} <= marks.keys():
        print("\nCalibration incomplete (need min, max AND home marked) — "
              "no config emitted. Nothing was written.")
        return 0

    smin, smax, shome = marks["min"], marks["max"], marks["home"]
    if not smin < shome < smax:
        print(f"\nMarks look inconsistent (min {smin:.1f}, home {shome:.1f}, "
              f"max {smax:.1f}) — fix and rerun.")
        return 1
    up = input("\nAt HIGHER servo values, was the head looking UP or DOWN? "
               "[up/down] ").strip().lower()
    sign = 1.0 if up == "up" else -1.0
    # Driver posts angle = clamp(sign*tilt + offset, min, max); servo = 90+angle.
    off = shome - 90.0
    amin, amax = smin - 90.0, smax - 90.0
    # Belief-space tilt window: tilt = sign * (servo - shome).
    t1, t2 = sign * (smin - shome), sign * (smax - shome)
    tmin, tmax = min(t1, t2), max(t1, t2)
    print("\nPaste into config.yaml:\n")
    print("  head:")
    print(f"    tilt_min_deg: {tmin:.1f}")
    print(f"    tilt_max_deg: {tmax:.1f}")
    print("    gateway:")
    print(f"      nod_sign: {sign:.0f}")
    print(f"      nod_offset_deg: {off:.1f}")
    print(f"      nod_min_deg: {amin:.1f}")
    print(f"      nod_max_deg: {amax:.1f}")
    print("      drive_nod: true")
    print("\nThen verify with small typed commands ('look up 5 degrees') in "
          "`python -m zero.main --text` BEFORE enabling track_tilt or "
          "head.hand.tilt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
