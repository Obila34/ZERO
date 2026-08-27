#!/usr/bin/env python3
"""Arm stream probe — SUPERVISED stepper bring-up for the gesture carrier.

The arm steppers (elbow, shoulder up/down, shoulder in/out) have only
ever been driven with single point-to-point moves (voice commands, sign
stance). The gesture carrier streams a new setpoint every tick instead
— this probe is the required evidence that the geared steppers track a
slow streamed oscillation cleanly BEFORE any of that runs unsupervised.

For each registered arm joint, one at a time, with your confirmation:
a slow sine, +/-AMP degrees about home, two periods, then back to home.
Watch for: smooth motion (no chatter/stalling), correct direction
(printed before each move), and a clean return to rest.

MOVES THE ARM STEPPERS. Eyes on the robot, e-stop within reach.
zero.service must be STOPPED. Run on the Pi:

    .venv/bin/python scripts/arm_stream_probe.py            # +/-4 deg
    .venv/bin/python scripts/arm_stream_probe.py --amp 2    # gentler
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.config import load_config  # noqa: E402
from zero.motion.drivers import get_bus, reset_bus  # noqa: E402

# (joint, what a POSITIVE command means — sign-stance ground truth,
#  hardware-verified 2026-08-17)
JOINTS = [
    ("right_elbow_joint", "positive = forearm back/straighten "
                          "(bend forward is negative)"),
    ("left_elbow_joint", "positive = forearm back/straighten "
                         "(bend forward is negative)"),
    ("right_up_down_joint", "positive = raise the right arm"),
    ("left_up_down_joint", "positive = LOWER the left arm "
                           "(mirrored motor: raise is negative)"),
    ("right_in_out_joint", "positive = right arm outward from the body"),
    ("left_in_out_joint", "positive = left arm INWARD "
                          "(mirrored: outward is negative)"),
]

RATE_HZ = 25.0
PERIOD_S = 6.0          # slow: one full out-and-back every 6 s
CYCLES = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=4.0,
                    help="oscillation amplitude in degrees about home")
    a = ap.parse_args()

    cfg = load_config()
    if str(cfg.get("motion.driver")) != "http":
        print("motion.driver is not http — nothing would move. Abort.")
        return 1
    bus = get_bus(cfg)
    # Register the arm joints EXACTLY the way production does — through
    # the arms driver (envelopes, offsets, stepper walking all included).
    from zero.arms.driver import load_joints, make_arm_driver
    make_arm_driver(cfg, load_joints(cfg))
    # Encoderless steppers stay MUTE until the gateway's stored zero
    # offsets are read — and the bus only FETCHES offsets when a stepper
    # write is pending (deliberate: no idle polling of a possibly-broken
    # endpoint). Prime it by writing each stepper's HOME — zero motion,
    # the arms are already at rest — then wait for the offsets to load.
    prime = {}
    for n, _ in JOINTS:
        spec = bus.spec(n)
        if spec is not None:
            prime[n] = spec.home_deg
    if prime:
        bus.write("gesture", prime)
    deadline = time.monotonic() + 15.0
    while getattr(bus, "_offsets", None) is None \
            and time.monotonic() < deadline:
        time.sleep(0.25)
    bus.release("gesture")
    if getattr(bus, "_offsets", None) is None:
        print("stepper zero offsets never arrived from the gateway — "
              "stepper commands would stay mute. Abort (is the AF-1 "
              "gateway up?).")
        return 1
    print("stepper offsets loaded — steppers are live.")

    present = [(n, d) for n, d in JOINTS if bus.spec(n) is not None]
    if not present:
        print("no arm stepper joints registered on the bus "
              "(arms.driver / arms.allow_steppers?)")
        return 1
    print(f"{len(present)} arm joint(s) registered. Amplitude +/-{a.amp} "
          f"deg, {CYCLES} cycles of {PERIOD_S:.0f} s each.\n"
          "EYES ON THE ROBOT. E-STOP IN REACH. Ctrl-C parks and exits.\n")
    try:
        for name, direction in present:
            spec = bus.spec(name)
            lo = spec.clamp(spec.home_deg - a.amp)
            hi = spec.clamp(spec.home_deg + a.amp)
            print(f"--- {name}")
            print(f"    {direction}")
            print(f"    home {spec.home_deg:+.1f}, will move "
                  f"{lo:+.1f}..{hi:+.1f}")
            input("    Enter to move this joint (or Ctrl-C to abort)... ")
            t0 = time.monotonic()
            dt = 1.0 / RATE_HZ
            while True:
                t = time.monotonic() - t0
                if t >= CYCLES * PERIOD_S:
                    break
                delta = a.amp * math.sin(2 * math.pi * t / PERIOD_S)
                bus.write("gesture", {name: spec.clamp(
                    spec.home_deg + delta)})
                time.sleep(dt)
            bus.write("gesture", {name: spec.home_deg})
            time.sleep(1.5)
            bus.release("gesture", [name])
            ans = input("    smooth + correct direction + back at rest? "
                        "[y/n] ").strip().lower()
            print(f"    recorded: {'PASS' if ans == 'y' else 'FAIL'}\n")
    except KeyboardInterrupt:
        print("\naborted — parking all probed joints at home")
        for name, _ in present:
            spec = bus.spec(name)
            if spec is not None:
                bus.write("gesture", {name: spec.home_deg})
        time.sleep(2.0)
        bus.release("gesture")
    reset_bus()
    print("probe complete. If every joint passed, the gesture carrier "
          "may be enabled: expression.arms.enabled: true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
