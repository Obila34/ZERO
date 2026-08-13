#!/usr/bin/env python3
"""Tell ZERO's head where to look — one command.

    .venv/bin/python scripts/head_say.py "look right"          # SIM (safe, no motion)
    .venv/bin/python scripts/head_say.py --move "look right"   # REAL motion (robot on)

Runs the exact same path ZERO uses when you say it out loud: parse the phrase ->
gaze command -> HeadController -> driver. Without --move the driver is the no-op
NullDriver (prints where the head would point). With --move it drives the real
head over Path A (the :5000 gateway). If the robot/gateway is offline it says so
and falls back to SIM, so this is always safe to run.

Phrases it understands: "look right/left/up/down", "face forward", "look at Sam".
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.head.commands import parse_gaze_command
from zero.head.system import HeadSystem

GATEWAY = "http://192.168.150.183:5000"


class _Cfg:
    def __init__(self, over):
        self.d = over

    def get(self, k, d=None):
        return self.d.get(k, d)

    def resolve_path(self, k, d=None):
        return d


class _Eyes:      # minimal stand-in so the real source loop runs (no camera)
    def attention(self):
        return None

    def current_frame(self):
        return None

    def suppress_changes(self, *a, **k):
        pass

    def resettle(self):
        pass


def _gateway_up(url):
    try:
        urllib.request.urlopen(url + "/", timeout=3).close()
        return True
    except Exception:
        return False


def _telemetry(url, joint):
    try:
        import json
        with urllib.request.urlopen(url + "/api/telemetry", timeout=3) as r:
            d = json.load(r)
        return d.get(joint, {}).get("angle_deg")
    except Exception:
        return None


def _run(head, phrase, *, move, gateway, pan_joint):
    cmd = parse_gaze_command(phrase)
    if cmd is None:
        print(f'\nyou: "{phrase}"  ->  (not a head command — ZERO would just talk)')
        return
    msg = head.apply_command(cmd)
    print(f'\nyou: "{phrase}"  ->  ZERO: {msg}')
    tgt = head._cmd_target
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.5:
        pan, tilt = head.position
        col = int(round(pan)) + 45
        print(f"   t={time.monotonic()-t0:4.1f}s  pan={pan:6.1f}  tilt={tilt:6.1f}  "
              f"|{' '*max(0,min(90,col))}O|")
        if tgt and abs(pan - tgt[0]) < 0.3 and abs(tilt - tgt[1]) < 0.3:
            break
        time.sleep(0.2)
    if move:
        echoed = _telemetry(gateway, pan_joint)
        if echoed is not None:
            print(f"   gateway confirms {pan_joint} = {echoed:.1f} deg")


def main():
    p = argparse.ArgumentParser(description="tell ZERO's head where to look")
    p.add_argument("phrases", nargs="*", default=["look right"])
    p.add_argument("--move", action="store_true",
                   help="drive the REAL head directly via the gateway (Path A)")
    p.add_argument("--udp", action="store_true",
                   help="drive via the Arm-side reflex service (jerk-limited + "
                        "safe-state). REAL motion; needs af1_gaze_reflex.py running")
    p.add_argument("--gateway", default=GATEWAY)
    p.add_argument("--home", action="store_true", help="return to center at the end")
    a = p.parse_args()
    phrases = a.phrases or ["look right"]

    driver = "null"
    pan_joint = "head_tilt_joint"
    if a.udp:
        driver = "udp"
        print("REAL motion via the Arm-side reflex service (UDP setpoints, "
              "jerk-limited + safe-state) — watch the robot.")
    elif a.move:
        if _gateway_up(a.gateway):
            driver = "http"
            print(f"REAL motion: driving the head via {a.gateway} — watch the robot.")
        else:
            print(f"robot/gateway OFFLINE at {a.gateway} — power the robot on to "
                  f"move for real. Running in SIM for now.")
    if driver == "null":
        print("SIM mode: nothing physical moves; the numbers show where the head "
              "would point.")

    cfg = _Cfg({"head.enabled": True, "head.driver": driver,
                "head.max_speed_dps": 36.0,
                "head.gateway.base_url": a.gateway,
                "head.gateway.pan_joint": pan_joint,
                "head.gateway.tilt_joint": "head_nod_joint",
                "head.setpoint.host": "192.168.150.183",
                "head.setpoint.port": 8099})
    head = HeadSystem(cfg, eyes=_Eyes())
    head.start()
    try:
        for phrase in phrases:
            _run(head, phrase, move=(driver in ("http","udp")), gateway=a.gateway,
                 pan_joint=pan_joint)
            time.sleep(0.4)
        if a.home:
            _run(head, "face forward", move=(driver in ("http","udp")), gateway=a.gateway,
                 pan_joint=pan_joint)
    finally:
        head.stop()
    print("\ndone.")


if __name__ == "__main__":
    main()
