#!/usr/bin/env python3
"""AF-1 gaze reflex / smoothing service — runs ON THE ARM PI, next to the motors.

This is the actuation half of ZERO's split reflex loop (see the head-movement
plan). The ZERO Pi computes WHERE to look and streams sparse gaze setpoints over
WiFi; this service — co-located with the serial/motor gateway, so its output loop
never crosses the network — turns that stream into smooth motion and owns the
two safety guarantees the bare :5000 gateway lacks:

  1. JERK-LIMITED SMOOTHING. A dependency-free third-order profiler (bounded
     velocity, acceleration AND jerk) tracks the latest setpoint at a fixed rate,
     so even a coarse/janky setpoint stream comes out as organic motion and the
     hobby servo (which has no internal profiling) is never step-commanded.

  2. HEARTBEAT + SAFE-STATE. If no setpoint arrives within --safe-timeout, the
     neck EASES TO HOME and holds. A dropped WiFi link can never leave the head
     running a stale command, frozen mid-travel, or thrashing. {"estop":true}
     freezes immediately and forwards /api/stop.

Transport in  : UDP JSON datagrams {"t":<s>,"pan":<deg>,"tilt":<deg>,"seq":<n>}
                or {"estop":true} / {"resume":true}. Matches zero.head.driver's
                UdpSetpointDriver. UDP is deliberate: a lost packet just means we
                keep profiling toward the last target — no blocking, no backlog.
Transport out : POST http://127.0.0.1:5000/api/joint_cmd  (localhost = no WiFi in
                the loop) — the confirmed gateway interface. --pan-joint /
                --tilt-joint name the two head axes (defaults match the live
                gateway; override per your firmware).

Stdlib only. Deploy: copy to the Arm Pi, run under systemd (unit at the bottom).

    python3 af1_gaze_reflex.py --dry-run     # print, don't move (SAFE to inspect)
    python3 af1_gaze_reflex.py               # drive the neck via localhost gateway
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
import urllib.request


class JerkLimitedAxis:
    """Third-order online profiler: tracks a moving target under vmax/amax/jmax.

    Per tick: a P-law maps position error to a desired velocity (clamped to
    vmax); acceleration chases that velocity clamped to amax; and the change in
    acceleration is clamped to jmax*dt. The result is C2-continuous — no velocity
    or acceleration steps — which is what reads as smooth/organic and what the
    slew-only ZERO-side controller cannot provide on its own."""

    def __init__(self, vmax, amax, jmax, kp=6.0, home=0.0):
        self.vmax = float(vmax)
        self.amax = float(amax)
        self.jmax = float(jmax)
        self.kp = float(kp)
        self.pos = float(home)
        self.vel = 0.0
        self.acc = 0.0

    def step(self, target, dt):
        if dt <= 0:
            return self.pos
        err = float(target) - self.pos
        des_v = max(-self.vmax, min(self.vmax, self.kp * err))
        des_a = (des_v - self.vel) / dt
        des_a = max(-self.amax, min(self.amax, des_a))
        dj = max(-self.jmax * dt, min(self.jmax * dt, des_a - self.acc))
        self.acc += dj
        self.vel += self.acc * dt
        self.vel = max(-self.vmax, min(self.vmax, self.vel))
        self.pos += self.vel * dt
        return self.pos


class GazeReflex:
    def __init__(self, args):
        self.a = args
        self.pan = JerkLimitedAxis(args.vmax, args.amax, args.jmax, home=args.home_pan)
        self.tilt = JerkLimitedAxis(args.vmax, args.amax, args.jmax, home=args.home_tilt)
        self.tgt_pan = args.home_pan
        self.tgt_tilt = args.home_tilt
        self.last_rx = 0.0            # monotonic time of last setpoint
        self.estop = False
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self._sent = (None, None)

    # ── UDP receiver ─────────────────────────────────────────────────────────
    def _rx_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.a.bind, self.a.port))
        s.settimeout(0.5)
        print(f"[reflex] listening for setpoints on {self.a.bind}:{self.a.port}")
        while not self.stop.is_set():
            try:
                data, _ = s.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            with self.lock:
                if msg.get("estop"):
                    self.estop = True
                    self._forward_estop()
                    continue
                if msg.get("resume"):
                    self.estop = False
                    continue
                if "pan" in msg:
                    self.tgt_pan = self._clamp(float(msg["pan"]), self.a.limit)
                if "tilt" in msg:
                    self.tgt_tilt = self._clamp(float(msg["tilt"]), self.a.limit)
                self.last_rx = time.monotonic()
        s.close()

    # ── control loop ─────────────────────────────────────────────────────────
    def _ctrl_loop(self):
        dt = 1.0 / self.a.rate
        while not self.stop.is_set():
            t0 = time.monotonic()
            with self.lock:
                estop = self.estop
                stale = (self.last_rx == 0.0
                         or (t0 - self.last_rx) > self.a.safe_timeout)
                tp, tt = self.tgt_pan, self.tgt_tilt
            if estop:
                # freeze: stop profiling, hold last commanded pose
                pass
            else:
                if stale:
                    # heartbeat lost → ease to home (the safe state)
                    tp, tt = self.a.home_pan, self.a.home_tilt
                p = self.pan.step(tp, dt)
                q = self.tilt.step(tt, dt)
                self._emit(p, q)
            wait = dt - (time.monotonic() - t0)
            if wait > 0:
                self.stop.wait(wait)

    # ── output (localhost gateway) ───────────────────────────────────────────
    def _emit(self, pan, tilt):
        sp, st = self._sent
        if sp is not None and abs(pan - sp) < self.a.deadband and abs(tilt - st) < self.a.deadband:
            return
        self._sent = (pan, tilt)
        if self.a.dry_run:
            print(f"[reflex] pan={pan:6.2f} tilt={tilt:6.2f}")
            return
        self._post(self.a.pan_joint, pan)
        # The nod servo has a physical range [nod_min, nod_max] (gateway servo
        # degrees) and rests at nod_home. ZERO's tilt is symmetric about 0 (home);
        # map it onto that range and convert to the gateway's angle_deg
        # (servo = 90 + angle_deg), so the nod never drives past its stops and
        # rests where it actually looks at the person.
        servo = max(self.a.nod_min, min(self.a.nod_max, self.a.nod_home + tilt))
        self._post(self.a.tilt_joint, servo - 90.0)

    def _post(self, joint, deg):
        body = json.dumps({"name": joint, "angle_deg": deg,
                           "angle_rad": deg * math.pi / 180.0}).encode()
        req = urllib.request.Request(self.a.gateway + "/api/joint_cmd", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            urllib.request.urlopen(req, timeout=0.5).close()
        except Exception as e:
            print(f"[reflex] gateway post failed ({joint}={deg:.1f}): {e}")

    def _forward_estop(self):
        if self.a.dry_run:
            print("[reflex] ESTOP")
            return
        try:
            urllib.request.urlopen(
                urllib.request.Request(self.a.gateway + "/api/stop", data=b"",
                                       method="POST"), timeout=0.5).close()
        except Exception as e:
            print(f"[reflex] estop forward failed: {e}")

    @staticmethod
    def _clamp(v, lim):
        return lim if v > lim else (-lim if v < -lim else v)

    def run(self):
        rx = threading.Thread(target=self._rx_loop, daemon=True)
        rx.start()
        try:
            self._ctrl_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()


def build_argparser():
    p = argparse.ArgumentParser(description="AF-1 gaze reflex/smoothing service")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--gateway", default="http://127.0.0.1:5000",
                   help="local motor gateway base URL")
    p.add_argument("--pan-joint", default="head_tilt_joint")
    p.add_argument("--tilt-joint", default="head_nod_joint")
    # Nod servo physical travel (gateway servo degrees) and its resting home.
    p.add_argument("--nod-home", type=float, default=50.0)
    p.add_argument("--nod-min", type=float, default=50.0)
    p.add_argument("--nod-max", type=float, default=90.0)
    p.add_argument("--rate", type=float, default=50.0, help="control loop Hz")
    p.add_argument("--vmax", type=float, default=40.0, help="deg/s velocity ceiling")
    p.add_argument("--amax", type=float, default=200.0, help="deg/s^2 accel ceiling")
    p.add_argument("--jmax", type=float, default=2000.0, help="deg/s^3 jerk ceiling")
    p.add_argument("--limit", type=float, default=45.0, help="+/- soft angle limit")
    p.add_argument("--home-pan", type=float, default=0.0)
    p.add_argument("--home-tilt", type=float, default=0.0)
    p.add_argument("--deadband", type=float, default=0.2,
                   help="skip a gateway post smaller than this (deg)")
    p.add_argument("--safe-timeout", type=float, default=0.4,
                   help="ease to home if no setpoint for this many seconds")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands, do NOT drive the gateway (safe)")
    return p


# systemd unit (write to /etc/systemd/system/af1-gaze-reflex.service):
#
#   [Unit]
#   Description=AF-1 gaze reflex/smoothing service
#   After=network.target
#   [Service]
#   ExecStart=/usr/bin/python3 /home/arm/af1-firmware/af1_gaze_reflex.py
#   Restart=always
#   RestartSec=2
#   [Install]
#   WantedBy=multi-user.target


if __name__ == "__main__":
    args = build_argparser().parse_args()
    print(f"[reflex] starting (dry_run={args.dry_run} gateway={args.gateway} "
          f"rate={args.rate}Hz vmax={args.vmax} amax={args.amax} jmax={args.jmax})")
    GazeReflex(args).run()
