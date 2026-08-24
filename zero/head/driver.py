"""Head motion sinks — where HeadController's per-tick (pan, tilt) command goes.

The controller is transport-agnostic: it calls `driver.send(head_x, head_y)` at
its tick rate and never blocks on the result. This module supplies the concrete
sinks and a config-driven factory. The DEFAULT is NullDriver — it moves nothing,
so the whole head subsystem can be built, wired and feel-tested on the ZERO Pi
with zero risk to real metal. Physical drivers are strictly opt-in via
`head.driver` in config.

Drivers, by `head.driver`:
  null    — NullDriver: log + remember the last command; no hardware. DEFAULT.
  http    — HttpGatewayDriver: POST /api/joint_cmd to the Arm Pi's :5000 gateway.
            Direct-LAN path. Rate-limited + per-axis deadband so the stdlib HTTP
            server is never hammered. MOVES THE ROBOT.
  udp     — UdpSetpointDriver: fire-and-forget setpoint datagrams to the Arm-side
            reflex/smoothing service (the recommended architecture). The Arm side
            does the jerk-limited profiling + heartbeat/safe-state. MOVES THE ROBOT.

Design notes:
  * stdlib only (socket/json/urllib/threading) — importing zero.head must never
    pull in cv2, requests, or the identity stack.
  * A physical send must NEVER raise into the control loop; every driver swallows
    and logs its own transport errors (HeadController also guards its call site).
  * Axis→joint mapping is configurable because the two known firmware stacks
    disagree on names (see plan Part 1.1); nothing is hard-coded to one stack.
"""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request

from zero.utils.logging import get_logger

log = get_logger("head.driver")


class HeadDriver:
    """Interface: a sink for (head_x=pan°, head_y=tilt/nod°), signed degrees."""

    moves_hardware: bool = False

    def send(self, head_x: float, head_y: float) -> None:  # pragma: no cover - iface
        raise NotImplementedError

    def estop(self) -> None:
        """Best-effort hard stop. Default: nothing to stop."""

    def close(self) -> None:
        """Release any transport resources."""

    @property
    def last(self) -> tuple[float, float]:
        return getattr(self, "_last", (0.0, 0.0))


class NullDriver(HeadDriver):
    """Moves nothing. Records the last command and optionally traces it.

    This is the default sink and the one phase 1 runs on: the entire ZERO-side
    reflex/attention/social-gaze stack is exercised and its emitted command
    stream is captured (for the smoothness harness) without any actuator.
    """

    moves_hardware = False

    def __init__(self, *, trace: bool = False):
        self._last = (0.0, 0.0)
        self._trace = bool(trace)
        self._n = 0

    def send(self, head_x: float, head_y: float) -> None:
        self._last = (float(head_x), float(head_y))
        self._n += 1
        if self._trace and (self._n % 25 == 0):  # ~1 Hz at a 25 Hz tick
            log.info("null-head: pan=%.1f tilt=%.1f", head_x, head_y)

    @property
    def count(self) -> int:
        return self._n


class HttpGatewayDriver(HeadDriver):
    """POST /api/joint_cmd to the Arm Pi's direct-LAN gateway (:5000).

    MOVES THE ROBOT. Opt-in only. Rate-limited to `max_hz` and gated by a
    per-axis `deadband_deg` so a 25 Hz controller does not flood the stdlib
    ThreadingHTTPServer or the two Arduino serial links. Sends are dispatched on
    a background worker so the control tick never blocks on WiFi/HTTP.
    """

    moves_hardware = True

    def __init__(self, *, base_url: str, pan_joint: str, tilt_joint: str,
                 max_hz: float = 15.0, deadband_deg: float = 0.4,
                 timeout_s: float = 0.5, nod_offset_deg: float = 0.0,
                 nod_min_deg: float = -90.0, nod_max_deg: float = 90.0,
                 drive_nod: bool = True, max_jump_deg: float = 12.0,
                 nod_sign: float = 1.0):
        self._base = base_url.rstrip("/")
        self._pan_joint = pan_joint
        self._tilt_joint = tilt_joint
        # The nod servo has a restricted travel and a home that isn't the joint's
        # 0. gateway servo = 90 + angle_deg, so posting (sign*tilt + offset)
        # clamped to [nod_min, nod_max] parks the nod at its true home and never
        # drives it past its physical stop. nod_sign flips the axis if
        # increasing servo turns out to look DOWN (set from calibration; the
        # controller convention is +tilt = up). (Default offset 0 = no remap.)
        self._nod_sign = float(nod_sign)
        self._nod_offset = float(nod_offset_deg)
        self._nod_min = float(nod_min_deg)
        self._nod_max = float(nod_max_deg)
        self._drive_nod = bool(drive_nod)   # False = never post the nod at all
        self._min_interval = 1.0 / max(1.0, float(max_hz))
        self._deadband = float(deadband_deg)
        self._timeout = float(timeout_s)
        # Bound how far a single post may move the hardware from the last
        # ACKNOWLEDGED pose. The controller's slew limit lives in belief space
        # only; after an outage the next absolute post would otherwise carry the
        # whole accumulated error and the neck would whip. <=0 disables.
        self._max_jump = float(max_jump_deg)
        self._last = (0.0, 0.0)
        self._sent = (float("nan"), float("nan"))
        self._last_t = 0.0
        # Link health: consecutive failed posts / last success, surfaced via
        # .healthy so status() can tell "moving" from "unplugged" (audit C2).
        self._fails = 0
        self._ok_t = 0.0
        self._lock = threading.Lock()
        self._pending: tuple[float, float] | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker = threading.Thread(target=self._run, name="head-http", daemon=True)
        self._worker.start()

    def send(self, head_x: float, head_y: float) -> None:
        self._last = (float(head_x), float(head_y))
        with self._lock:
            self._pending = self._last
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                target = self._pending
                self._pending = None
            if target is None:
                continue
            now = time.monotonic()
            if now - self._last_t < self._min_interval:
                # too soon — requeue and coalesce
                with self._lock:
                    if self._pending is None:
                        self._pending = target
                self._wake.set()
                time.sleep(self._min_interval - (now - self._last_t))
                continue
            px, py = target
            sx, sy = self._sent
            first = sx != sx          # nothing acknowledged yet (NaN sentinel)
            # Bounded hop: never ask the hardware to move more than max_jump
            # from the last acknowledged pose in one post. Requeue the real
            # target so the worker keeps walking until it arrives.
            if not first and self._max_jump > 0:
                dx, dy = px - sx, py - sy
                m = max(abs(dx), abs(dy))
                if m > self._max_jump:
                    s = self._max_jump / m
                    px, py = sx + dx * s, sy + dy * s
                    with self._lock:
                        if self._pending is None:
                            self._pending = target
                    self._wake.set()
            # Per-axis gating: only post an axis that actually moved (the nod
            # rides a separate serial bus — don't spam it with pan twitches).
            pan_due = first or abs(px - sx) >= self._deadband
            nod_due = self._drive_nod and (first or abs(py - sy) >= self._deadband)
            if not pan_due and not nod_due:
                continue
            self._last_t = now
            ok = True
            if pan_due:
                ok = self._post(self._pan_joint, px) and ok
            if nod_due:
                nod = max(self._nod_min,
                          min(self._nod_max, self._nod_sign * py + self._nod_offset))
                ok = self._post(self._tilt_joint, nod) and ok
            if ok:
                # Acknowledge only on success, so a failed post is retried
                # instead of silently skipped (audit H1).
                self._sent = (px, py)
                if self._fails >= 3:
                    log.warning("head gateway recovered after %d failed posts",
                                self._fails)
                self._fails = 0
                self._ok_t = time.monotonic()
            else:
                self._fails += 1
                if self._fails == 3:
                    log.warning("head gateway unreachable (%s) — will keep "
                                "retrying; the head is NOT moving", self._base)
                with self._lock:
                    if self._pending is None:
                        self._pending = target
                self._wake.set()
                # Back off while the link is down so a dead gateway isn't
                # hammered at full rate (timeouts already slow each attempt).
                time.sleep(min(2.0, 0.2 * self._fails))

    @property
    def healthy(self) -> bool:
        """False once several consecutive posts have failed."""
        return self._fails < 3

    def _post(self, joint: str, deg: float) -> bool:
        import math
        body = json.dumps({
            "name": joint, "angle_deg": deg,
            "angle_rad": deg * math.pi / 180.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/api/joint_cmd", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=self._timeout).close()
            return True
        except Exception as e:   # never raise into the loop
            log.debug("gateway post failed (%s=%.1f): %s", joint, deg, e)
            return False

    def estop(self) -> None:
        try:
            req = urllib.request.Request(f"{self._base}/api/stop", data=b"",
                                         method="POST")
            urllib.request.urlopen(req, timeout=self._timeout).close()
        except Exception as e:
            log.debug("gateway estop failed: %s", e)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()


class UdpSetpointDriver(HeadDriver):
    """Fire-and-forget gaze setpoints to the Arm-side reflex service (recommended).

    MOVES THE ROBOT (via the Arm service). Each send is one small JSON datagram
    {t, pan, tilt, seq}; the Arm side owns jerk-limited profiling and the
    heartbeat/safe-state timeout. UDP is deliberate: a dropped packet just means
    the Arm keeps executing its current profile toward the last target — no
    blocking, no backpressure into the 25 Hz control loop. The stream itself is
    the heartbeat.
    """

    moves_hardware = True

    def __init__(self, *, host: str, port: int):
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._last = (0.0, 0.0)
        self._seq = 0

    def send(self, head_x: float, head_y: float) -> None:
        self._last = (float(head_x), float(head_y))
        self._seq += 1
        pkt = json.dumps({
            "t": time.time(), "pan": round(head_x, 3), "tilt": round(head_y, 3),
            "seq": self._seq,
        }).encode("utf-8")
        try:
            self._sock.sendto(pkt, self._addr)
        except Exception as e:   # never raise into the loop (e.g. buffer full)
            log.debug("setpoint send failed: %s", e)

    def estop(self) -> None:
        try:
            self._sock.sendto(b'{"estop":true}', self._addr)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


def make_driver(cfg) -> HeadDriver:
    """Build the head motion sink from config. Defaults to NullDriver (no motion).

    `cfg` is the project Config (dotted .get). Reads:
      head.driver            null | http | udp        (default: null)
      head.trace             bool                      (null-driver ~1 Hz trace)
      head.gateway.base_url  http://192.168.150.183:5000
      head.gateway.pan_joint / tilt_joint             (firmware joint names)
      head.gateway.max_hz / deadband_deg / timeout_s
      head.setpoint.host / port                        (Arm reflex service)
    """
    kind = str(cfg.get("head.driver", "null")).lower()
    if kind in ("", "null", "none", "off", "mock"):
        return NullDriver(trace=bool(cfg.get("head.trace", False)))
    if kind == "bus":
        # Gaze joins the MotionBus: same nod mapping and walking behaviour as
        # the http driver, but arbitrated with gestures and sign on one clock
        # and one shared e-stop. Whether the bus moves metal is motion.driver's
        # call (motion.driver: http), not this one.
        from zero.motion.drivers import BusHeadDriver, get_bus

        bus = get_bus(cfg)
        drv = BusHeadDriver(
            bus,
            pan_joint=cfg.get("head.gateway.pan_joint", "head_tilt_joint"),
            tilt_joint=cfg.get("head.gateway.tilt_joint", "head_nod_joint"),
            limit_deg=float(cfg.get("head.limit_deg", 80.0)),
            deadband_deg=float(cfg.get("head.gateway.deadband_deg", 0.4)),
            max_jump_deg=float(cfg.get("head.gateway.max_jump_deg", 12.0)),
            nod_offset_deg=float(cfg.get("head.gateway.nod_offset_deg", 0.0)),
            nod_min_deg=float(cfg.get("head.gateway.nod_min_deg", -90.0)),
            nod_max_deg=float(cfg.get("head.gateway.nod_max_deg", 90.0)),
            drive_nod=bool(cfg.get("head.gateway.drive_nod", True)),
            nod_sign=float(cfg.get("head.gateway.nod_sign", 1.0)),
        )
        if bus.moves_hardware:
            log.warning("head.driver=bus + motion.driver=http — HEAD WILL "
                        "MOVE via the MotionBus")
        return drv
    if kind == "http":
        drv = HttpGatewayDriver(
            base_url=cfg.get("head.gateway.base_url", "http://192.168.150.183:5000"),
            pan_joint=cfg.get("head.gateway.pan_joint", "head_tilt_joint"),
            tilt_joint=cfg.get("head.gateway.tilt_joint", "head_nod_joint"),
            max_hz=float(cfg.get("head.gateway.max_hz", 15.0)),
            deadband_deg=float(cfg.get("head.gateway.deadband_deg", 0.4)),
            timeout_s=float(cfg.get("head.gateway.timeout_s", 0.5)),
            nod_offset_deg=float(cfg.get("head.gateway.nod_offset_deg", 0.0)),
            nod_min_deg=float(cfg.get("head.gateway.nod_min_deg", -90.0)),
            nod_max_deg=float(cfg.get("head.gateway.nod_max_deg", 90.0)),
            drive_nod=bool(cfg.get("head.gateway.drive_nod", True)),
            max_jump_deg=float(cfg.get("head.gateway.max_jump_deg", 12.0)),
            nod_sign=float(cfg.get("head.gateway.nod_sign", 1.0)),
        )
        log.warning("head.driver=http — HEAD WILL MOVE via %s", drv._base)
        return drv
    if kind == "udp":
        drv = UdpSetpointDriver(
            host=cfg.get("head.setpoint.host", "127.0.0.1"),
            port=int(cfg.get("head.setpoint.port", 8099)),
        )
        log.warning("head.driver=udp — HEAD WILL MOVE via setpoints to %s:%s",
                    drv._addr[0], drv._addr[1])
        return drv
    log.warning("unknown head.driver=%r — falling back to NullDriver", kind)
    return NullDriver()
