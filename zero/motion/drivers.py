"""Bus-backed drivers — the existing subsystems' plugs into the MotionBus.

HeadController and ArmSystem keep their tuned control logic untouched; only
the transport under them changes. Each driver here implements the same sink
interface as the http driver it replaces (send / estop / close / healthy /
last), translates to joint targets exactly the way the http driver did, and
writes them onto its track. The bus does the rest: arbitration, clamping,
max_jump walking, batching, retry, the shared e-stop.

The shared bus is created once per process by get_bus(cfg); every driver and
the sign engine attach to the same instance — that is the whole point.
"""
from __future__ import annotations

import threading

from zero.arms.hands import HAND_JOINTS
from zero.motion.bus import BusJoint, MotionBus
from zero.motion.transport import HttpTransport, NullTransport
from zero.utils.logging import get_logger

log = get_logger("motion.drivers")

_bus: MotionBus | None = None
_sampler = None
_bus_lock = threading.Lock()


def get_bus(cfg) -> MotionBus:
    """The process-wide MotionBus, created on first use from `motion.*`
    config. motion.driver: null (default — records, moves nothing) | http
    (MOVES THE ROBOT via the AF-1 gateway)."""
    global _bus
    with _bus_lock:
        if _bus is None:
            kind = str(cfg.get("motion.driver", "null")).lower()
            if kind == "http":
                transport = HttpTransport(
                    base_url=cfg.get("motion.gateway.base_url",
                                     cfg.get("arms.gateway.base_url",
                                             "http://100.67.233.65:5000")),
                    timeout_s=float(cfg.get("motion.gateway.timeout_s", 0.7)))
                log.warning("motion.driver=http — the MotionBus MOVES THE "
                            "ROBOT via %s", transport._base)
            else:
                transport = NullTransport()
            blackbox = None
            if bool(cfg.get("motion.blackbox.enabled", True)):
                try:
                    from zero.motion.blackbox import JointAngleLog

                    blackbox = JointAngleLog(
                        cfg.get("motion.blackbox.db_path",
                                "zero_joints.sqlite"))
                except Exception as e:   # recording must never block motion
                    log.warning("joint black box unavailable: %s", e)
            _bus = MotionBus(transport,
                             rate_hz=float(cfg.get("motion.rate_hz", 30.0)),
                             blackbox=blackbox)
            # Continuous telemetry sampling (http only — with a null
            # transport there is no robot to sample): catches motion ZERO
            # did not command, e.g. someone driving from the AF-1 cockpit.
            global _sampler
            poll_s = float(cfg.get("motion.blackbox.poll_s", 60.0))
            if kind == "http" and blackbox is not None and poll_s > 0:
                from zero.motion.blackbox import TelemetrySampler

                _sampler = TelemetrySampler(
                    blackbox, transport._base, period_s=poll_s)
        return _bus


def reset_bus() -> None:
    """Tear down the shared bus (tests, shutdown)."""
    global _bus, _sampler
    with _bus_lock:
        if _sampler is not None:
            _sampler.stop()
            _sampler = None
        if _bus is not None:
            _bus.close()
            _bus = None


class BusArmDriver:
    """ArmSystem's sink onto the bus. Same interface as HttpArmDriver; every
    joint it may drive is registered up front from its calibrated JointSpec,
    hand joints marked batchable (the firmware's own fingerspell path proves
    pose_cmd for exactly that set)."""

    def __init__(self, bus: MotionBus, joints: dict, *, track: str = "gesture",
                 deadband_deg: float = 0.3):
        self._bus = bus
        self._track = track
        self._names = set(joints)
        for name, spec in joints.items():
            bus.register(BusJoint(
                name, min_deg=spec.min_deg, max_deg=spec.max_deg,
                home_deg=spec.home_deg, deadband_deg=deadband_deg,
                # Encoderless steppers keep the effective-degrees convention:
                # mute until the gateway's stored offsets are known — and
                # they WALK (max_jump) so an outage recovery never whips the
                # whole accumulated error into a geared arm in one hop
                # (audit 2026-08-25 #7). Hand servos stay unwalked: they are
                # fast, absolute, and signing needs their snap.
                use_offset=spec.is_stepper,
                max_jump_deg=16.0 if spec.is_stepper else 0.0,
                batch=name in HAND_JOINTS))

    @property
    def moves_hardware(self) -> bool:
        return self._bus.moves_hardware

    @property
    def healthy(self) -> bool:
        return self._bus.healthy

    @property
    def last(self) -> dict[str, float]:
        return {k: v for k, v in self._bus.last.items() if k in self._names}

    def send(self, pose: dict[str, float]) -> None:
        self._bus.write(self._track, pose)

    def estop(self) -> None:
        self._bus.estop()

    def resume(self) -> None:
        # NOTHING called MotionBus.resume() before this existed: one estop
        # froze the whole bus until process restart while the tool kept
        # claiming success (audit sign #2).
        self._bus.resume()

    @property
    def estopped(self) -> bool:
        return self._bus.estopped

    def close(self) -> None:
        self._bus.release(self._track)


class BusHeadDriver:
    """HeadController's sink onto the bus, a faithful port of the http
    driver's joint mapping: pan posts raw degrees to the pan joint; tilt is
    remapped into the nod servo's calibrated window (servo = 90 + angle;
    angle = clamp(nod_sign * tilt + nod_offset, nod_min, nod_max)) — the
    2026-08-17 recalibration, unchanged. Writes ride the "gaze" track, so a
    sign's non-manual marker can outrank face tracking and hand it back."""

    def __init__(self, bus: MotionBus, *, pan_joint: str, tilt_joint: str,
                 limit_deg: float = 80.0, deadband_deg: float = 0.3,
                 max_jump_deg: float = 16.0, nod_offset_deg: float = 0.0,
                 nod_min_deg: float = -90.0, nod_max_deg: float = 90.0,
                 drive_nod: bool = True, nod_sign: float = 1.0):
        self._bus = bus
        self._pan_joint = pan_joint
        self._tilt_joint = tilt_joint
        self._nod_sign = float(nod_sign)
        self._nod_offset = float(nod_offset_deg)
        self._nod_min = float(nod_min_deg)
        self._nod_max = float(nod_max_deg)
        self._drive_nod = bool(drive_nod)
        self._last = (0.0, 0.0)
        bus.register(BusJoint(pan_joint,
                              min_deg=-abs(limit_deg), max_deg=abs(limit_deg),
                              home_deg=0.0, max_jump_deg=max_jump_deg,
                              deadband_deg=deadband_deg))
        if drive_nod:
            bus.register(BusJoint(tilt_joint,
                                  min_deg=self._nod_min, max_deg=self._nod_max,
                                  home_deg=self._nod_offset,
                                  max_jump_deg=max_jump_deg,
                                  deadband_deg=deadband_deg))

    @property
    def moves_hardware(self) -> bool:
        return self._bus.moves_hardware

    @property
    def healthy(self) -> bool:
        return self._bus.healthy

    @property
    def last(self) -> tuple[float, float]:
        return self._last

    def send(self, head_x: float, head_y: float) -> None:
        self._last = (float(head_x), float(head_y))
        targets = {self._pan_joint: float(head_x)}
        if self._drive_nod:
            targets[self._tilt_joint] = max(
                self._nod_min,
                min(self._nod_max,
                    self._nod_sign * float(head_y) + self._nod_offset))
        self._bus.write("gaze", targets)

    def estop(self) -> None:
        self._bus.estop()

    def resume(self) -> None:
        self._bus.resume()

    @property
    def estopped(self) -> bool:
        return self._bus.estopped

    def close(self) -> None:
        self._bus.release("gaze")
