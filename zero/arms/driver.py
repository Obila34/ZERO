"""Arm motion sink — posts named-joint targets to the AF1 gateway.

Same honesty rules as the head driver (audit H1/C2): a post is acknowledged
only on HTTP success, failures are retried, and consecutive failures flip
`healthy`. Simpler than the head driver on purpose: gestures are low-rate
keyframe playback, not a 40 Hz servo loop, so each send is a small batch of
joint posts from one worker thread.

Every angle is clamped to the joint's calibrated envelope HERE, regardless of
what the caller asked for — the driver is the last line before the wire.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

from zero.utils.logging import get_logger

log = get_logger("arms.driver")

# Joints on Nano 1 (steppers; gateway routes S/M commands). Kept in sync with
# the gateway's stepper_index_map — used only to gate them behind
# arms.allow_steppers, never to bypass the gateway's own routing.
STEPPER_JOINTS = frozenset({
    "right_elbow_joint", "right_in_out_joint", "right_up_down_joint",
    "left_elbow_joint", "left_up_down_joint", "left_in_out_joint",
    "right_bicep_joint", "left_bicep_joint",
})

# Joints the gesture layer must never drive, whatever config says — a
# denylist rather than a config flag: it describes the robot right now, and
# a stray config line should not be able to re-enable an excluded joint.
# Currently EMPTY: the wrists came back when the PCA rail revived
# (2026-08-24), and the shoulder in/out pair re-entered scope at the
# operator's Phase-5 decision (2026-08-24 session — the KSL signing stance
# needs it). The mechanism stays: put a joint name here the day its
# hardware is not to be trusted.
EXCLUDED_JOINTS = frozenset()


class JointSpec:
    __slots__ = ("name", "min_deg", "max_deg", "home_deg", "is_stepper")

    def __init__(self, name: str, min_deg: float, max_deg: float,
                 home_deg: float):
        self.name = name
        self.min_deg = float(min_deg)
        self.max_deg = float(max_deg)
        self.home_deg = min(self.max_deg, max(self.min_deg, float(home_deg)))
        self.is_stepper = name in STEPPER_JOINTS

    def clamp(self, deg: float) -> float:
        return min(self.max_deg, max(self.min_deg, float(deg)))


class NullArmDriver:
    """Records the last pose; moves nothing. Default sink."""

    moves_hardware = False
    healthy = True

    def __init__(self):
        self.last: dict[str, float] = {}

    def send(self, pose: dict[str, float]) -> None:
        self.last.update(pose)

    def estop(self) -> None:
        pass

    def close(self) -> None:
        pass


class HttpArmDriver:
    """POST each joint of a pose to /api/joint_cmd. MOVES THE ROBOT."""

    moves_hardware = True

    def __init__(self, *, base_url: str, joints: dict[str, JointSpec],
                 max_hz: float = 10.0, deadband_deg: float = 0.5,
                 timeout_s: float = 1.0):
        self._base = base_url.rstrip("/")
        self._joints = joints
        self._min_interval = 1.0 / max(1.0, float(max_hz))
        self._deadband = float(deadband_deg)
        self._timeout = float(timeout_s)
        # The gateway ADDS its stored per-joint offset before converting to
        # steps, and those offsets are large: right_up_down sat at -150 on
        # 2026-08-17, so posting a raw 0 would have commanded the shoulder 150
        # degrees away from where it booted. This driver therefore works in
        # EFFECTIVE degrees — degrees from the motor's zero, which for these
        # encoderless steppers is the pose they were in at the last reset —
        # and subtracts the offset on the way out. Nothing is posted until the
        # offsets are known: guessing them is how you break an arm.
        self._offsets: dict[str, float] | None = None
        self._sent: dict[str, float] = {}      # last ACKNOWLEDGED angle per joint
        self._pending: dict[str, float] = {}
        self._fails = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker = threading.Thread(target=self._run, name="arm-http",
                                        daemon=True)
        self._worker.start()

    @property
    def healthy(self) -> bool:
        return self._fails < 3

    @property
    def last(self) -> dict[str, float]:
        return dict(self._sent)

    def send(self, pose: dict[str, float]) -> None:
        """Queue a pose. Unknown (uncalibrated) joints are dropped with a log —
        they must never reach the wire."""
        clean = {}
        for name, deg in pose.items():
            spec = self._joints.get(name)
            if spec is None:
                log.warning("arm joint %r has no calibrated envelope — dropped",
                            name)
                continue
            clean[name] = spec.clamp(deg)
        if not clean:
            return
        with self._lock:
            self._pending.update(clean)
        self._wake.set()

    def _run(self) -> None:
        last_t = 0.0
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                batch = self._pending
                self._pending = {}
            if not batch:
                continue
            wait = self._min_interval - (time.monotonic() - last_t)
            if wait > 0:
                time.sleep(wait)
            last_t = time.monotonic()
            retry: dict[str, float] = {}
            for name, deg in batch.items():
                acked = self._sent.get(name)
                if acked is not None and abs(deg - acked) < self._deadband:
                    continue
                if self._post(name, deg):
                    self._sent[name] = deg
                    if self._fails >= 3:
                        log.warning("arm gateway recovered after %d failures",
                                    self._fails)
                    self._fails = 0
                else:
                    self._fails += 1
                    if self._fails == 3:
                        log.warning("arm gateway unreachable (%s) — retrying; "
                                    "the arms are NOT moving", self._base)
                    retry[name] = deg
            if retry:
                with self._lock:
                    for name, deg in retry.items():
                        self._pending.setdefault(name, deg)
                self._wake.set()
                time.sleep(min(2.0, 0.2 * self._fails))

    def _fetch_offsets(self) -> bool:
        """Read the gateway's stored zero-offsets. Until this succeeds the
        driver stays mute — see the note in __init__."""
        if self._offsets is not None:
            return True
        try:
            with urllib.request.urlopen(f"{self._base}/api/calibration",
                                        timeout=self._timeout) as r:
                self._offsets = {k: float(v)
                                 for k, v in json.loads(r.read().decode()).items()}
            log.info("arm gateway offsets loaded (%d joints)", len(self._offsets))
            return True
        except Exception as e:
            log.warning("arm offsets unavailable (%s) — NOT moving: a command "
                        "sent without them lands wherever the offset says", e)
            return False

    def _post(self, joint: str, deg: float) -> bool:
        import math
        if not self._fetch_offsets():
            return False
        deg = deg - self._offsets.get(joint, 0.0)    # effective -> raw command
        body = json.dumps({"name": joint, "angle_deg": deg,
                           "angle_rad": deg * math.pi / 180.0}).encode()
        req = urllib.request.Request(
            f"{self._base}/api/joint_cmd", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=self._timeout).close()
            return True
        except Exception as e:
            log.debug("arm post failed (%s=%.1f): %s", joint, deg, e)
            return False

    def estop(self) -> None:
        try:
            req = urllib.request.Request(f"{self._base}/api/stop", data=b"",
                                         method="POST")
            urllib.request.urlopen(req, timeout=self._timeout).close()
        except Exception as e:
            log.debug("arm estop failed: %s", e)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()


def load_joints(cfg) -> dict[str, JointSpec]:
    """Calibrated joints from `arms.joints` config, merged over the hand
    joints' firmware-truth envelopes. Entries without min/max are refused (an
    envelope is not optional). Steppers are dropped unless
    `arms.allow_steppers` — their zero is wherever the Nano booted.

    The 12 PCA hand joints (wrists + fingers) default to the calibration in
    zero/arms/hands.py, which is read off the gateway firmware itself — the
    first sign build hand-wrote 0-180 envelopes into config and drove the
    right thumb 70 degrees past its stop. Config may still override a hand
    joint, but only by naming it explicitly."""
    from zero.arms.hands import hand_joint_specs

    raw = dict(hand_joint_specs())
    raw.update(cfg.get("arms.joints") or {})
    allow_steppers = bool(cfg.get("arms.allow_steppers", False))
    # Envelopes come from the robot's URDF, which describes each joint's
    # mechanical travel. What it CANNOT tell us is where inside that travel the
    # arm happens to be sitting: these steppers have no encoders, so zero is
    # wherever they were at the last reset. frac shrinks the envelope around
    # home for exactly that reason — use a fraction until supervised motion
    # has confirmed the arm really does rest near the modelled zero.
    frac = min(1.0, max(0.05, float(cfg.get("arms.limit_frac", 1.0))))
    # Envelopes are written in JOINT space (the URDF's frame, where + is the
    # direction "raise" means). A mirrored motor turns the other way, so its
    # window has to mirror with it: a shoulder that lifts 106 up and 68.8 down
    # would otherwise only manage 68.8 of lift once its sign was flipped.
    signs = cfg.get("arms.joint_sign") or {}
    joints: dict[str, JointSpec] = {}
    for name, ent in raw.items():
        if name in EXCLUDED_JOINTS:
            log.info("arms: %s is excluded from the gesture layer — ignored",
                     name)
            continue
        try:
            home = float(ent.get("home", 0.0))
            # frac exists for the encoderless steppers, whose true resting
            # pose is unverified; PCA servos are absolute, so their firmware
            # envelopes are used whole — shrinking a finger's travel would
            # just deform every handshape.
            f = frac if name in STEPPER_JOINTS else 1.0
            lo = home + (float(ent["min"]) - home) * f
            hi = home + (float(ent["max"]) - home) * f
            if float(signs.get(name, 1.0)) < 0:      # motor mirrored
                lo, hi = 2 * home - hi, 2 * home - lo
            spec = JointSpec(str(name), lo, hi, home)
        except (KeyError, TypeError, ValueError) as e:
            log.warning("arms.joints.%s invalid (%s) — ignored", name, e)
            continue
        if spec.is_stepper and not allow_steppers:
            log.info("arms: stepper joint %s configured but allow_steppers is "
                     "false — inert", name)
            continue
        joints[spec.name] = spec
    return joints


def make_arm_driver(cfg, joints: dict[str, JointSpec]):
    kind = str(cfg.get("arms.driver", "null")).lower()
    if kind == "bus" and joints:
        # The recommended transport: gestures share the MotionBus with gaze
        # and sign — one clock, one arbiter, one e-stop. Whether the bus
        # moves metal is motion.driver's call, not this one.
        from zero.motion.drivers import BusArmDriver, get_bus

        bus = get_bus(cfg)
        drv = BusArmDriver(bus, joints,
                           deadband_deg=float(
                               cfg.get("arms.gateway.deadband_deg", 0.3)))
        if bus.moves_hardware:
            log.warning("arms.driver=bus + motion.driver=http — ARMS WILL "
                        "MOVE (%d calibrated joint(s))", len(joints))
        return drv
    if kind == "bus":
        log.warning("arms.driver=bus but no calibrated joints — NullArmDriver")
        return NullArmDriver()
    if kind == "http" and joints:
        drv = HttpArmDriver(
            base_url=cfg.get("arms.gateway.base_url",
                             cfg.get("head.gateway.base_url",
                                     "http://100.67.233.65:5000")),
            joints=joints,
            max_hz=float(cfg.get("arms.gateway.max_hz", 10.0)),
            deadband_deg=float(cfg.get("arms.gateway.deadband_deg", 0.5)),
            timeout_s=float(cfg.get("arms.gateway.timeout_s", 1.0)),
        )
        log.warning("arms.driver=http — ARMS WILL MOVE via %s (%d calibrated "
                    "joint(s))", drv._base, len(joints))
        return drv
    if kind == "http":
        log.warning("arms.driver=http but no calibrated joints — NullArmDriver")
    return NullArmDriver()
