"""MotionBus — the single owner of every joint on the robot.

Before the bus, the head driver (28 Hz), the arm driver (10 Hz) and the Pi's
sign code each posted to the AF-1 gateway independently: three writers, three
clocks, no shared e-stop, and a 12-joint sign pose cost 12 HTTP requests.
The bus replaces all of that with one arbiter:

  * Producers (gaze controller, gesture playback, sign engine) WRITE target
    setpoints onto named TRACKS. Writing claims the joint; releasing frees it.
  * One tick — the bus's own clock — samples every track at the same instant,
    resolves each joint to its highest-priority claimant, and posts the result.
    That simultaneous sample is what "the whole body moves in sync" means
    mechanically.
  * Hand joints the firmware itself batches go out as ONE /api/pose_cmd; the
    rest go joint-by-joint, exact parity with the proven drivers.

Priority (fixed, highest wins): sign > command > gesture > gaze > idle.
Signing therefore owns the hands while a speech beat fires harmlessly, and
can later claim the head for non-manual markers, outranking face tracking;
the moment it releases, the next gaze write takes the head back.

Safety, inherited from the drivers this replaces (audit H1/C2):
  * every angle is clamped to the joint's registered envelope HERE — the bus
    is the last line before the wire;
  * a post is acknowledged only on transport success; unacknowledged targets
    are simply retried next tick (the setpoint is still there), with backoff
    while the link is down, and `healthy` flips after 3 straight failures;
  * per-joint max_jump: after an outage the bus WALKS to the target instead
    of whipping — slew limits upstream live in belief space only;
  * stepper joints marked use_offset stay MUTE until the gateway's stored
    zero offsets have been read — commanding an encoderless stepper without
    them lands wherever the offset says (that is how you break an arm);
  * estop() posts /api/stop, freezes ALL posting (every track, every joint)
    until resume() — the shared e-stop no pair of independent drivers had.

Release semantics: when a joint's winning track releases it, the bus HOLDS
the last posted value; a lower-priority track regains the joint only on its
next write. Stale setpoints from before a claim never resurface on their own
— a sign must not end with the hand snapping back to a beat gesture that
finished during the spell.

Stdlib only.
"""
from __future__ import annotations

import threading
import time

from zero.utils.logging import get_logger

log = get_logger("motion.bus")

# Fixed track priorities — highest wins the joint. A deliberate constant, not
# config: reordering these changes what the robot does mid-conversation, and a
# stray config line should not be able to put a speech beat above a sign.
TRACK_PRIORITY = {
    "sign": 40,       # sign language playback (letters, lexicon signs)
    "command": 30,    # explicit voice/tool commands ("raise your arm")
    "gesture": 20,    # expressive layer (beats, waves, shrugs)
    "gaze": 10,       # head tracking / social gaze
    "idle": 0,        # future: breathing sway, rest drift
}


class BusJoint:
    """A registered joint: envelope + transport behaviour. min/max/home are in
    EFFECTIVE degrees (the caller's frame); offsets, when configured, are
    subtracted on the way to the wire."""

    __slots__ = ("name", "min_deg", "max_deg", "home_deg", "max_jump_deg",
                 "deadband_deg", "use_offset", "batch")

    def __init__(self, name: str, *, min_deg: float, max_deg: float,
                 home_deg: float = 0.0, max_jump_deg: float = 0.0,
                 deadband_deg: float = 0.3, use_offset: bool = False,
                 batch: bool = False):
        self.name = str(name)
        self.min_deg = float(min_deg)
        self.max_deg = float(max_deg)
        self.home_deg = min(self.max_deg, max(self.min_deg, float(home_deg)))
        self.max_jump_deg = float(max_jump_deg)      # <=0 disables walking
        self.deadband_deg = float(deadband_deg)
        self.use_offset = bool(use_offset)
        self.batch = bool(batch)

    def clamp(self, deg: float) -> float:
        return min(self.max_deg, max(self.min_deg, float(deg)))


class MotionBus:
    """One writer, one clock, one e-stop for all 23 joints."""

    def __init__(self, transport, *, rate_hz: float = 30.0):
        self._transport = transport
        self._interval = 1.0 / max(1.0, float(rate_hz))
        self._joints: dict[str, BusJoint] = {}
        # targets[track][joint] = effective degrees. A joint key present in a
        # track dict IS the claim; release deletes the key.
        self._targets: dict[str, dict[str, float]] = {}
        self._posted: dict[str, float] = {}     # last ACKNOWLEDGED, effective
        self._offsets: dict[str, float] | None = None
        self._estop = False
        self._fails = 0
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="motion-bus",
                                        daemon=True)
        self._thread.start()

    # ── registration ─────────────────────────────────────────────────────────
    def register(self, joint: BusJoint) -> None:
        """Idempotent: re-registering a name replaces its spec (a driver
        rebuilding after a config reload must not stack duplicates)."""
        with self._lock:
            self._joints[joint.name] = joint

    def joints(self) -> list[str]:
        with self._lock:
            return sorted(self._joints)

    def spec(self, name: str) -> BusJoint | None:
        with self._lock:
            return self._joints.get(name)

    # ── producer surface ─────────────────────────────────────────────────────
    def write(self, track: str, targets: dict[str, float]) -> list[str]:
        """Set targets (effective degrees) on a track, claiming each joint.
        Unknown joints are dropped WITH a log — they must never reach the
        wire — and the accepted joint names are returned so the caller can
        say honestly what will move."""
        if track not in TRACK_PRIORITY:
            log.warning("unknown motion track %r — write ignored", track)
            return []
        accepted = []
        prio = TRACK_PRIORITY[track]
        with self._lock:
            tr = self._targets.setdefault(track, {})
            for name, deg in targets.items():
                spec = self._joints.get(name)
                if spec is None:
                    log.info("motion: %s not registered — dropped", name)
                    continue
                tr[name] = spec.clamp(deg)
                accepted.append(name)
                # A write PREEMPTS lower-priority standing claims on the same
                # joint. Without this, a beat gesture that fired mid-sign
                # would sit in its track and snap the hand back the moment
                # the sign released. A producer that is still live (the
                # 28 Hz gaze loop, a running gesture playback) reclaims the
                # joint naturally with its next write; a finished one stays
                # gone — which is exactly the difference that matters.
                for other, otr in self._targets.items():
                    if TRACK_PRIORITY[other] < prio:
                        otr.pop(name, None)
        if accepted:
            self._wake.set()
        return accepted

    def release(self, track: str, joints: list[str] | None = None) -> None:
        """Free joints claimed by a track (all of them when joints is None).
        The bus holds each joint's last posted value until some track writes
        it again."""
        with self._lock:
            tr = self._targets.get(track)
            if not tr:
                return
            if joints is None:
                tr.clear()
            else:
                for name in joints:
                    tr.pop(name, None)

    def owner(self, joint: str) -> str | None:
        """Which track currently wins this joint (None = unclaimed)."""
        with self._lock:
            return self._owner_locked(joint)

    def _owner_locked(self, joint: str) -> str | None:
        best, best_p = None, -1
        for track, tr in self._targets.items():
            if joint in tr and TRACK_PRIORITY[track] > best_p:
                best, best_p = track, TRACK_PRIORITY[track]
        return best

    # ── safety ───────────────────────────────────────────────────────────────
    def estop(self) -> None:
        """Freeze everything and tell the gateway to stop. Affects every
        track — this is the shared hard-stop."""
        self._estop = True
        try:
            self._transport.stop()
        except Exception:
            pass
        log.warning("MOTION E-STOP — all tracks frozen until resume()")

    def resume(self) -> None:
        self._estop = False
        self._wake.set()

    @property
    def estopped(self) -> bool:
        return self._estop

    @property
    def healthy(self) -> bool:
        return self._fails < 3

    @property
    def moves_hardware(self) -> bool:
        return bool(getattr(self._transport, "moves_hardware", False))

    @property
    def last(self) -> dict[str, float]:
        with self._lock:
            return dict(self._posted)

    # ── the clock ────────────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop_evt.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            while not self._stop_evt.is_set():
                t0 = time.monotonic()
                busy = self._tick()
                if not busy:
                    break                   # nothing pending — sleep on wake
                # Link down: back off instead of hammering a dead gateway
                # (each timeout already slows the tick further).
                if self._fails:
                    time.sleep(min(2.0, 0.2 * self._fails))
                dt = self._interval - (time.monotonic() - t0)
                if dt > 0:
                    time.sleep(dt)

    def _tick(self) -> bool:
        """One resolution + post cycle. Returns True while any joint still
        has an unacknowledged delta (keeps the clock running)."""
        if self._estop:
            return False
        # Resolve under the lock; post outside it (HTTP must never block a
        # producer's write call).
        with self._lock:
            resolved: dict[str, tuple[BusJoint, float]] = {}
            for name, spec in self._joints.items():
                track = self._owner_locked(name)
                if track is None:
                    continue
                resolved[name] = (spec, self._targets[track][name])
        batch: dict[str, float] = {}        # effective deg, batch-capable
        singles: dict[str, tuple[BusJoint, float]] = {}
        pending = False
        for name, (spec, target) in resolved.items():
            acked = self._posted.get(name)
            step = target
            if acked is not None:
                if abs(target - acked) < spec.deadband_deg:
                    continue
                # Bounded hop: walk toward the target from the last
                # acknowledged pose, never whip the whole accumulated error.
                if spec.max_jump_deg > 0:
                    delta = target - acked
                    if abs(delta) > spec.max_jump_deg:
                        step = acked + spec.max_jump_deg * (
                            1.0 if delta > 0 else -1.0)
                        pending = True      # keep walking next tick
            if spec.use_offset:
                if not self._ensure_offsets():
                    pending = True          # mute until offsets are known
                    continue
                singles[name] = (spec, step)
            elif spec.batch:
                batch[name] = step
            else:
                singles[name] = (spec, step)
        ok_all = True
        if batch:
            if self._transport.post_pose(
                    {n: round(v, 2) for n, v in batch.items()}):
                self._posted.update(batch)
            else:
                ok_all = False
                pending = True
        for name, (spec, step) in singles.items():
            wire = step - (self._offsets or {}).get(name, 0.0) \
                if spec.use_offset else step
            if self._transport.post_joint(name, round(wire, 2)):
                self._posted[name] = step
            else:
                ok_all = False
                pending = True
        if ok_all:
            if self._fails >= 3:
                log.warning("gateway recovered after %d failed cycles",
                            self._fails)
            self._fails = 0
        else:
            self._fails += 1
            if self._fails == 3:
                log.warning("gateway unreachable — retrying; the robot is "
                            "NOT moving")
        return pending

    def _ensure_offsets(self) -> bool:
        if self._offsets is not None:
            return True
        got = self._transport.fetch_offsets()
        if got is None:
            return False
        self._offsets = got
        log.info("gateway offsets loaded (%d joints)", len(got))
        return True

    def close(self) -> None:
        self._stop_evt.set()
        self._wake.set()
        try:
            self._transport.close()
        except Exception:
            pass
