"""ArmSystem — keyframed gesture playback over calibrated joints.

A gesture is a list of frames; each frame is ({joint: target}, seconds).
Targets may be numbers (degrees, gateway command space), "home", or
"home+N"/"home-N". Playback interpolates linearly between the current pose and
each frame's targets at ~20 Hz with a per-joint speed cap, so a hobby servo is
never asked to snap across its range. One gesture at a time; a new one
preempts. Gestures come from config (`arms.gestures`) merged over a minimal
built-in set, so calibration values slot in without code changes.

The system NEVER moves on its own: no idle animation, no tracking. It moves
only when a gesture is requested (voice/LLM/API) and eases every calibrated
joint back to home on `rest()` and on shutdown.
"""
from __future__ import annotations

import threading
import time

from zero.arms.driver import load_joints, make_arm_driver
from zero.utils.logging import get_logger

log = get_logger("arms.system")

# Built-in gestures reference wrists only (servo joints, offset-calibrated so
# command 0 = the gateway's saved neutral). Hand open/close poses need the
# supervised finger calibration first — add them in config arms.gestures.
BUILTIN_GESTURES: dict[str, list] = {
    "wave_right": [({"right_wrist_joint": "home+25"}, 0.4),
                   ({"right_wrist_joint": "home-25"}, 0.4),
                   ({"right_wrist_joint": "home+25"}, 0.4),
                   ({"right_wrist_joint": "home-25"}, 0.4),
                   ({"right_wrist_joint": "home"}, 0.4)],
    "wave_left": [({"left_wrist_joint": "home+25"}, 0.4),
                  ({"left_wrist_joint": "home-25"}, 0.4),
                  ({"left_wrist_joint": "home+25"}, 0.4),
                  ({"left_wrist_joint": "home-25"}, 0.4),
                  ({"left_wrist_joint": "home"}, 0.4)],
    "rest": [({}, 0.5)],       # special-cased: every calibrated joint -> home
}


class ArmSystem:
    def __init__(self, cfg):
        self._joints = load_joints(cfg)
        self._driver = make_arm_driver(cfg, self._joints)
        self._rate = float(cfg.get("arms.rate_hz", 20.0))
        self._max_dps = float(cfg.get("arms.max_speed_dps", 60.0))
        self._gestures = dict(BUILTIN_GESTURES)
        for name, frames in (cfg.get("arms.gestures") or {}).items():
            try:
                self._gestures[str(name)] = [
                    (dict(f["joints"]), float(f["s"])) for f in frames]
            except (KeyError, TypeError, ValueError) as e:
                log.warning("arms.gestures.%s invalid (%s) — ignored", name, e)
        # pose belief: last commanded angle per joint; starts at home.
        self._pose = {n: s.home_deg for n, s in self._joints.items()}
        self._estop = False
        self._gen = 0                       # bumps to preempt a running gesture
        self._lock = threading.Lock()
        self._player: threading.Thread | None = None

    # ── public surface ───────────────────────────────────────────────────────
    def gesture_names(self) -> list[str]:
        return sorted(self._gestures)

    def play(self, name: str) -> bool:
        """Start a gesture (preempting any running one). False if unknown or
        e-stopped. Returns immediately; playback is threaded."""
        frames = self._gestures.get(name)
        if frames is None or self._estop:
            return False
        if name == "rest":
            frames = [({j: "home" for j in self._joints}, 0.8)]
        elif not any(j in self._joints for tg, _ in frames for j in tg):
            # every joint this gesture touches is uncalibrated — playing it
            # would be a silent no-op while the voice claims success. Refuse.
            return False
        with self._lock:
            self._gen += 1
            gen = self._gen
        self._player = threading.Thread(
            target=self._play, args=(name, frames, gen),
            name="arm-gesture", daemon=True)
        self._player.start()
        return True

    def rest(self) -> bool:
        return self.play("rest")

    def estop(self) -> None:
        self._estop = True
        with self._lock:
            self._gen += 1                  # kills any running playback
        try:
            self._driver.estop()
        except Exception:
            pass

    def resume(self) -> None:
        self._estop = False

    def status(self) -> dict:
        return {"joints": sorted(self._joints),
                "gestures": self.gesture_names(),
                "driver": type(self._driver).__name__,
                "moves_hardware": getattr(self._driver, "moves_hardware", False),
                "driver_healthy": getattr(self._driver, "healthy", None),
                "estop": self._estop,
                "pose": {k: round(v, 1) for k, v in self._pose.items()}}

    def stop(self) -> None:
        """Ease home, then release the driver. Bounded."""
        if self._joints and not self._estop:
            self.rest()
            t = self._player
            if t is not None:
                t.join(timeout=3.0)
        try:
            self._driver.close()
        except Exception:
            pass

    # ── playback ─────────────────────────────────────────────────────────────
    def _resolve(self, target, spec) -> float | None:
        if isinstance(target, (int, float)):
            return float(target)
        s = str(target).strip().lower()
        if s == "home":
            return spec.home_deg
        if s.startswith("home+") or s.startswith("home-"):
            try:
                return spec.home_deg + float(s[4:])
            except ValueError:
                return None
        return None

    def _play(self, name: str, frames: list, gen: int) -> None:
        dt = 1.0 / max(1.0, self._rate)
        step_cap = self._max_dps * dt
        log.info("arm gesture %r (%d frame(s))", name, len(frames))
        for targets, hold_s in frames:
            goal: dict[str, float] = {}
            for jname, tgt in targets.items():
                spec = self._joints.get(jname)
                if spec is None:
                    continue                # uncalibrated — inert by design
                v = self._resolve(tgt, spec)
                if v is not None:
                    goal[jname] = spec.clamp(v)
            t_end = time.monotonic() + max(0.05, float(hold_s))
            while time.monotonic() < t_end:
                with self._lock:
                    if gen != self._gen:
                        return              # preempted / e-stopped
                moved = {}
                for jname, tgt in goal.items():
                    cur = self._pose[jname]
                    if cur != tgt:
                        step = max(-step_cap, min(step_cap, tgt - cur))
                        self._pose[jname] = cur + step
                        moved[jname] = self._pose[jname]
                if moved:
                    try:
                        self._driver.send(moved)
                    except Exception as e:
                        log.debug("arm send failed: %s", e)
                time.sleep(dt)
        log.info("arm gesture %r done", name)
