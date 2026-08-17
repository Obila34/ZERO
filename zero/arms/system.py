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

# Built on the SHOULDER-LIFT / BICEP / ELBOW joints only: the shoulder in-out
# pair and the wrists are excluded from the gesture layer (driver.EXCLUDED_
# JOINTS), so a wave here is an arm swing from the elbow, not a hand flap.
# Offsets are home-relative, so the same definitions hold whatever each joint's
# calibration turns out to be, and every one returns to home.
BUILTIN_GESTURES: dict[str, list] = {
    # Sized to be SEEN. The first version beat a 10-degree elbow twitch, which
    # completed correctly and was invisible across a room (2026-08-17); the
    # shoulder carries 106 degrees of travel, so that is what actually reads as
    # a gesture. Frame durations are long enough for max_speed_dps to reach the
    # target — a frame that ends early leaves the gesture half-made.
    "wave_right": [({"right_up_down_joint": "home+45"}, 0.7),
                   ({"right_elbow_joint": "home+18"}, 0.4),
                   ({"right_elbow_joint": "home-18"}, 0.4),
                   ({"right_elbow_joint": "home+18"}, 0.4),
                   ({"right_elbow_joint": "home",
                     "right_up_down_joint": "home"}, 0.8)],
    "wave_left": [({"left_up_down_joint": "home+45"}, 0.7),
                  ({"left_elbow_joint": "home+18"}, 0.4),
                  ({"left_elbow_joint": "home-18"}, 0.4),
                  ({"left_elbow_joint": "home+18"}, 0.4),
                  ({"left_elbow_joint": "home",
                    "left_up_down_joint": "home"}, 0.8)],
    # A conversational beat: a real lift of the arm, not a twitch.
    "beat_right": [({"right_up_down_joint": "home+22"}, 0.45),
                   ({"right_up_down_joint": "home"}, 0.5)],
    "beat_both": [({"right_up_down_joint": "home+22",
                    "left_up_down_joint": "home+22"}, 0.45),
                  ({"right_up_down_joint": "home",
                    "left_up_down_joint": "home"}, 0.5)],
    "shrug": [({"right_up_down_joint": "home+25",
                "left_up_down_joint": "home+25"}, 0.5),
              ({"right_up_down_joint": "home+25",
                "left_up_down_joint": "home+25"}, 0.4),
              ({"right_up_down_joint": "home",
                "left_up_down_joint": "home"}, 0.6)],
    "point_right": [({"right_up_down_joint": "home+40",
                      "right_elbow_joint": "home-15"}, 0.6),
                    ({"right_up_down_joint": "home+40",
                      "right_elbow_joint": "home-15"}, 0.7),
                    ({"right_up_down_joint": "home",
                      "right_elbow_joint": "home"}, 0.7)],
    "point_left": [({"left_up_down_joint": "home+40",
                     "left_elbow_joint": "home-15"}, 0.6),
                   ({"left_up_down_joint": "home+40",
                     "left_elbow_joint": "home-15"}, 0.7),
                   ({"left_up_down_joint": "home",
                     "left_elbow_joint": "home"}, 0.7)],
    "show_big": [({"right_up_down_joint": "home+35",
                   "left_up_down_joint": "home+35"}, 0.6),
                 ({"right_up_down_joint": "home+35",
                   "left_up_down_joint": "home+35"}, 0.5),
                 ({"right_up_down_joint": "home",
                   "left_up_down_joint": "home"}, 0.7)],
    "offer_right": [({"right_up_down_joint": "home+30",
                      "right_elbow_joint": "home-20"}, 0.6),
                    ({"right_up_down_joint": "home+30",
                      "right_elbow_joint": "home-20"}, 0.8),
                    ({"right_up_down_joint": "home",
                      "right_elbow_joint": "home"}, 0.7)],
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
        # Expression gating (see express()). Gestures are opt-in per hand,
        # suppressible wholesale when someone is close to the arms, paced so
        # most turns stay idle, and deictics are refused unless perception has
        # actually grounded a target.
        self._hand_state = {"left": "free", "right": "free"}
        self._suppress_until = 0.0
        self._last_gesture_t = 0.0
        self._min_gap_s = float(cfg.get("arms.min_gesture_gap_s", 12.0))
        self._can_point = bool(cfg.get("arms.allow_pointing", False))
        # Per-joint direction correction: +1 means the spoken verb's direction
        # ("raise") matches the joint's positive rotation. Flip to -1 for any
        # joint that turns out to move the opposite way — config, not code.
        self._joint_sign = dict(cfg.get("arms.joint_sign") or {})
        # Speech beats: a small gesture on some spoken sentences even when the
        # model cued nothing, so the arms are visibly alive while ZERO talks.
        from zero.arms.commands import set_default_step
        set_default_step(float(cfg.get("arms.step_deg", 30.0)))
        self._speech_beat = bool(cfg.get("arms.speech_beat", True))
        self._beat_gap_s = float(cfg.get("arms.beat_gap_s", 8.0))
        self._beat_n = 0
        # Gesture units: how long a stroke is held before retracting. A
        # following gesture inside this window cancels the retraction, so the
        # arm flows between gestures instead of returning to rest each time.
        self._unit_hold_s = float(cfg.get("arms.unit_hold_s", 1.2))
        # Speech pacing, for putting the stroke on its word (see express()).
        self._words_per_s = float(cfg.get("arms.words_per_sec", 2.8))
        self._prep_lead_s = float(cfg.get("arms.prep_lead_s", 0.25))
        self._timer = None
        self._gen = 0                       # bumps to preempt a running gesture
        self._lock = threading.Lock()
        self._player: threading.Thread | None = None

    # ── public surface ───────────────────────────────────────────────────────
    def gesture_names(self) -> list[str]:
        """Gestures that could actually run — every joint they touch has a
        calibrated envelope. The prompt is built from this, so the model is
        never taught a gesture the robot cannot perform."""
        out = []
        for name, frames in self._gestures.items():
            if name == "rest":
                continue
            joints = {j for tg, _s in frames for j in tg}
            if joints and joints <= set(self._joints):
                out.append(name)
        return sorted(out)

    def move_joint(self, joints, degrees: float, *, relative: bool = True) -> list:
        """Drive named joints by (or to) an angle — the voice path for "raise
        your right elbow". Returns the joints actually moved, so the caller can
        say honestly what happened; a joint with no calibrated envelope is
        skipped rather than silently ignored.

        `degrees` is in the operator's frame: positive = the direction the
        spoken verb asked for ("raise"). Which way that is on the metal is
        per-joint, so arms.joint_sign flips one without touching code — the
        same pattern the nod needed.
        """
        if isinstance(joints, str):
            joints = [joints]
        if self._estop:
            return []
        targets, moved = {}, []
        for name in joints:
            spec = self._joints.get(name)
            if spec is None:
                log.info("arms: %s has no calibrated envelope — not moved", name)
                continue
            sign = float(self._joint_sign.get(name, 1.0))
            want = float(degrees) * sign
            base = self._pose.get(name, spec.home_deg)
            targets[name] = spec.clamp(base + want if relative else want)
            moved.append(name)
        if not targets:
            return []
        with self._lock:
            self._gen += 1
            gen = self._gen
        self._player = threading.Thread(
            target=self._play, args=("move", [(targets, 1.0)], gen),
            name="arm-move", daemon=True)
        self._player.start()
        return moved

    def joint_pose(self) -> dict:
        """Current commanded angle per calibrated joint (effective degrees)."""
        return {k: round(v, 1) for k, v in self._pose.items()}

    def set_hand_state(self, left: str = "free", right: str = "free") -> None:
        """Mark a hand occupied (holding something) so no gesture uses it."""
        self._hand_state = {"left": str(left), "right": str(right)}

    def suppress(self, seconds: float) -> None:
        """Block gestures for a while — someone is close enough to the arms
        that a swinging limb is not acceptable."""
        self._suppress_until = time.monotonic() + max(0.0, float(seconds))

    def express(self, text: str, *, now: float | None = None) -> str | None:
        """Fire the gesture cued in a spoken sentence, if any, and return the
        gesture played. The whole safety stack lives here, so it applies
        wherever expression comes from.

        Returns None (and plays nothing) when: nothing is cued, gestures are
        suppressed or e-stopped, the gesture's hand is occupied, a deictic
        cannot be grounded, or the pacing floor hasn't elapsed — a hand that
        moves on every sentence reads as nervous rather than alive.
        """
        from zero.arms.cues import (CUE_FUNCTION, CUE_TO_GESTURE, cue_positions,
                                    find_cues)

        now = time.monotonic() if now is None else now
        cues = find_cues(text)
        if self._estop or now < self._suppress_until:
            return None
        if cues and now - self._last_gesture_t < self._min_gap_s:
            return None                     # pacing: keep most turns idle
        cue = cues[0] if cues else None      # one gesture per sentence, at most
        if cue is None:
            # Nothing cued — which is nearly always, because the model rarely
            # writes one. Read the sentence itself: a greeting wants a wave, "I
            # don't know" a shrug, "huge" a size gesture, "over there" a point.
            # Without this only the beat below ever fires and the other four
            # gesture classes never happen at all.
            from zero.arms.cues import infer_gesture

            if now - self._last_gesture_t < self._beat_gap_s:
                return None
            inferred = infer_gesture(text)
            if inferred is not None:
                name, word_i = inferred
                if name.startswith("point") and not self._can_point:
                    inferred = None          # never point at an unseen target
                elif self._hand_state.get(
                        "left" if name.endswith("_left") else "right",
                        "free") != "free":
                    inferred = None          # that hand is holding something
                elif self._play_at(name, word_i):
                    self._last_gesture_t = now
                    return name
            # Nothing the words called for: people's hands still move while
            # they talk, so fall back to an occasional beat.
            if not self._speech_beat or len(text.split()) < 5:
                return None
            self._beat_n += 1
            name = "beat_both" if self._beat_n % 3 == 0 else "beat_right"
            if not self._play_on_word(name, text, None):
                return None
            self._last_gesture_t = now
            return name
        name = CUE_TO_GESTURE.get(cue)
        if name is None:
            return None
        if CUE_FUNCTION.get(cue) == "deictic" and not self._can_point:
            return None                     # never point at an unseen target
        hand = ("left" if name.endswith("_left") or "left" in name
                else "right")
        if self._hand_state.get(hand, "free") != "free":
            return None                     # that hand is holding something
        if not self._play_on_word(name, text, cue):
            return None                     # uncalibrated / unknown: refuse
        self._last_gesture_t = now
        return name

    def _play_on_word(self, name: str, text: str, cue: str | None) -> bool:
        """Start the gesture so its STROKE lands on the word it belongs to.

        McNeill's phonological synchrony rule: the stroke coincides with, or
        slightly precedes, the stressed syllable of its word — and never
        follows it. So the gesture waits until that word is about to be
        spoken, minus a preparation lead, because the arm must already be on
        its way before the word arrives. Firing at sentence onset (what this
        did before) put the stroke seconds early on a long sentence.
        """
        from zero.arms.cues import cue_positions

        delay = 0.0
        if cue is not None:
            for c, idx, _total in cue_positions(text):
                if c == cue:
                    delay = idx / max(0.1, self._words_per_s)
                    break
        else:
            # An uncued speech beat has no affiliate word, so put it on the
            # phrase's first stressed beat rather than the opening syllable.
            delay = 2.0 / max(0.1, self._words_per_s)
        delay = max(0.0, delay - self._prep_lead_s)
        if delay < 0.02:
            return self.play(name)
        if not self._can_play(name):
            return False               # check BEFORE promising, not after
        t = threading.Timer(delay, self.play, args=(name,))
        t.daemon = True
        self._timer = t
        t.start()
        return True

    def _play_at(self, name: str, word_index: int) -> bool:
        """Schedule a gesture's stroke onto a known word index."""
        delay = max(0.0, word_index / max(0.1, self._words_per_s)
                    - self._prep_lead_s)
        if delay < 0.02:
            return self.play(name)
        if not self._can_play(name):
            return False
        t = threading.Timer(delay, self.play, args=(name,))
        t.daemon = True
        self._timer = t
        t.start()
        return True

    def _can_play(self, name: str) -> bool:
        """Whether play(name) would succeed, without starting it."""
        frames = self._gestures.get(name)
        if frames is None or self._estop:
            return False
        if name == "rest":
            return True
        return any(j in self._joints for tg, _s in frames for j in tg)

    def set_pointing_allowed(self, allowed: bool) -> None:
        """Whether a deictic cue may be honoured — set from perception, so a
        point is only ever made toward something actually seen."""
        self._can_point = bool(allowed)

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
        """A gesture keyframe -> an absolute motor angle. Home-relative offsets
        are written in JOINT space (+ = the way "raise" means), so a mirrored
        motor gets the sign applied here; otherwise every gesture would run
        backwards on that side."""
        sign = float(self._joint_sign.get(spec.name, 1.0))
        if isinstance(target, (int, float)):
            return float(target)
        s = str(target).strip().lower()
        if s == "home":
            return spec.home_deg
        if s.startswith("home+") or s.startswith("home-"):
            try:
                return spec.home_deg + sign * float(s[4:])
            except ValueError:
                return None
        return None

    def _play(self, name: str, frames: list, gen: int) -> None:
        """Run a gesture as Kendon phases, with human motion kinematics.

        Each frame is driven by the MINIMUM-JERK profile that natural reaching
        follows — a bell-shaped velocity curve, rather than the constant-speed
        chase this used to do, which reads as machinery. The frame's duration
        is stretched if the move could not be made inside it at max_speed_dps,
        so a gesture can no longer arrive half-made.

        The final frame is the RETRACTION, and it is deliberately deferred:
        Kendon's gesture units show that when gestures come in sequence the
        retraction is shortened or dropped entirely, the hand staying up
        between strokes. So the stroke is held for unit_hold_s first, and if
        another gesture preempts during that hold, this one never retracts —
        the arm flows from one gesture into the next instead of bobbing back
        to rest between every one.
        """
        dt = 1.0 / max(1.0, self._rate)
        stroke, retract = (frames[:-1], frames[-1:]) if len(frames) > 1 else (frames, [])
        log.info("arm gesture %r (%d frame(s))", name, len(frames))
        if not self._run_frames(stroke, gen, dt):
            return
        if retract:
            # Post-stroke hold — the window in which a following gesture can
            # cancel the retraction and chain into this one.
            t_end = time.monotonic() + self._unit_hold_s
            while time.monotonic() < t_end:
                with self._lock:
                    if gen != self._gen:
                        return          # chained: leave the hand where it is
                time.sleep(dt)
            self._run_frames(retract, gen, dt)
        log.info("arm gesture %r done", name)

    def _run_frames(self, frames: list, gen: int, dt: float) -> bool:
        """Play frames on a minimum-jerk profile. False if preempted."""
        for targets, dur in frames:
            goal = {}
            for jname, tgt in targets.items():
                spec = self._joints.get(jname)
                if spec is None:
                    continue            # uncalibrated — inert by design
                v = self._resolve(tgt, spec)
                if v is not None:
                    goal[jname] = spec.clamp(v)
            if not goal:
                continue
            start = {j: self._pose.get(j, 0.0) for j in goal}
            far = max(abs(goal[j] - start[j]) for j in goal)
            # Minimum jerk peaks at 1.875 * distance / duration. Stretch the
            # frame if that would exceed the joint speed cap, so the gesture
            # completes instead of being cut off mid-move.
            dur = max(float(dur), 1.875 * far / max(1e-6, self._max_dps))
            t0 = time.monotonic()
            while True:
                with self._lock:
                    if gen != self._gen:
                        return False
                tau = min(1.0, (time.monotonic() - t0) / dur)
                # 10t^3 - 15t^4 + 6t^5: the minimum-jerk position profile,
                # zero velocity AND zero acceleration at both ends.
                ease = tau * tau * tau * (10.0 + tau * (-15.0 + 6.0 * tau))
                moved = {}
                for j, g in goal.items():
                    self._pose[j] = start[j] + (g - start[j]) * ease
                    moved[j] = self._pose[j]
                try:
                    self._driver.send(moved)
                except Exception as e:
                    log.debug("arm send failed: %s", e)
                if tau >= 1.0:
                    break
                time.sleep(dt)
        return True
