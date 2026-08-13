"""HeadSystem — composition root for ZERO's head/attention subsystem.

Ties together the pieces so main.py has one small surface and never touches the
control internals:

  driver      — the motion sink (NullDriver by default: nothing moves).
  controller  — HeadController: the single 25 Hz motion emitter (slew-limited).
  tracker     — FaceTracker: closed-loop visual servo toward the addressee.
  scheduler   — GazeScheduler: the social layer (look at / look away by state).
  source loop — pulls the face target from Eyes' digital gaze at ~15 Hz, applies
                the eyes-lead-head rule (neck holds for small offsets; the digital
                crop is the fast 'eye'), composites the social bias, and drives
                the tracker/controller. Efference copy: whenever the neck is
                commanded to move it tells Eyes to suppress scene-change novelty,
                and re-settles the baseline when it stops.

Runs entirely on the ZERO Pi. With the default NullDriver the whole thing is
exercised — social gaze, tracking, suppression — without moving real metal, so
its feel and its emitted command stream can be judged before any actuator is
wired. Everything is guarded: a dead sink, a missing frame, or a None target
never breaks the loop or the conversation.

Dependency-light: no cv2, no sockets (the driver owns transport). numpy is only
touched via a frame's .shape.
"""
from __future__ import annotations

import threading
import time

from zero.head.controller import HeadController
from zero.head.driver import make_driver
from zero.head.social import GazeScheduler
from zero.head.stabilizer import Stabilizer
from zero.head.hand import HandPoseSource
from zero.head.tracker import FaceTracker
from zero.utils.logging import get_logger

log = get_logger("head.system")


class HeadSystem:
    # A source loop pulling the face target from Eyes; independent of the 25 Hz
    # control tick (the controller smooths between these updates).
    tracks_itself = True

    def __init__(self, cfg, eyes=None, gate=None):
        self._cfg = cfg
        self._eyes = eyes
        self._ext_gate = gate            # AF1 ownership gate (optional)
        self._estop = False

        self._driver = make_driver(cfg)
        self._controller = HeadController(
            self._driver.send,
            rate_hz=float(cfg.get("head.rate_hz", 25.0)),
            max_speed_dps=float(cfg.get("head.max_speed_dps", 36.0)),
            limit_deg=float(cfg.get("head.limit_deg", 45.0)),
            gate=self._gate,
            home_x=float(cfg.get("head.home_x", 0.0)),
            home_y=float(cfg.get("head.home_y", 0.0)),
        )
        # Tilt's mechanical range is asymmetric and much smaller than pan's.
        # Install the calibrated window in the CONTROLLER too (not only the
        # driver's servo clamp) so belief can never sit outside what the
        # hardware can reach — else efference copy reports motion that never
        # happened and the tilt tracker winds up against the clamp (audit H5).
        tmin = cfg.get("head.tilt_min_deg", None)
        tmax = cfg.get("head.tilt_max_deg", None)
        if tmin is not None or tmax is not None:
            self._controller.set_calibration(
                y_min=float(tmin) if tmin is not None else None,
                y_max=float(tmax) if tmax is not None else None)
        self._tracker = FaceTracker(
            self._controller,
            kp_pan=float(cfg.get("head.tracker.kp_pan", 0.22)),
            kp_tilt=float(cfg.get("head.tracker.kp_tilt", 0.20)),
            deadband=float(cfg.get("head.tracker.deadband", 0.22)),
            pan_sign=float(cfg.get("head.tracker.pan_sign", -1.0)),
            tilt_sign=float(cfg.get("head.tracker.tilt_sign", -1.0)),
            half_fov_pan_deg=float(cfg.get("head.tracker.half_fov_pan_deg", 35.0)),
            half_fov_tilt_deg=float(cfg.get("head.tracker.half_fov_tilt_deg", 25.0)),
            max_offset_deg=float(cfg.get("head.limit_deg", 45.0)),   # ±pan range

            # idle look-around scan OFF by default — the 28ce9ee tuning war landed
            # on "hold, don't revolve" (a scanning head reads as agitated, and it
            # was leaking waypoints into commands). >0 re-enables it.
            scan_after_s=float(cfg.get("head.tracker.scan_after_s", 0.0)),
            settle_delay_s=float(cfg.get("head.tracker.settle_delay_s", 0.6)),
            lead_s=float(cfg.get("head.tracker.lead_s", 0.12)),
        )
        self._scheduler = GazeScheduler(
            sideways_deg=float(cfg.get("head.social.sideways_deg", 12.0)),
            up_deg=float(cfg.get("head.social.up_deg", 10.0)),
            max_mutual_s=float(cfg.get("head.social.max_mutual_s", 5.0)),
            think_hold_s=float(cfg.get("head.social.think_hold_s", 3.5)),
            rhythm={
                "listening": (float(cfg.get("head.social.listen_avert_s", 1.14)),
                              float(cfg.get("head.social.listen_face_frac", 0.71))),
                "speaking":  (float(cfg.get("head.social.speak_avert_s", 1.96)),
                              float(cfg.get("head.social.speak_face_frac", 0.41))),
            },
        )
        # eyes-lead-head: the neck stays still while the face offset is within
        # this fraction of the half-frame — the digital crop (the 'eye') covers
        # it. Beyond it, the neck engages. ~0.2 ≈ 15° at a 35° half-FOV.
        self._engage_frac = float(cfg.get("head.engage_frac", 0.2))
        # When false, the neck tracks horizontally only — the vertical offset is
        # ignored so a persistent low/high face (camera-mount offset the nod can't
        # fix) can't keep the tracker engaged and wind the tilt up.
        self._track_tilt = bool(cfg.get("head.track_tilt", True))
        # Input mode: "face" = closed-loop face tracking (default); "hand" =
        # direct hand teleoperation (body-relative wrist offset -> pan angle).
        self._input = str(cfg.get("head.input", "face")).lower()
        self._limit_deg = float(cfg.get("head.limit_deg", 45.0))
        self._hand = None
        self._hand_seen = 0.0
        self._hand_lost_s = float(cfg.get("head.hand.lost_s", 1.5))
        if self._input in ("hand", "head"):
            kpt = "head" if self._input == "head" else "wrist"
            # head-yaw signal has a smaller natural range than a hand sweep, so
            # it gets its own (higher) gain.
            gain = (float(cfg.get("head.hand.gain_head", 2.0)) if self._input == "head"
                    else float(cfg.get("head.hand.gain", 1.3)))
            self._hand = HandPoseSource(
                cfg.resolve_path("head.hand.model_path", "models/yolo11n-pose.onnx"),
                keypoint=kpt,
                conf=float(cfg.get("head.hand.conf", 0.5)),
                kp_conf=float(cfg.get("head.hand.kp_conf", 0.3)),
                min_shoulder_frac=float(cfg.get("head.hand.min_shoulder_frac", 0.10)),
                deadzone=float(cfg.get("head.hand.deadzone", 0.12)),
                min_cutoff=float(cfg.get("head.hand.min_cutoff", 1.5)),
                beta=float(cfg.get("head.hand.beta", 0.5)),
                mirror=bool(cfg.get("head.hand.mirror", True)),
                gain=gain,
                gain_y=float(cfg.get("head.hand.gain_y", 1.0)),
                deadzone_y=float(cfg.get("head.hand.deadzone_y", 0.15)))
        # Vertical teleop channel — kill switch + its own conservative range.
        # Ships dark: even with the nod driven, the follow stays horizontal
        # until head.hand.tilt is turned on after the supervised calibration.
        self._hand_tilt = bool(cfg.get("head.hand.tilt", False))
        self._hand_tilt_deg = float(cfg.get("head.hand.tilt_range_deg", 15.0))
        self._source_hz = float(cfg.get("head.source_hz", 15.0))
        self._suppress_lead_s = float(cfg.get("head.suppress_lead_s", 0.5))
        self._resettle_dwell_s = float(cfg.get("head.resettle_dwell_s", 0.4))

        # Stabilizer (efference copy by default; IMU-ready — attach a gyro reader
        # later with self._stab.attach_imu(...) and nothing else changes).
        self._stab = Stabilizer(
            moving_eps_dps=float(cfg.get("head.stab.moving_eps_dps", 3.0)),
            saccade_dps=float(cfg.get("head.stab.saccade_dps", 15.0)))

        self._src_thread = None
        self._src_stop = threading.Event()
        self._last_aim = (0.0, 0.0)
        self._moved = False
        self._still_since = 0.0
        self._prev_pos = (0.0, 0.0)
        self._prev_t = 0.0
        # A commanded gaze (voice fast-path or the LLM tool) temporarily overrides
        # face tracking: aim open-loop at a target for a dwell, then resume. This
        # is the SAME controller as tracking — no separate 'command mode'.
        self._cmd_target = None          # (pan, tilt) degrees, or None
        self._cmd_until = 0.0
        self._cmd_hold = False           # True = hold the commanded pose indefinitely
        self._cmd_dwell = float(cfg.get("head.command_dwell_s", 4.0))
        self._cmd_deg = float(cfg.get("head.command_deg", 25.0))
        # lightweight diagnostics surfaced via status()["dbg"]
        self._dbg = {"ticks": 0, "face": 0, "nowin": 0, "off": 0.0,
                     "branch": "-", "aim": (0.0, 0.0), "err": ""}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._controller.start()
        if self._hand is not None and self._eyes is not None:
            self._hand.start(self._eyes)     # background pose thread (~8 fps)
        # Always run the source loop — even with no camera it carries out
        # commanded gaze (voice/typed "turn left", "stop", ...). Head-FOLLOW and
        # face tracking need eyes and idle-home without them; commands do not.
        self._src_stop.clear()
        self._src_thread = threading.Thread(
            target=self._source_loop, name="head-source", daemon=True)
        self._src_thread.start()
        log.info("head system up (driver=%s moves_hardware=%s)",
                 type(self._driver).__name__, self._driver.moves_hardware)

    def stop(self) -> None:
        self._src_stop.set()
        if self._hand is not None:
            self._hand.stop()
        try:
            # Ease to home before releasing, and actually WAIT for it: the
            # gateway holds the last commanded pose forever (no watchdog), so
            # cutting the loop mid-travel leaves the robot staring sideways
            # until the next run (audit C3/M5). Bounded so shutdown never hangs.
            self._controller.center()
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.5:
                px, py = self._controller.position
                if abs(px) < 2.0 and abs(py) < 2.0:
                    break
                time.sleep(0.05)
        except Exception:
            pass
        self._controller.stop()
        try:
            self._driver.close()
        except Exception:
            pass

    # ── main.py surface ──────────────────────────────────────────────────────
    def set_state(self, state: str) -> None:
        """Forward a Zero._to() transition into the social layer."""
        self._scheduler.set_state(state, time.monotonic())

    def on_sentence_end(self) -> None:
        """A spoken sentence just ended — look back to hand over the turn."""
        self._scheduler.notice_sentence_end(time.monotonic())

    def express(self, text: str) -> None:
        """Fire any gesture cues embedded in a reply ([laughs], <sigh>, 'yes'…)."""
        if not text:
            return
        import re
        for tok in re.findall(r"[a-z]+", text.lower()):
            # gesture() ignores unknown names, so this is safe to spray.
            self._controller.gesture(tok)

    def gesture(self, name: str) -> None:
        self._controller.gesture(name)

    # ── commanded gaze (voice fast-path + LLM tool) ──────────────────────────
    def apply_command(self, cmd: dict) -> str:
        """Apply a parsed gaze intent; return a short sentence to speak. The move
        runs asynchronously — the control loop carries it out smoothly."""
        if not cmd:
            return ""
        kind = cmd.get("kind")
        if kind == "center":
            return self.look_center()
        if kind == "hold":
            return self.hold_here()
        if kind == "track":
            return self.resume_tracking()
        if kind == "direction":
            deg = float(cmd.get("degrees", self._cmd_deg))
            return self.look_direction(cmd.get("axis"), float(cmd.get("sign", 1.0)) * deg)
        if kind == "person":
            return self.look_person(cmd.get("name"))
        return ""

    def look_direction(self, axis: str, signed_deg: float, *, dwell=None) -> str:
        # Anchor the UN-commanded axis on where the head actually is now (its real
        # committed pose), not the tracker's internal aim — so 'look up' keeps the
        # current facing instead of inheriting stale tracker state.
        ax, ay = self._controller.position
        tgt = (signed_deg, ay) if axis == "pan" else (ax, signed_deg)
        self._set_command(tgt, dwell, hold=False)
        word = ({"pan": "left" if signed_deg < 0 else "right",
                 "tilt": "up" if signed_deg > 0 else "down"}).get(axis, "there")
        return f"Okay, looking {word}."

    def look_center(self, *, dwell=None) -> str:
        self._set_command((0.0, 0.0), dwell, hold=False)
        return "Okay, facing forward."

    def look_person(self, name, *, dwell=None) -> str:
        # Best-effort (phase 3): drop any directional override and let the face
        # tracker re-engage on the visible person; reset aversions so the gaze
        # lands and holds. True name→face binding (look at Sam specifically when
        # several faces are present) is phase 4/5.
        self._cmd_target = None
        self._cmd_until = 0.0
        self._cmd_hold = False
        self._scheduler.set_state(self._scheduler.state, time.monotonic())
        who = "you" if (not name or name == "me") else name.capitalize()
        return f"Okay, looking at {who}."

    def look_and_settle(self, cmd: dict, *, timeout: float = 1.2) -> str:
        """Apply a command and block until the neck reaches it (or timeout) — for
        the LLM's 'turn, look, THEN answer' (look_then_speak)."""
        msg = self.apply_command(cmd)
        tgt = self._cmd_target
        if tgt is not None:
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                px, py = self._controller.position
                if abs(px - tgt[0]) < 3.0 and abs(py - tgt[1]) < 3.0:
                    break
                time.sleep(0.03)
        return msg

    def hold_here(self) -> str:
        """Freeze the head where it is — stop the servos until told to move or
        track again. Full manual control: the neck stays put."""
        self._set_command(self._controller.position, hold=True)
        return "Okay, holding here."

    def resume_tracking(self) -> str:
        """Release any held/commanded pose and resume the active input mode
        (head/hand follow, or face tracking)."""
        self._cmd_target = None
        self._cmd_hold = False
        if self._hand is not None:
            self._hand.reset_filters()   # avoid a jump when the follow resumes
        self._scheduler.set_state(self._scheduler.state, time.monotonic())
        what = "following you" if self._input in ("hand", "head") else "tracking"
        return f"Okay, {what} again."

    def _set_command(self, target, dwell=None, hold=False) -> None:
        self._cmd_hold = bool(hold)
        d = float(dwell if dwell is not None else self._cmd_dwell)
        # Clamp into the controller's envelope so sentinel magnitudes (the
        # parser's FULL_DEG=999) never leak into state — look_and_settle and
        # status() must see a reachable target (audit M3).
        tx, ty = self._controller.clamp_to_envelope(float(target[0]),
                                                    float(target[1]))
        # Deadline BEFORE target: the source tick reads target-then-deadline,
        # so writing in this order can't expose a new target with the previous
        # (expired) deadline and drop the command (audit M2).
        self._cmd_until = time.monotonic() + d
        self._cmd_target = (tx, ty)
        if self._eyes is not None:
            try:
                self._eyes.suppress_changes((3.0 if hold else d) + 0.3)  # efference copy
            except Exception:
                pass

    def estop(self) -> None:
        self._estop = True
        try:
            self._driver.estop()
        except Exception:
            pass

    def resume(self) -> None:
        self._estop = False

    def set_calibration(self, **kw) -> None:
        self._controller.set_calibration(**kw)

    def attach_imu(self, reader) -> None:
        """Wire a gyro reader() -> (pan_dps, tilt_dps) deg/s for true VOR. Until
        one is attached, stabilization runs on efference copy (fine for a static
        base). Drop-in — nothing else changes."""
        self._stab.attach_imu(reader)

    @property
    def position(self):
        return self._controller.position

    def status(self) -> dict:
        x, y = self._controller.position
        return {"pan": round(x, 1), "tilt": round(y, 1),
                "state": self._scheduler.state,
                "driver": type(self._driver).__name__,
                "moves_hardware": self._driver.moves_hardware,
                # False = posts are failing: the hardware is NOT following
                # belief. None = the driver has no link-health notion (null/udp).
                "driver_healthy": getattr(self._driver, "healthy", None),
                "stabilizer": "imu" if self._stab.has_imu else "efference",
                "estop": self._estop,
                "dbg": dict(self._dbg)}

    # ── internals ────────────────────────────────────────────────────────────
    def _hand_tick(self, now: float) -> None:
        """Direct hand teleoperation: the body-relative wrist offset drives pan.
        The pose runs on its own thread; we read the latest smoothed value and
        set the target (the controller slews to it at rate_hz). When the hand is
        lost for a moment we ease back to home rather than freezing at an angle."""
        x, y, conf = (self._hand.value if self._hand is not None
                      else (0.0, 0.0, 0.0))
        if conf > 0.0:
            self._hand_seen = now
            ty = y * self._hand_tilt_deg if self._hand_tilt else 0.0
            aim = self._controller.clamp_to_envelope(x * self._limit_deg, ty)
            self._controller.set_target(*aim)
            self._last_aim = aim
            self._dbg["branch"] = "hand"
            self._dbg["aim"] = (round(aim[0], 1), round(aim[1], 1))
        elif now - self._hand_seen > self._hand_lost_s:
            self._controller.set_target(0.0, 0.0)   # hand gone -> recentre
            self._last_aim = (0.0, 0.0)
            self._dbg["branch"] = "hand-lost"
        # else: brief dropout within grace -> hold the last target

    def _gate(self) -> str:
        if self._estop:
            return "freeze"
        if self._ext_gate is not None:
            try:
                return self._ext_gate() or "track"
            except Exception:
                return "home"
        return "track"

    def _source_loop(self) -> None:
        """Pull the face target from Eyes' digital gaze, apply eyes-lead-head +
        the social bias, and drive the tracker/controller. ~15 Hz."""
        dt = 1.0 / max(1.0, self._source_hz)
        while not self._src_stop.is_set():
            t0 = time.monotonic()
            try:
                self._source_tick(time.monotonic())
            except Exception as e:      # the head must never crash perception
                self._dbg["err"] = repr(e)[:100]
                log.debug("head source tick failed: %s", e)
            self._efference_copy(time.monotonic())
            wait = dt - (time.monotonic() - t0)
            if wait > 0:
                self._src_stop.wait(wait)

    def _source_tick(self, now: float) -> None:
        # A commanded gaze (voice fast-path or the LLM tool) overrides EVERYTHING
        # — face tracking OR hand/head teleoperation — for its dwell, then control
        # returns to the active mode. Same controller, open-loop toward the target.
        # This is why "turn far left" works even while you are head-controlling.
        if self._cmd_target is not None:
            if self._cmd_hold or now < self._cmd_until:
                self._controller.set_target(*self._cmd_target)
                self._last_aim = self._cmd_target
                self._dbg["branch"] = "hold" if self._cmd_hold else "command"
                return
            self._cmd_target = None       # dwell elapsed -> resume the input mode
        if self._input in ("hand", "head"):
            self._hand_tick(now)
            return
        bias = self._scheduler.tick(now)
        if not bias.on_face:
            # Aversion (or thinking): hold the last face aim plus a small offset,
            # open-loop. Don't run the tracker — we're deliberately not tracking.
            ox, oy = bias.offset_deg
            ax, ay = self._last_aim
            self._controller.set_target(ax + ox, ay + oy)
            return

        self._dbg["ticks"] += 1
        eyes = self._eyes
        win = eyes.attention() if eyes is not None else None
        frame = eyes.current_frame() if eyes is not None else None
        if win is None or frame is None:
            self._dbg["nowin"] += 1
            self._dbg["branch"] = "no-win" if win is None else "no-frame"
            self._tracker.update(None, 0, 0)     # scan / ease home
            self._last_aim = self._tracker.aim
            return
        self._dbg["face"] += 1
        h, w = frame.shape[:2]
        x, y, ww, hh = win
        cx, cy = x + ww / 2.0, y + hh / 2.0
        # Eyes-lead-head: if the face sits comfortably near frame centre, the
        # digital crop already has it — hold the neck (feed a centred target so
        # the tracker latches). Only engage the neck past the threshold.
        if not self._track_tilt:
            cy = h / 2.0                      # horizontal-only: ignore vertical
        off = (abs(cx / w - 0.5) if not self._track_tilt
               else max(abs(cx / w - 0.5), abs(cy / h - 0.5))) * 2.0   # 0..1
        self._dbg["off"] = round(off, 3)
        if off < self._engage_frac:
            # Face near centre — the digital crop covers it; hold the neck. Do NOT
            # feed the tracker a centred target: that latches its hysteresis and
            # then modest real offsets can't un-latch it. Just hold the current aim.
            self._dbg["branch"] = "hold(eye)"
            self._last_aim = self._tracker.aim
            return
        self._dbg["branch"] = "engage"
        self._tracker.update((cx, cy, ww, hh), w, h)
        self._last_aim = self._tracker.aim
        self._dbg["aim"] = (round(self._tracker.aim[0], 1),
                            round(self._tracker.aim[1], 1))

    def _efference_copy(self, now: float) -> None:
        """Make commanded neck motion visible to the vision layer so it never
        reports false novelty ('a shelf appeared') after ZERO merely looked
        elsewhere. Watches the controller's committed position; while it moves,
        suppress; once still after a move, re-settle the baseline once."""
        px, py = self._controller.position
        ppx, ppy = self._prev_pos
        dt = max(1e-3, now - self._prev_t) if self._prev_t else 0.05
        self._prev_pos = (px, py)
        self._prev_t = now
        # commanded camera angular velocity (deg/s) — the efference-copy estimate
        pan_dps, tilt_dps = (px - ppx) / dt, (py - ppy) / dt
        self._stab.note_commanded(pan_dps, tilt_dps)
        if self._eyes is not None:
            try:
                self._eyes.set_ego_motion(pan_dps, tilt_dps)
            except Exception:
                pass
        if self._eyes is None:
            return
        moving = self._stab.is_moving()
        if moving:
            try:
                self._eyes.suppress_changes(self._suppress_lead_s)
            except Exception:
                pass
            self._moved = True
            self._still_since = 0.0
        elif self._moved:
            if self._still_since == 0.0:
                self._still_since = now
            elif now - self._still_since >= self._resettle_dwell_s:
                try:
                    self._eyes.resettle()
                except Exception:
                    pass
                self._moved = False
