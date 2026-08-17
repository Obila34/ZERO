"""FaceTracker — turn a face's position in the frame into a head aim.

The camera rides ON the head, so this is a closed loop: when a face sits off to
the right of the frame, the head turns right, which brings the face back toward
centre, which shrinks the error to zero. That makes a simple integral controller
both stable and self-correcting — the aim accumulates toward whatever keeps the
face centred, then stops. A deadband around centre stops micro-hunting, and when
the face is lost the aim eases back to neutral rather than freezing mid-turn.

Pure logic — no cv2, no I/O. It reads face centres (the source finds them) and
pushes an aim to the HeadController; unit-testable by feeding synthetic faces.
"""
from __future__ import annotations

import time

from zero.utils.logging import get_logger

log = get_logger("head.tracker")


class FaceTracker:
    def __init__(self, controller, *, kp_pan: float = 0.45, kp_tilt: float = 0.40,
                 half_fov_pan_deg: float = 35.0, half_fov_tilt_deg: float = 25.0,
                 deadband: float = 0.06, max_offset_deg: float = 45.0,
                 pan_sign: float = 1.0, tilt_sign: float = 1.0,
                 recenter_after_s: float = 4.0, recenter_decay: float = 0.9,
                 scan_after_s: float = 6.0, scan_dwell_s: float = 4.0,
                 scan_max_cycles: int = 3,
                 sat_reset_after_s: float = 2.0, sat_reset_pause_s: float = 1.0,
                 micro_idle_period_s: float = 32.0,
                 micro_idle_return_s: float = 1.5,
                 settle_delay_s: float = 0.30, lead_s: float = 0.30):
        self._c = controller
        self._kp_pan = float(kp_pan)
        self._kp_tilt = float(kp_tilt)
        self._hfov_pan = float(half_fov_pan_deg)
        self._hfov_tilt = float(half_fov_tilt_deg)
        self._deadband = float(deadband)
        self._max = float(max_offset_deg)
        # Calibration signs: +pan_sign turns toward a face on the right; +tilt_sign
        # looks UP toward a face high in the frame. Flip in config if a hardware
        # axis is mounted mirrored.
        self._pan_sign = float(pan_sign)
        self._tilt_sign = float(tilt_sign)
        self._recenter_after = float(recenter_after_s)
        self._recenter_decay = float(recenter_decay)
        self._tx = 0.0            # accumulated aim (degrees)
        self._ty = 0.0
        self._last_face_t = 0.0
        self._dbg_t = 0.0         # throttle for the convergence trace
        # Hysteresis hold: once centred within `deadband`, LATCH and hold still
        # until the face drifts past `hold_band` — a stationary (or slightly
        # jittering) person is held rock-steady, not chased/jerked; a real move
        # re-engages. hold_band ≈ 2.3× the deadband.
        self._settled = False
        self._hold_band = max(0.12, float(deadband) * 2.3)
        # Settle-gate: after issuing an aim, WAIT until the head reaches it AND a
        # short beat passes (so the laggy cloud-jogged neck + camera catch up)
        # before correcting again — else corrections pile onto an unreached target
        # and the head winds up / hunts (the ±40° 'revolving').
        self._settle_t = 0.0
        # Beat between corrections. On a laggy actuator (stepper pan + hobby nod,
        # with a second smoothing stage in the reflex service), too tight a beat
        # over-drives: the tracker keeps correcting before the physical head — and
        # thus the camera — has caught up, so the error never clears and the aim
        # winds up. Wait long enough for the head+camera to settle before the next
        # correction.
        self._settle_delay = float(settle_delay_s)
        self._lead_s = float(lead_s)
        # Predictive lead: the cloud-jogged neck LAGS the face, so aim where the
        # face is HEADING, not where it was. Smoothed velocity of the frame-offset
        # drives a small, clamped lead — cancels the lag without letting a noisy
        # frame fling the aim.
        self._pred_ex = 0.0
        self._pred_ey = 0.0
        self._pred_t = 0.0
        self._vx = 0.0
        self._vy = 0.0
        # SCAN state (face-lost look-around). Waypoints sweep the workspace
        # after `scan_after_s` of no face; after `scan_max_cycles` full sweeps
        # we PARK at 0,0 and stop issuing set_target calls until a face returns.
        self._scan_after = float(scan_after_s)
        self._scan_dwell = float(scan_dwell_s)
        self._scan_max_cycles = int(scan_max_cycles)
        self._SCAN_WAYPOINTS = ((-20.0, 0.0), (0.0, 0.0), (20.0, 0.0), (0.0, 0.0))
        self._scan_idx = 0
        self._scan_next_t = 0.0
        self._scan_cycles = 0
        self._scan_active = False
        # SATURATION micro-reset — track continuous fight at ±max_offset when
        # |ex|/|ey| KEEPS growing. After sat_reset_after_s snap to 0,0 + drop
        # the latch + pause `sat_reset_pause_s` so neck/camera catch up.
        self._sat_reset_after = float(sat_reset_after_s)
        self._sat_reset_pause = float(sat_reset_pause_s)
        self._sat_since = 0.0
        self._prev_ex_abs = 0.0
        self._prev_ey_abs = 0.0
        self._sat_pause_until = 0.0
        # MICRO-IDLE nudge — while _settled and the head is AT its target,
        # fire a tiny offset from a fixed table every micro_idle_period_s,
        # hold micro_idle_return_s, snap back. The nudges themselves are
        # cheap servo moves — no voice, no rate cap impact.
        self._micro_period = float(micro_idle_period_s)
        self._micro_return = float(micro_idle_return_s)
        self._MICRO_TABLE = ((3.0, 0.0), (-3.0, 0.0), (0.0, 2.0),
                             (0.0, -2.0), (2.0, -2.0), (-2.0, 2.0))
        self._micro_idx = 0
        self._micro_next_t = 0.0
        self._micro_phase = "idle"     # idle | nudging
        self._micro_return_t = 0.0
        self._micro_saved: tuple[float, float] | None = None
        # Baselines captured ONCE so set_attention() multipliers are idempotent —
        # 100 successive at_me calls stay at deadband * 0.5, not decay to zero.
        # Placed at the end of __init__ so any subsequent init-time mutation of
        # kp_pan/kp_tilt/deadband is captured too.
        self._kp_pan_base = self._kp_pan
        self._kp_tilt_base = self._kp_tilt
        self._deadband_base = self._deadband
        self._attention = "unknown"

    def set_attention(self, state: str) -> None:
        """Adapt gains + deadband to the CURRENT SESSION SPEAKER's gaze.

        Called by the sentience layer (FaceAnalysisWorker after 5-frame majority
        vote — reviewer MAJOR #4) on state CHANGES only. State labels:

          at_me     — direct gaze → deadband * 0.5, kp * 1.10 (tighter follow)
          away      — looking off → deadband * 2.0, kp * 0.80 (loose glances OK)
          sideways  — half-turn   → baseline (don't fight a natural profile)
          unknown   — no analysis → baseline

        Idempotent: re-issuing the same state is a no-op. Recomputes from
        captured baselines each call so 100 successive at_me calls do NOT
        decay deadband to zero. hold_band tracks deadband so the settled
        behavior stays the same shape after a change.

        Reviewer MAJOR #4 LOCKOUT: while `_settled` is True (the tracker is
        holding on a still face), refuse to change the gains — the majority
        vote has done its job upstream, but a mid-hold gain shift would
        recompute hold_band and trip the settled latch. The next unlatched
        moment applies the pending state cleanly.
        """
        state = str(state or "unknown").lower()
        if state == self._attention:
            return
        # LOCKOUT during a held gaze — do not disturb the latch. The pending
        # state is discarded (attention is fresh every pulse), so the next
        # unlatched moment naturally accepts the current state.
        if getattr(self, "_settled", False):
            log.debug("attention pending %s (locked during hold)", state)
            return
        multipliers = {
            "at_me":    (0.5, 1.10),
            "away":     (2.0, 0.80),
            "sideways": (1.0, 1.00),
            "unknown":  (1.0, 1.00),
        }
        db_mul, kp_mul = multipliers.get(state, (1.0, 1.00))
        self._deadband = self._deadband_base * db_mul
        self._kp_pan = self._kp_pan_base * kp_mul
        self._kp_tilt = self._kp_tilt_base * kp_mul
        # hold_band ≈ 2.3× the (new) deadband, floored so hysteresis remains real.
        self._hold_band = max(0.12, self._deadband * 2.3)
        self._attention = state
        log.info("attention -> %s (deadband=%.3f kp=(%.2f,%.2f))",
                 state, self._deadband, self._kp_pan, self._kp_tilt)

    def _clamp(self, v: float) -> float:
        return self._max if v > self._max else (-self._max if v < -self._max else v)

    def update(self, face, frame_w: int, frame_h: int) -> None:
        """`face` = (cx, cy, w, h) in frame pixels, or None when no face is seen."""
        now = time.monotonic()
        if face is None:
            # Lost the face: hold briefly (they may just have turned away), then
            # ease the aim home so the head doesn't sit frozen at a hard angle.
            elapsed = now - self._last_face_t
            if elapsed > self._recenter_after:
                self._tx *= self._recenter_decay
                self._ty *= self._recenter_decay
                if abs(self._tx) < 0.3 and abs(self._ty) < 0.3:
                    self._tx = self._ty = 0.0
                self._c.set_target(self._tx, self._ty)
            # SCAN state (Design 3): after scan_after_s of no face, sweep pan
            # waypoints so ZERO visibly LOOKS AROUND instead of freezing. After
            # scan_max_cycles full sweeps with no acquisition, PARK at 0,0.
            if (self._scan_after > 0 and elapsed > self._scan_after
                    and self._scan_cycles < self._scan_max_cycles):
                if not self._scan_active:
                    self._scan_active = True
                    self._scan_idx = 0
                    self._scan_next_t = 0.0
                    log.info("head: no face for %.1fs — scanning", elapsed)
                if now >= self._scan_next_t:
                    wx, wy = self._SCAN_WAYPOINTS[self._scan_idx]
                    self._tx, self._ty = wx, wy
                    self._c.set_target(wx, wy)
                    self._scan_next_t = now + self._scan_dwell
                    self._scan_idx += 1
                    if self._scan_idx >= len(self._SCAN_WAYPOINTS):
                        self._scan_idx = 0
                        self._scan_cycles += 1
                        if self._scan_cycles >= self._scan_max_cycles:
                            self._tx = self._ty = 0.0
                            self._c.set_target(0.0, 0.0)
                            log.info("head: scan gave up — parked at 0,0")
            return
        # Face re-acquired: cancel any active scan and reset the cycle counter
        # so the next lost-face event starts fresh.
        if self._scan_active or self._scan_cycles:
            self._scan_active = False
            self._scan_idx = 0
            self._scan_cycles = 0

        self._last_face_t = now
        cx, cy, _w, _h = face
        if frame_w <= 0 or frame_h <= 0:
            return
        ex = cx / frame_w - 0.5     # -0.5 (left) .. +0.5 (right)
        ey = cy / frame_h - 0.5     # -0.5 (top)  .. +0.5 (bottom)

        # Predictive lead — anticipate the face's motion to cancel the neck's lag.
        # Smoothed frame-offset velocity, projected over a short horizon, clamped
        # so a jittery detection can't overshoot. A stationary face has ~0 velocity
        # → no lead → it still latches and holds rock-steady below.
        dt = now - self._pred_t
        if 0.0 < dt < 0.5:
            self._vx = 0.5 * self._vx + 0.5 * ((ex - self._pred_ex) / dt)
            self._vy = 0.5 * self._vy + 0.5 * ((ey - self._pred_ey) / dt)
        self._pred_ex, self._pred_ey, self._pred_t = ex, ey, now
        ex += max(-0.15, min(0.15, self._vx * self._lead_s))
        ey += max(-0.15, min(0.15, self._vy * self._lead_s))

        # Hysteresis hold: LATCH a hold once the face is centred within the
        # deadband, and keep holding until it drifts past hold_band. This makes a
        # STATIONARY person rock-steady — the head stops dead once centred and
        # won't chase sub-band jitter; only a real move (past hold_band) re-engages
        # tracking. Fixes the "jerks and passes me when I'm still".
        mag = abs(ex) if abs(ex) > abs(ey) else abs(ey)
        if self._settled:
            if mag > self._hold_band:
                self._settled = False       # they moved — track again
        elif mag < self._deadband:
            self._settled = True            # centred tight — latch + hold
        if self._settled:
            # MICRO-IDLE nudge — a tiny, timer-driven offset that returns to the
            # pre-idle aim. Turns 'catatonic latch' into 'quiet breathing'. If the
            # face moves and unlatches us (mag > hold_band), the nudge is dropped
            # without a return-snap — the tracker re-aims anyway.
            if self._micro_phase == "idle":
                if self._micro_next_t == 0.0:
                    self._micro_next_t = now + self._micro_period
                elif now >= self._micro_next_t:
                    dx, dy = self._MICRO_TABLE[self._micro_idx]
                    self._micro_idx = (self._micro_idx + 1) % len(self._MICRO_TABLE)
                    self._micro_saved = (self._tx, self._ty)
                    self._tx = self._clamp(self._tx + dx)
                    self._ty = self._clamp(self._ty + dy)
                    self._c.set_target(self._tx, self._ty)
                    self._micro_phase = "nudging"
                    self._micro_return_t = now + self._micro_return
                    log.debug("head: micro-nudge %+.1f,%+.1f", dx, dy)
            elif self._micro_phase == "nudging" and now >= self._micro_return_t:
                if self._micro_saved is not None:
                    sx, sy = self._micro_saved
                    self._tx, self._ty = sx, sy
                    self._c.set_target(sx, sy)
                    self._micro_saved = None
                self._micro_phase = "idle"
                self._micro_next_t = now + self._micro_period
            ex = ey = 0.0
        else:
            # Unlatched — reset the nudge clock so it re-arms on the NEXT settle.
            if self._micro_saved is not None:
                self._micro_saved = None
            self._micro_phase = "idle"
            self._micro_next_t = 0.0

        # SETTLE-GATED closed-loop servo. The neck is jogged over the cloud, so it
        # LAGS ZERO's committed position `_cur`. If we keep adding corrections while
        # it's still travelling to the last target, they wind up and the head hunts
        # (the ±40° 'revolving'). So issue a NEW correction only once the head has
        # REACHED the last aim AND a settle beat has passed (neck + camera caught
        # up); otherwise HOLD. One measured step at a time → converges and holds,
        # no windup. Face ABOVE centre (ey<0) → look UP (+head_y).
        base_x, base_y = self._c.position
        reached = abs(base_x - self._tx) < 2.5 and abs(base_y - self._ty) < 2.5
        if not reached:
            self._settle_t = now                 # still travelling — reset the beat
            return                               # HOLD the current aim
        if now - self._settle_t < self._settle_delay:
            return                               # reached; let the neck + camera settle
        if now < self._sat_pause_until:
            # Post-saturation settle. This used to guard only the saturation
            # bookkeeping at the end of update(), so corrections kept flowing
            # and the aim walked straight back to the limit it had just been
            # rescued from — wind up, snap home, wind up again. Hold for real.
            return
        if ex == 0.0 and ey == 0.0:
            return                               # centred (deadband/hysteresis) → hold
        self._tx = self._clamp(base_x + self._pan_sign  * self._kp_pan  * ex * self._hfov_pan  * 2.0)
        self._ty = self._clamp(base_y - self._tilt_sign * self._kp_tilt * ey * self._hfov_tilt * 2.0)
        self._c.set_target(self._tx, self._ty)
        self._settle_t = now                     # started a new move — reset the beat

        # SATURATION micro-reset (Design 3). If the aim is pinned at ±max AND the
        # frame-error is GROWING vs last frame, we're fighting a wall (person
        # beyond mechanical range, mis-mounted, or a sign genuinely inverted).
        # After sat_reset_after_s of continuous fight, snap to 0,0, drop the
        # latch, and pause the loop briefly so neck+camera settle. Every reset
        # cycle IS motion, so ZERO never looks dead even if the underlying
        # problem persists.
        clamped = (abs(self._tx) >= self._max - 0.1
                   or abs(self._ty) >= self._max - 0.1)
        # NOT-SHRINKING, not merely growing. The original test only fired when
        # the error was getting worse, so the commonest runaway slipped past it:
        # a FROZEN target (face lost, window held) holds the error dead
        # constant while the head walks to the limit — measured 2026-08-17 as
        # ex pinned at +0.203 across a 0°->80° sweep, with no reset. Anything
        # that isn't actually converging counts as a fight now.
        growing = (abs(ex) > self._prev_ex_abs - 1e-3
                   or abs(ey) > self._prev_ey_abs - 1e-3)
        self._prev_ex_abs = abs(ex)
        self._prev_ey_abs = abs(ey)
        if now < self._sat_pause_until:
            return                                # still in the settle pause
        if clamped and growing:
            if self._sat_since == 0.0:
                self._sat_since = now
            elif now - self._sat_since >= self._sat_reset_after:
                log.warning("head: saturation fight at (%.1f,%.1f) ex=%+.3f "
                            "ey=%+.3f — snapping to home",
                            self._tx, self._ty, ex, ey)
                self._tx = self._ty = 0.0
                self._c.set_target(0.0, 0.0)
                self._settled = False
                self._sat_since = 0.0
                self._sat_pause_until = now + self._sat_reset_pause
        else:
            self._sat_since = 0.0

        # Convergence trace (1 Hz). A correct loop drives ex,ey → 0 as the head
        # moves. If ex GROWS while the head turns (aim runs to the ±limit and the
        # face slides off-frame), the pan axis is inverted → flip
        # head.tracker.pan_sign (tilt → tilt_sign). This is the one-look sign test.
        if now - self._dbg_t > 1.0:
            self._dbg_t = now
            log.info("servo: ex=%+.3f ey=%+.3f | head=%.1f,%.1f -> aim=%.1f,%.1f",
                     ex, ey, base_x, base_y, self._tx, self._ty)

    @property
    def aim(self) -> tuple[float, float]:
        return self._tx, self._ty
