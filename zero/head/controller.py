"""HeadController — the smooth motion core for ZERO's pan/tilt head.

Everything the head does resolves to two signed-degree axes (af1AxisMap.ts):
  head_x = pan / yaw   (turn left-right)
  head_y = nod / pitch (look up-down)

The controller runs its own tick loop (~25 Hz) and is the ONE place that emits
motion, so tracking and expression never fight over the servos:

  * a LIVE baseline  — where the face tracker wants the head aimed (set_target),
  * a GESTURE overlay — a short keyframed offset ADDED on top (nod, tilt, laugh…)
    so ZERO can express WHILE still facing the person,

composited, slew-rate limited (never jerky — the "smooth" requirement), clamped
to the ±limit envelope, and pushed to a sink (the AF1 mirror link). Pure logic:
no cv2, no sockets — the sink is injected, so this is unit-testable headless.
"""
from __future__ import annotations

import threading
import time

from zero.utils.logging import get_logger

log = get_logger("head.controller")

# Conversation-linked gestures: keyframes of (Δpan, Δtilt, hold_seconds) offsets
# ADDED to the live aim, played in sequence then released back to 0. Small by
# design (≤8°) so ZERO reads as expressive, never a bobblehead. The emotion names
# mirror af1HeadGesture.ts POSES (gemma4 emits <laugh>/<sigh>/… inline); the rest
# are conversational beats ZERO fires itself (nod on "yes", tilt when curious…).
GESTURES: dict[str, list[tuple[float, float, float]]] = {
    # deliberate conversational beats
    "nod":     [(0, -6, 0.16), (0, 3, 0.16), (0, -4, 0.14), (0, 0, 0.12)],
    "shake":   [(-6, 0, 0.14), (6, 0, 0.16), (-5, 0, 0.14), (0, 0, 0.12)],
    "tilt":    [(6, 3, 0.45), (6, 3, 0.30), (0, 0, 0.20)],   # curious hold
    "think":   [(-7, 5, 0.55), (-5, 4, 0.30), (0, 0, 0.22)],  # look up-and-away
    "greet":   [(0, 4, 0.22), (0, -3, 0.18), (0, 0, 0.14)],   # small up-nod hello
    # emotion tags from the LLM (same table AF1 uses)
    "laugh":   [(0, 6, 0.55), (2, 3, 0.30), (0, 0, 0.16)],
    "chuckle": [(4, 2, 0.45), (0, 0, 0.16)],
    "sigh":    [(-3, -4, 0.55), (0, 0, 0.22)],
    "gasp":    [(0, 8, 0.35), (0, 4, 0.20), (0, 0, 0.16)],
    "cough":   [(0, -3, 0.28), (0, 0, 0.16)],
}
# Aliases so ZERO's cue words ([laughs], [hmm], …) and plain sentiment words map
# onto a gesture without the caller special-casing anything.
_ALIASES = {"laughs": "laugh", "chuckles": "chuckle", "sighs": "sigh",
            "gasps": "gasp", "coughs": "cough", "hmm": "think", "hmms": "think",
            "thinks": "think", "nods": "nod", "tilts": "tilt", "greets": "greet",
            "yes": "nod", "agree": "nod", "no": "shake", "curious": "tilt",
            "hello": "greet", "hi": "greet"}


def _clamp(v: float, lim: float) -> float:
    return lim if v > lim else (-lim if v < -lim else v)


def _clamp_range(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class HeadController:
    def __init__(self, send, *, rate_hz: float = 25.0, max_speed_dps: float = 90.0,
                 limit_deg: float = 45.0, gate=None, home_x: float = 0.0,
                 home_y: float = 0.0):
        """`send(head_x, head_y)` is the motion sink (MirrorLink.send). rate_hz is
        the tick rate; max_speed_dps caps how fast either axis may move (the slew
        limit that keeps motion smooth); limit_deg is the ± soft envelope about the
        calibrated home (home_x, home_y).

        `gate()` — the AF1 motion gate — returns each tick what the head is allowed
        to do: "track" (follow + express), "home" (locked / not checked in → ease
        to the calibrated home and hold), "yield" (AF1 owns the head → send
        nothing, don't fight it), or "freeze" (e-stop → stop dead). Default None =
        always "track" (backward-compatible)."""
        self._send = send
        self._dt = 1.0 / max(1.0, float(rate_hz))
        self._max_step = float(max_speed_dps) * self._dt   # deg per tick
        self._limit = float(limit_deg)
        # Per-axis soft-limit window [min, max] — symmetric ±limit by default, but
        # set_calibration() can install the operator's asymmetric saved limits
        # (head_x pan and head_y tilt have different ranges).
        self._xmin = self._ymin = -self._limit
        self._xmax = self._ymax = self._limit
        self._gate = gate
        self._home_x = float(home_x)
        self._home_y = float(home_y)
        self._lock = threading.Lock()
        self._live_x = 0.0        # tracker's desired aim (baseline)
        self._live_y = 0.0
        self._cur_x = 0.0         # committed position (slewed toward target)
        self._cur_y = 0.0
        self._gesture: list[tuple[float, float, float]] | None = None
        self._g_start = 0.0
        self._g_count = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── inputs (thread-safe) ──────────────────────────────────────────────────
    def set_target(self, head_x: float, head_y: float) -> None:
        """Aim the head here (the face tracker's output). Slew does the smoothing."""
        with self._lock:
            self._live_x = _clamp_range(head_x, self._xmin, self._xmax)
            self._live_y = _clamp_range(head_y, self._ymin, self._ymax)

    def center(self) -> None:
        self.set_target(0.0, 0.0)

    def gesture(self, name: str, now: float | None = None) -> None:
        """Play a conversational/emotion gesture as an overlay on the live aim.
        Unknown names are ignored (so raw LLM tags are safe to pass straight in)."""
        key = (name or "").strip().lower()
        key = _ALIASES.get(key, key)
        frames = GESTURES.get(key)
        if not frames:
            return
        with self._lock:
            self._g_count += 1
            # Tiny deterministic jitter (no RNG) so repeats don't look identical.
            j = ((self._g_count * 37) % 7 - 3) * 0.4   # ~[-1.2, +1.2]°
            self._gesture = [(dx + j, dy, hold) for (dx, dy, hold) in frames]
            self._g_start = time.monotonic() if now is None else now

    # ── loop ──────────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="head-ctrl", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _gesture_offset(self, now: float) -> tuple[float, float]:
        """Current (Δpan, Δtilt) from the playing gesture, advancing by elapsed
        time; (0,0) when nothing is playing or it just finished."""
        g = self._gesture
        if g is None:
            return 0.0, 0.0
        t = now - self._g_start
        acc = 0.0
        for dx, dy, hold in g:
            if t <= acc + hold:
                return dx, dy
            acc += hold
        self._gesture = None   # done — release the overlay
        return 0.0, 0.0

    def set_calibration(self, *, limit_deg=None, home_x=None, home_y=None,
                        x_min=None, x_max=None, y_min=None, y_max=None) -> None:
        """Apply the robot's head calibration — the soft-limit envelope and the
        home (rest) pose the head returns to when locked. Refreshable at runtime
        (e.g. after fetching the operator's saved per-axis limits). `limit_deg`
        sets a symmetric ±window; `x_min/x_max/y_min/y_max` set the operator's
        asymmetric saved window per axis (pan and tilt differ)."""
        with self._lock:
            if limit_deg is not None:
                self._limit = float(limit_deg)
                self._xmin = self._ymin = -self._limit
                self._xmax = self._ymax = self._limit
            if x_min is not None:
                self._xmin = float(x_min)
            if x_max is not None:
                self._xmax = float(x_max)
            if y_min is not None:
                self._ymin = float(y_min)
            if y_max is not None:
                self._ymax = float(y_max)
            if home_x is not None:
                self._home_x = _clamp_range(float(home_x), self._xmin, self._xmax)
            if home_y is not None:
                self._home_y = _clamp_range(float(home_y), self._ymin, self._ymax)

    def _step(self, now: float, state: str = "track") -> tuple[bool, float, float]:
        """One control step under gate `state`. Returns (should_send, pan, tilt).
        Pure + deterministic (takes the clock + state), so the loop and the tests
        share exactly this logic.
          track  — live aim + gesture overlay, slewed (follow + express).
          home   — ease to the calibrated home and hold (locked / not checked in).
          yield  — hold, send nothing (AF1 owns the head; don't fight it).
          freeze — e-stop: stop dead, send nothing.
        """
        with self._lock:
            if state in ("freeze", "yield"):
                return False, self._cur_x, self._cur_y
            if state == "home":
                tx = _clamp_range(self._home_x, self._xmin, self._xmax)
                ty = _clamp_range(self._home_y, self._ymin, self._ymax)
            else:  # track
                gx, gy = self._gesture_offset(now)
                tx = _clamp_range(self._live_x + gx, self._xmin, self._xmax)
                ty = _clamp_range(self._live_y + gy, self._ymin, self._ymax)
            # Slew-rate limit each axis toward the target — the smoothing.
            self._cur_x += _clamp(tx - self._cur_x, self._max_step)
            self._cur_y += _clamp(ty - self._cur_y, self._max_step)
            return True, self._cur_x, self._cur_y

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            state = "track"
            if self._gate is not None:
                try:
                    state = self._gate() or "track"
                except Exception:
                    state = "home"   # gate error → fail safe (ease home, don't follow)
            should_send, cx, cy = self._step(t0, state)
            if should_send:
                try:
                    self._send(cx, cy)
                except Exception as e:   # a dead sink must never kill the loop
                    log.debug("head send failed: %s", e)
            dt = self._dt - (time.monotonic() - t0)
            if dt > 0:
                self._stop.wait(dt)

    # ── introspection (for tests / status) ───────────────────────────────────
    @property
    def position(self) -> tuple[float, float]:
        with self._lock:
            return self._cur_x, self._cur_y
