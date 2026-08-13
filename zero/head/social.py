"""GazeScheduler — the social layer that makes gaze turn-taking machinery.

Where FaceTracker decides WHERE the addressee is, this decides WHETHER ZERO is
looking at them right now, and when to look away — the difference between a
camera locked on a face (unsettling) and a conversational partner.

It maps ZERO's conversation state onto measured human gaze behaviour:

  LISTENING — gaze at the partner ~71% of the time (Argyle & Cook), broken by
              brief sideways aversions (~1.1 s every ~7 s; Andrist et al. 2014).
  SPEAKING  — gaze at the partner only ~41%; more frequent/longer aversions
              (~2.0 s every ~4.75 s), and a deliberate look-BACK as a sentence
              ends to hand over the turn (Kendon 1967).
  THINKING  — a real planning pause: look away and UP and hold (cognitive
              aversion ~3.5 s; the "look up-and-away" beat).
  IDLE      — no addressee; the alive layer (FaceTracker micro-idle) carries it.

And two hard rules: never hold unbroken mutual gaze past ~5 s (preferred ~3.3 s,
Binetti 2016), and never whip — aversions are small offsets the smoothing turns
into gentle glances.

Pure logic + deterministic: tick(now, state) takes the clock, and aversion
timing uses a deterministic jitter (no RNG), so it is unit-testable headless and
never surprises a test. All numbers are config-overridable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GazeBias:
    """One tick of social intent, composited onto the tracker's aim by HeadSystem.

    on_face   — True: look at the addressee (let the tracker drive).
                False: avert — add `offset_deg` to the last face aim, open-loop.
    offset_deg — (dpan, dtilt) degrees to add while averting; (0,0) on face.
    reason    — short tag for logs/tests ("face", "listen-avert", "think",
                "turn-yield", "idle").
    """
    on_face: bool
    offset_deg: tuple[float, float]
    reason: str


# Per-state rhythm: (aversion_duration_s, face_fraction). The face fraction is
# the target — Argyle & Cook's 71% listening / 41% speaking, the brief's headline
# asymmetry — and the gap between aversions is DERIVED from it so the long-run
# fraction is exact: gap = dur * f / (1 - f). The durations are Andrist et al.
# 2014 means. (Andrist's own cadence implies ~86% face-time; we keep his aversion
# shape but honour Argyle's ratio, which the brief prioritises.)
_RHYTHM = {
    "listening": (1.14, 0.71),
    "speaking":  (1.96, 0.41),
}


class GazeScheduler:
    def __init__(self, *, sideways_deg: float = 12.0, up_deg: float = 10.0,
                 max_mutual_s: float = 5.0, think_hold_s: float = 3.5,
                 rhythm: dict | None = None):
        self._sideways = float(sideways_deg)
        self._up = float(up_deg)
        self._max_mutual = float(max_mutual_s)
        self._think_hold = float(think_hold_s)
        self._rhythm = dict(_RHYTHM)
        if rhythm:
            self._rhythm.update(rhythm)
        self._state = "idle"
        self._state_since = 0.0
        self._face_since = 0.0        # when the current on-face stretch began
        self._avert_until = 0.0       # >now while an aversion episode plays
        self._next_avert_t = 0.0      # when the next aversion may start
        self._episode = 0             # counts aversions, drives side alternation

    def set_state(self, state: str, now: float) -> None:
        """Called on every Zero._to() transition. Resets the episode clock so a
        new state starts with a fresh rhythm (and SPEAKING/THINKING get their
        characteristic opening beat)."""
        state = str(state or "idle").lower()
        if state == self._state:
            return
        self._state = state
        self._state_since = now
        self._face_since = now
        self._avert_until = 0.0
        # Schedule the first aversion one full gap out, so a turn opens on-face.
        _, gap = self._dur_gap(state)
        self._next_avert_t = now + gap

    def tick(self, now: float) -> GazeBias:
        """The current social gaze intent. Cheap; call at the control rate."""
        st = self._state
        if st == "thinking":
            # Look up-and-away and hold — a visible planning pause. After
            # think_hold_s, drift back toward the partner (answer is coming).
            if now - self._state_since < self._think_hold:
                return GazeBias(False, (0.0, self._up), "think")
            return GazeBias(True, (0.0, 0.0), "think-return")
        if st not in self._rhythm:      # idle / unknown → no social bias
            return GazeBias(True, (0.0, 0.0), "idle")

        # Are we mid-aversion?
        if now < self._avert_until:
            return GazeBias(False, self._avert_offset(), self._reason("avert"))

        # An aversion just ended: the current on-face stretch begins now (this is
        # what the mutual-gaze ceiling measures — continuous eye contact, not time
        # since the turn began).
        if self._avert_until:
            self._face_since = self._avert_until
            self._avert_until = 0.0

        # On face. Enforce the mutual-gaze ceiling, then the scheduled rhythm.
        dur, gap = self._dur_gap(st)
        forced = (now - self._face_since) >= self._max_mutual
        if forced or now >= self._next_avert_t:
            self._episode += 1
            self._avert_until = now + self._jittered(dur)
            self._next_avert_t = self._avert_until + gap
            return GazeBias(False, self._avert_offset(), self._reason("avert"))
        return GazeBias(True, (0.0, 0.0), "face")

    def notice_sentence_end(self, now: float) -> None:
        """SPEAKING: a sentence just ended — look back at the partner to yield the
        turn (Kendon). Cancels any active aversion so the look-back lands."""
        if self._state == "speaking":
            self._avert_until = 0.0
            self._face_since = now
            # push the next aversion out so the hand-over gaze is held a beat
            _, gap = self._dur_gap("speaking")
            self._next_avert_t = now + gap

    # ── helpers ──────────────────────────────────────────────────────────────
    def _dur_gap(self, state: str) -> tuple[float, float]:
        """(aversion_duration, gap_until_next) for a state. Gap is derived from
        the target face fraction so the long-run on-face ratio equals it."""
        dur, f = self._rhythm.get(state, (0.0, 0.0))
        f = min(max(float(f), 0.0), 0.999)
        gap = dur * f / (1.0 - f) if f > 0.0 else 1e9
        return dur, gap

    def face_fraction(self, state: str) -> float:
        return float(self._rhythm.get(state, (0.0, 1.0))[1])

    def _avert_offset(self) -> tuple[float, float]:
        # Alternate sides each episode so aversions don't always go one way.
        side = -1.0 if (self._episode % 2 == 0) else 1.0
        return (side * self._sideways, 0.0)

    def _jittered(self, base: float) -> float:
        # Deterministic ±15% jitter keyed on the episode count (no RNG) so
        # repeats aren't identical but tests stay reproducible.
        j = ((self._episode * 37) % 7 - 3) / 3.0 * 0.15   # ~[-0.15, +0.15]
        return max(0.2, base * (1.0 + j))

    def _reason(self, kind: str) -> str:
        return f"{self._state[:6]}-{kind}"

    @property
    def state(self) -> str:
        return self._state
