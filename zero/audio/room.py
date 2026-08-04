"""Room sense — hear how loud the space is, and behave accordingly.

Every threshold in this pipeline was tuned in one quiet room: how loud you
must be to interrupt, how loud a frame must be to count as speech, how loud
ZERO talks. Carry the same numbers into a busy hall and all three are wrong
at once — it talks too quietly to hear, and it treats the crowd as speech.

So the room gets measured instead of assumed. While nobody is talking and
nothing is playing, mic frames feed a slow percentile estimate of the ambient
floor. That single number then drives three things:

* how loud ZERO speaks — people raise their voice in noise without noticing
  (the Lombard effect); a machine that doesn't sounds meek and gets lost;
* the level a voice must beat to interrupt a reply;
* the level a frame must beat to count as the start of speech.

Deliberately slow. Ambient level is a property of the room, not of the last
half second, and a fast tracker would chase the very speech it exists to
ignore. It only learns from frames the caller says are quiet, and it moves in
small steps.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("audio.room")


class RoomSense:
    def __init__(self, block_ms: int = 30, window_s: float = 12.0,
                 percentile: float = 50.0, quiet_ref: float = 120.0,
                 loud_ref: float = 900.0, max_boost: float = 1.6,
                 min_boost: float = 0.75):
        n = max(8, int(window_s * 1000 / max(1, block_ms)))
        self._win: deque = deque(maxlen=n)
        self._pct = float(percentile)
        # The two anchors: at or below `quiet_ref` the room is a quiet study
        # and ZERO speaks at (or slightly below) nominal; at `loud_ref` it is
        # a busy hall and ZERO is at `max_boost`. Between them it scales.
        self._quiet = float(quiet_ref)
        self._loud = max(float(loud_ref), float(quiet_ref) + 1.0)
        self._max_boost = float(max_boost)
        self._min_boost = float(min_boost)
        self._floor = float(quiet_ref)
        self._logged = 0.0

    # -- measurement --------------------------------------------------------
    def observe(self, frame) -> None:
        """Feed a mic frame that is believed to be ROOM, not speech and not
        playback. Never raises."""
        try:
            rms = float(np.sqrt(np.mean(np.asarray(frame, dtype=np.float32) ** 2)))
            self._win.append(rms)
            if len(self._win) >= 8:
                self._floor = float(np.percentile(
                    np.fromiter(self._win, dtype=np.float32), self._pct))
        except Exception as e:
            log.debug("room observe failed: %s", e)

    @property
    def floor(self) -> float:
        """Current ambient noise floor, in the same int16 RMS units as the
        rest of the audio path."""
        return self._floor

    # -- what the room implies ---------------------------------------------
    def _t(self) -> float:
        """0.0 in a quiet room, 1.0 in a loud one."""
        span = self._loud - self._quiet
        return max(0.0, min(1.0, (self._floor - self._quiet) / span))

    def speech_gain(self) -> float:
        """Output gain for ZERO's own voice. Above 1.0 in a noisy room, below
        in a very quiet one — the Lombard effect, which is what stops it
        sounding either lost in a crowd or startling in a silent room."""
        t = self._t()
        if t <= 0.0:
            return self._min_boost + (1.0 - self._min_boost) * (
                self._floor / max(1.0, self._quiet))
        return 1.0 + (self._max_boost - 1.0) * t

    def gate(self, base: float) -> float:
        """Scale a loudness threshold (barge-in, VAD start) for this room. In
        silence the configured value stands; in noise it rises with the floor
        so the crowd itself can't clear it."""
        return max(base, base * 0.5 + 1.8 * self._floor)

    def note(self) -> str:
        t = self._t()
        return ("quiet" if t < 0.25 else
                "normal" if t < 0.55 else
                "lively" if t < 0.85 else "loud")

    def maybe_log(self, now: float, every_s: float = 30.0) -> None:
        if now - self._logged >= every_s:
            self._logged = now
            # gate() returns a THRESHOLD, not a multiplier — the old line
            # printed it as "x216.50", which reads as a 216x scale factor.
            log.info("room: %s (ambient rms %.0f) -> voice x%.2f, "
                     "interrupt gate %.0f rms", self.note(), self._floor,
                     self.speech_gain(), self.gate(250.0))
