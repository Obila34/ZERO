"""Speech barge-in — detect the user talking over ZERO while it speaks.

There is no sample-accurate AEC on a Bluetooth speaker (its latency drifts
beyond what speex can adapt to), so discrimination is statistical instead.
RMS is the baseline signal; three independent guards decide whether a loud
frame is the USER or ZERO's own echo:

1. Percentile echo floor. While ZERO plays, the mic mostly hears the reply's
   echo. Non-foreground frames feed a rolling window whose PERCENTILE is the
   floor — unlike the old max-of-learn-window, a brief loud syllable early in
   the reply can't wedge the gate above the user's voice, and the floor keeps
   tracking as the reply gets louder or quieter.
2. VAD. The frame must actually be speech-shaped.
3. Envelope correlation (the BT-proof echo test). The speaker reports the RMS
   envelope of what it's ACTUALLY playing; a mic envelope that rises and falls
   with the played envelope (at any lag up to ~400 ms) is our own voice coming
   back, however loud. A human interrupting is uncorrelated with the reply.
   Envelopes tolerate the drifting latency that defeats sample-level AEC.

``trigger_ms`` of near-consecutive foreground frames fires the interrupt.

Pre-roll carry: speech that STARTS in the gap before the reply's first audio
(the "thinking gap") is collected too. If the user is still talking when
playback begins, that can't be echo — the trigger window shortens so ZERO
yields almost immediately instead of talking over them for 300 ms.

The frames around the trigger are kept, so the interrupting words ("wait,
stop—") can be transcribed at once and the user never repeats themselves.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Callable

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("audio.bargein")


class SpeechBargeIn:
    def __init__(self, is_speech: Callable[[np.ndarray], bool],
                 block_ms: int = 30, learn_ms: int = 900,
                 trigger_ms: int = 300, ratio: float = 2.0,
                 min_rms: float = 250.0, keep_ms: int = 1500,
                 played_env: Callable[[], list] | None = None,
                 env_corr_max: float = 0.65,
                 floor_percentile: float = 80.0,
                 carry_ms: int = 150):
        self._is_speech = is_speech
        self._block_ms = max(1, int(block_ms))
        self._learn_blocks = max(1, learn_ms // self._block_ms)
        self._trigger_blocks = max(1, trigger_ms // self._block_ms)
        self._ratio = float(ratio)
        self._min_rms = float(min_rms)
        # Envelope tap into the speaker: () -> [(monotonic_t, rms), ...] of
        # recently played chunks. None = correlation guard off (unit tests).
        self._played_env = played_env
        self._env_corr_max = float(env_corr_max)
        self._pct = float(floor_percentile)
        self._carry_blocks = max(1, carry_ms // self._block_ms)
        self._seen = 0
        # Rolling window of echo-candidate RMS (~2 s); floor = its percentile.
        self._floor_win: deque = deque(maxlen=max(8, 2000 // self._block_ms))
        self._run = 0              # near-consecutive foreground frames
        self._misses = 0           # dropout tolerance inside a run
        self._armed_logged = False
        self._ring: deque = deque(maxlen=max(1, keep_ms // self._block_ms))
        # Thinking-gap speech (before any reply audio): collected but NEVER a
        # trigger by itself — the user's own trailing speech lands here and
        # must not cut off a reply that hasn't made a sound yet. It only
        # matters if the speech CONTINUES into playback (then it can't be echo).
        self._gap_run = 0
        self._gap_misses = 0
        self._gap_frames: list = []
        self._gap_cap = max(1, 4000 // self._block_ms)  # keep <=4 s of gap speech
        # Carry is a short window after playback starts (not a permanent state):
        # past it, gap speech has ended and normal echo discipline resumes.
        self._carry_deadline = 0.0
        self._was_active = False
        # Mic RMS envelope (for correlation against the played envelope).
        self._mic_env: deque = deque(maxlen=max(8, 900 // self._block_ms))

    # -- internals -----------------------------------------------------------
    def _floor(self) -> float:
        if not self._floor_win:
            return 0.0
        return float(np.percentile(np.fromiter(self._floor_win, dtype=np.float32),
                                   self._pct))

    def _echo_correlated(self, now: float) -> bool:
        """True when the mic envelope tracks the played envelope (own echo).
        Compared at 0..~400 ms lags on a block_ms grid; envelopes survive the
        drifting BT latency that breaks sample-level AEC. Never raises."""
        if self._played_env is None or len(self._mic_env) < 8:
            return False
        try:
            played = self._played_env()
            if not played or len(played) < 4:
                return False
            # Resample both envelopes onto a common block_ms grid ending now.
            step = self._block_ms / 1000.0
            n = len(self._mic_env)
            grid = now - step * np.arange(n - 1, -1, -1)
            mic = np.fromiter((r for _, r in self._mic_env), dtype=np.float64)
            pt = np.array([t for t, _ in played], dtype=np.float64)
            pr = np.array([r for _, r in played], dtype=np.float64)
            best = 0.0
            for lag_blocks in range(0, int(0.4 / step) + 1):
                ref = np.interp(grid - lag_blocks * step, pt, pr,
                                left=0.0, right=0.0)
                ms, rs = mic.std(), ref.std()
                if ms < 1e-6 or rs < 1e-6:
                    continue
                c = float(np.corrcoef(mic, ref)[0, 1])
                best = max(best, c)
            return best >= self._env_corr_max
        except Exception as e:  # a correlation hiccup must never break the mic
            log.debug("envelope correlation failed: %s", e)
            return False

    # -- public API ----------------------------------------------------------
    def rearm(self) -> None:
        """Clear the trigger run (after a backchannel was waved through) while
        keeping the learned floor — detection continues without re-learning."""
        self._run = 0
        self._misses = 0

    def update(self, frame: np.ndarray, active: bool = True) -> bool:
        """Feed one mic frame; True the moment sustained foreground speech is
        detected. ``active`` = audio is REALLY coming out of the speaker right
        now. Before that there is no echo to discriminate against, so inactive
        frames never trigger — they only pre-collect (see pre-roll carry).
        Never raises."""
        self._ring.append(frame)
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        now = time.monotonic()
        self._mic_env.append((now, rms))

        if not active:
            self._was_active = False
            self._run = 0
            # Pre-roll: track speech in the thinking gap. Collected, not acted on.
            try:
                voiced = rms >= self._min_rms and self._is_speech(frame)
            except Exception:
                voiced = False
            if voiced:
                self._gap_run += 1
                self._gap_misses = 0
                self._gap_frames.append(frame)
                if len(self._gap_frames) > self._gap_cap:
                    self._gap_frames.pop(0)
            else:
                self._gap_misses += 1
                if self._gap_misses > 8:  # ~250 ms of quiet: the gap speech ended
                    self._gap_run = 0
                    self._gap_frames = []
            return False

        if not self._was_active:
            # Rising edge: playback just started. If real speech was running in
            # the gap right up to this moment, open a short carry window. Carry
            # leans on the envelope-correlation guard to reject the reply's own
            # onset echo, so it needs the tap — no tap, no carry.
            if (self._played_env is not None
                    and self._gap_run >= self._carry_blocks):
                self._carry_deadline = now + 0.6
            else:
                self._gap_frames = []  # stale gap audio must not ride along later
            self._gap_run = 0
            self._gap_misses = 0
            self._was_active = True
        carrying = now < self._carry_deadline
        self._seen += 1
        if self._seen <= self._learn_blocks and not carrying:
            # Learning window (first frames of REAL playback): all echo.
            self._floor_win.append(rms)
            if self._seen == self._learn_blocks and not self._armed_logged:
                self._armed_logged = True
                floor = self._floor()
                # The calibration line: your voice must beat `gate` to
                # interrupt. If it never triggers, lower barge_in_ratio /
                # barge_in_min_rms; if it self-triggers, raise them.
                log.info("barge-in armed: echo floor %.0f, speech gate %.0f",
                         floor, max(self._min_rms, self._ratio * floor))
            return False

        floor = self._floor()
        gate = max(self._min_rms, self._ratio * floor)
        if carrying:
            # Speech began BEFORE any audio played — it cannot be our echo.
            # Only the absolute level + VAD gate it, and the trigger shortens:
            # ZERO yields nearly instantly instead of talking over them.
            gate = self._min_rms
        try:
            voiced = self._is_speech(frame)
        except Exception:
            voiced = False
        foreground = rms >= gate and voiced
        if foreground and self._echo_correlated(now):
            foreground = False  # loud, speech-shaped — but it moves WITH the reply
        if foreground:
            self._run += 1
            self._misses = 0
            need = 2 if carrying else self._trigger_blocks
            if self._run >= need:
                if self._gap_frames:
                    # Prepend the gap speech so the transcription gets the
                    # words from their very first syllable.
                    for f in reversed(self._gap_frames):
                        self._ring.appendleft(f)
                    self._gap_frames = []
                self._carry_deadline = 0.0
                return True
        else:
            # Slow floor adaptation — but only while playback sound is actually
            # audible. Adapting on silence (inter-sentence gaps) decayed the
            # floor to nothing, so the NEXT sentence's own echo fired a false
            # barge-in mid-reply.
            if rms > 0.3 * floor:
                self._floor_win.append(rms)
            self._misses += 1
            if self._misses > 1:  # allow a single-frame dropout inside a run
                self._run = 0
        return False

    @property
    def frames(self) -> list:
        """The audio around the trigger (the user's interrupting words)."""
        return list(self._ring)
