"""Speech barge-in — detect the user talking over ZERO while it speaks.

There is no acoustic echo cancellation (a BT speaker's variable latency makes
reference-aligned AEC impractical), so this works adaptively instead: while
ZERO plays audio, the mic mostly hears ZERO's own echo. The first ``learn_ms``
of playback establish that echo floor (its RMS envelope); afterwards, a frame
counts as FOREGROUND speech only when the VAD says speech AND its level is
well above the learned floor — a person barging in near the mic is loud,
echo off a speaker across the room is not. ``trigger_ms`` of near-consecutive
foreground frames fires the interrupt.

The frames around the trigger are kept, so the interrupting words ("wait,
stop—") can be prepended to the next capture and transcribed — the user never
has to repeat themselves.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np


class SpeechBargeIn:
    def __init__(self, is_speech: Callable[[np.ndarray], bool],
                 block_ms: int = 30, learn_ms: int = 900,
                 trigger_ms: int = 300, ratio: float = 2.0,
                 min_rms: float = 250.0, keep_ms: int = 1500):
        self._is_speech = is_speech
        self._block_ms = max(1, int(block_ms))
        self._learn_blocks = max(1, learn_ms // self._block_ms)
        self._trigger_blocks = max(1, trigger_ms // self._block_ms)
        self._ratio = float(ratio)
        self._min_rms = float(min_rms)
        self._seen = 0
        self._floor = 0.0          # EMA of the echo RMS envelope
        self._run = 0              # near-consecutive foreground frames
        self._misses = 0           # dropout tolerance inside a run
        self._ring: deque = deque(maxlen=max(1, keep_ms // self._block_ms))

    def update(self, frame: np.ndarray, active: bool = True) -> bool:
        """Feed one mic frame; True the moment sustained foreground speech is
        detected. ``active`` = audio is REALLY coming out of the speaker right
        now — before that there is no echo to calibrate against, and the wake
        word is the only interrupt (otherwise the user's own trailing speech,
        arriving before the reply's first audio, fires a false barge-in).
        Never raises."""
        self._ring.append(frame)
        if not active:
            self._run = 0
            return False
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        self._seen += 1
        if self._seen <= self._learn_blocks:
            # Learning window (first frames of REAL playback): all echo;
            # track the loudest of it.
            self._floor = max(self._floor, rms) if self._floor else rms
            return False
        gate = max(self._min_rms, self._ratio * self._floor)
        foreground = rms >= gate and self._is_speech(frame)
        if foreground:
            self._run += 1
            self._misses = 0
            if self._run >= self._trigger_blocks:
                return True
        else:
            # Slow floor adaptation — but only while playback sound is
            # actually audible. Adapting on silence (inter-sentence gaps)
            # decayed the floor to nothing, so the NEXT sentence's own echo
            # fired a false barge-in mid-reply.
            if rms > 0.3 * self._floor:
                self._floor = 0.95 * self._floor + 0.05 * rms
            self._misses += 1
            if self._misses > 1:  # allow a single-frame dropout inside a run
                self._run = 0
        return False

    @property
    def frames(self) -> list:
        """The audio around the trigger (the user's interrupting words)."""
        return list(self._ring)
