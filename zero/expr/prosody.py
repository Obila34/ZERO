"""Prosody analyzer — accent times from raw audio, numpy only, no ML.

Prominence in speech is carried by energy and pitch excursion on stressed
syllables (the acoustic correlates the gesture-timing literature measures
against). This module finds those moments cheaply enough for a Pi 4:

  * RMS energy per 10 ms hop (20 ms frames)
  * voiced-frame f0 by autocorrelation on 40 ms windows
  * prominence score = z(energy) + z(f0), voiced frames only
  * accents = local maxima of the smoothed score, min 250 ms apart,
    above an adaptive threshold

find_accents() is a pure function (tested against synthetic stress bursts);
RollingProsody wraps it for streaming TTS, where a sentence's audio arrives
in pieces and later pieces may not exist when the first plays: feed() chunks
as they synthesize, poll() re-analyzes the grown buffer and yields only the
NEW accents past what it already reported, always a safe margin behind the
buffer's end (a peak at the very edge may still be rising).
"""
from __future__ import annotations

import numpy as np

FRAME_S = 0.020
HOP_S = 0.010
F0_WIN_S = 0.040
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
MIN_GAP_S = 0.25        # two accents can't be closer than a syllable pair
EDGE_GUARD_S = 0.12     # never report a peak this close to the buffer edge


def _frame(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        return np.empty((0, frame), dtype=np.float32)
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _f0_autocorr(frames: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame f0 in Hz (0 = unvoiced). Vectorised autocorrelation over
    the plausible-pitch lag band only."""
    if len(frames) == 0:
        return np.empty(0, dtype=np.float32)
    lo = max(2, int(sr / F0_MAX_HZ))
    hi = min(frames.shape[1] - 1, int(sr / F0_MIN_HZ))
    if hi <= lo:
        return np.zeros(len(frames), dtype=np.float32)
    fr = frames - frames.mean(axis=1, keepdims=True)
    denom = (fr * fr).sum(axis=1) + 1e-9
    best_lag = np.zeros(len(fr), dtype=np.int64)
    best_val = np.zeros(len(fr), dtype=np.float32)
    for lag in range(lo, hi):
        v = (fr[:, :-lag] * fr[:, lag:]).sum(axis=1) / denom
        better = v > best_val
        best_val[better] = v[better]
        best_lag[better] = lag
    f0 = np.where((best_val > 0.30) & (best_lag > 0),
                  sr / np.maximum(best_lag, 1), 0.0)
    return f0.astype(np.float32)


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def find_accents(audio: np.ndarray, sr: int) -> list[float]:
    """Accent times (seconds from audio start) for a mono float array."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame, hop = int(FRAME_S * sr), int(HOP_S * sr)
    frames = _frame(x, frame, hop)
    if len(frames) < 8:
        return []
    energy = np.sqrt((frames * frames).mean(axis=1))
    f0frames = _frame(x, int(F0_WIN_S * sr), hop)
    f0 = _f0_autocorr(f0frames, sr)
    n = min(len(energy), len(f0))
    energy, f0 = energy[:n], f0[:n]
    voiced = (f0 > 0) & (energy > max(1e-4, 0.15 * energy.max()))
    if voiced.sum() < 4:
        return []
    # Absolute-variation floor BEFORE z-scoring: on near-constant input the
    # z-score divides by a vanishing std and inflates numerical ripple into
    # fake prominence (a pure tone scored four "accents"). No real variation,
    # no accents — monotone speech genuinely carries none.
    ev, fv = energy[voiced], f0[voiced]
    if ev.std() < 0.08 * max(float(ev.mean()), 1e-6) and fv.std() < 8.0:
        return []
    # z-scores computed over VOICED frames only, so silence doesn't drag the
    # baseline down and promote every voiced frame to an accent.
    score = np.full(n, -np.inf, dtype=np.float32)
    score[voiced] = _z(energy[voiced]) + _z(f0[voiced])
    # light smoothing so jitter doesn't fragment one peak into three
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    finite = np.where(np.isfinite(score), score, 0.0)
    smooth = np.convolve(finite, k, mode="same")
    smooth = np.where(np.isfinite(score), smooth, -np.inf)
    # Prominence IS contrast: if the voiced material is all at one level
    # (a flat stretch, or a streaming buffer that so far holds only weak
    # syllables), there is no accent in it yet — relative scoring would
    # otherwise crown the least-weak syllable. Spread below ~1 z-unit
    # between peak and median means no genuine stress present.
    fin = smooth[np.isfinite(smooth)]
    if len(fin) == 0 or (fin.max() - np.median(fin)) < 1.0:
        return []
    gap = max(1, int(MIN_GAP_S / HOP_S))
    thresh = 0.5   # in z-units: clearly above the utterance's own average
    accents: list[float] = []
    order = np.argsort(smooth)[::-1]
    taken = np.zeros(n, dtype=bool)
    for i in order:
        if not np.isfinite(smooth[i]) or smooth[i] < thresh:
            break
        if taken[max(0, i - gap):i + gap].any():
            continue
        taken[i] = True
        accents.append(float(i * HOP_S + FRAME_S / 2))
    accents.sort()
    return accents


class RollingProsody:
    """Streaming wrapper for one sentence's audio arriving in pieces."""

    def __init__(self, sr: int, max_s: float = 30.0):
        self._sr = int(sr)
        self._buf = np.empty(0, dtype=np.float32)
        self._max = int(max_s * sr)
        self._reported: list[float] = []

    @property
    def duration_s(self) -> float:
        return len(self._buf) / self._sr

    def feed(self, piece) -> None:
        p = np.asarray(piece, dtype=np.float32).reshape(-1)
        if len(self._buf) < self._max:
            self._buf = np.concatenate([self._buf, p])[: self._max]

    def poll(self) -> list[float]:
        """New accent times since the last poll (seconds from sentence
        start), holding back anything within the edge guard."""
        horizon = self.duration_s - EDGE_GUARD_S
        fresh = []
        for t in find_accents(self._buf, self._sr):
            if t >= horizon:
                continue
            if any(abs(t - r) < MIN_GAP_S / 2 for r in self._reported):
                continue
            fresh.append(t)
        self._reported.extend(fresh)
        return fresh
