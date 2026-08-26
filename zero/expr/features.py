"""Audio features for the neural gesture model — ONE implementation.

Training (scripts/neural_train.py) and serving (server/gesture_server.py)
import THIS module, so the features a checkpoint was trained on are the
features it is served with, by construction rather than by discipline.

20 Hz frames, each: [log-energy, voicedness, f0-norm, 16 log-mel bands]
= 19 dims. Numpy only — the sidecar must run without torch for the mock,
and the Pi never imports this at all.
"""
from __future__ import annotations

import numpy as np

FRAME_HZ = 20.0
N_MEL = 16
FEAT_DIM = 3 + N_MEL
_FFT = 512
_FMIN, _FMAX = 60.0, 4000.0


def _mel_filterbank(sr: int, n_fft: int, n_mel: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(_FMIN), hz_to_mel(min(_FMAX, sr / 2)),
                       n_mel + 2)
    freqs = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mel, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mel):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            fb[i, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        if hi > mid:
            fb[i, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return fb


def extract(audio: np.ndarray, sr: int) -> np.ndarray:
    """(n_frames, FEAT_DIM) float32 at FRAME_HZ. Deterministic."""
    from zero.expr.prosody import _f0_autocorr, _frame

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    hop = max(1, int(sr / FRAME_HZ))
    win = min(2 * hop, len(x))
    frames = _frame(x, win, hop)
    if len(frames) == 0:
        return np.empty((0, FEAT_DIM), dtype=np.float32)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    log_e = np.log1p(energy * 100.0)
    f0 = _f0_autocorr(frames, sr)
    n = min(len(log_e), len(f0))
    voiced = (f0[:n] > 0).astype(np.float32)
    f0n = np.clip(f0[:n] / 400.0, 0.0, 1.0)
    # mel on the same frames (zero-padded/truncated to _FFT)
    fb = _mel_filterbank(sr, _FFT, N_MEL)
    fr = frames[:n]
    if fr.shape[1] < _FFT:
        fr = np.pad(fr, ((0, 0), (0, _FFT - fr.shape[1])))
    else:
        fr = fr[:, :_FFT]
    spec = np.abs(np.fft.rfft(fr * np.hanning(_FFT), axis=1)) ** 2
    mel = np.log1p(spec @ fb.T)
    out = np.concatenate([log_e[:n, None], voiced[:, None], f0n[:, None],
                          mel], axis=1).astype(np.float32)
    return out
