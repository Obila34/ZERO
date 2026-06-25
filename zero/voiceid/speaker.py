"""Speaker verification — "only my voice", the way Siri does it.

A small ONNX speaker-embedding model (WeSpeaker ECAPA-TDNN) turns a clip of speech
into a 192-d "voiceprint". We enroll the owner once (average a few clips), then at
runtime compare each utterance's voiceprint to the enrolled one by cosine
similarity. Above the threshold => it's the owner; below => ignore it.

Features (80-d log-mel "fbank") are computed in pure NumPy to match WeSpeaker's
Kaldi front-end closely — no compiled dependency, so the SAME code runs on a PC
(any Python) and the Pi's ARM CPU. Only needs onnxruntime + numpy.

Note: the model scales features by cepstral-mean-normalization (per-dim mean
subtracted over time), which also cancels any overall gain — so absolute mic
level doesn't matter, only the spectral shape of your voice.

Model: https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM
       (voxceleb_ECAPA512_LM.onnx, ~25 MB)
"""
from __future__ import annotations

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("voiceid")

_SR = 16000
_FRAME_LEN = 400      # 25 ms @ 16 kHz
_FRAME_SHIFT = 160    # 10 ms
_FFT = 512            # next pow2 >= frame_len
_NUM_MEL = 80
_PREEMPH = 0.97


def _povey_window(n: int) -> np.ndarray:
    a = 2.0 * np.pi / (n - 1)
    w = 0.5 - 0.5 * np.cos(a * np.arange(n))
    return np.power(w, 0.85).astype(np.float32)


def _mel_filterbank(num_bins: int, fft: int, sr: int,
                    low: float = 20.0, high: float | None = None) -> np.ndarray:
    if high is None:
        high = sr / 2.0
    num_fft_bins = fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, num_fft_bins)

    def hz2mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (np.exp(m / 1127.0) - 1.0)

    mel_pts = np.linspace(hz2mel(low), hz2mel(high), num_bins + 2)
    hz_pts = mel2hz(mel_pts)
    fb = np.zeros((num_bins, num_fft_bins), dtype=np.float32)
    for i in range(num_bins):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


class SpeakerVerifier:
    def __init__(self, model_path: str, threshold: float = 0.5):
        import onnxruntime as ort  # lazy

        self.threshold = threshold
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name
        self._window = _povey_window(_FRAME_LEN)
        self._melfb = _mel_filterbank(_NUM_MEL, _FFT, _SR)
        log.info("speaker verifier loaded (%s, threshold=%.2f)", model_path, threshold)

    # -- features (Kaldi-style fbank in NumPy) ------------------------------
    def _fbank(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if len(audio) < _FRAME_LEN:
            return np.zeros((0, _NUM_MEL), dtype=np.float32)

        n_frames = 1 + (len(audio) - _FRAME_LEN) // _FRAME_SHIFT
        idx = np.arange(_FRAME_LEN)[None, :] + _FRAME_SHIFT * np.arange(n_frames)[:, None]
        frames = audio[idx]                                   # [T, 400]
        frames = frames - frames.mean(axis=1, keepdims=True)  # remove DC
        # pre-emphasis (Kaldi: first sample uses itself)
        emph = np.empty_like(frames)
        emph[:, 1:] = frames[:, 1:] - _PREEMPH * frames[:, :-1]
        emph[:, 0] = frames[:, 0] - _PREEMPH * frames[:, 0]
        frames = emph * self._window
        power = np.abs(np.fft.rfft(frames, n=_FFT, axis=1)) ** 2   # [T, 257]
        mel = power.astype(np.float32) @ self._melfb.T            # [T, 80]
        mel = np.log(np.maximum(mel, 1e-10)).astype(np.float32)
        mel -= mel.mean(axis=0, keepdims=True)                    # CMN
        return mel

    # -- embeddings ---------------------------------------------------------
    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        feats = self._fbank(audio)
        if feats.shape[0] < 10:  # < ~0.1s of usable speech
            return None
        out = self._sess.run(None, {self._input: feats[None, :, :]})[0][0]
        emb = np.asarray(out, dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    @staticmethod
    def score(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # both unit-length -> cosine similarity

    def verify(self, enrolled: np.ndarray, audio: np.ndarray) -> tuple[float, bool]:
        emb = self.embed(audio)
        if emb is None:
            return 0.0, False
        s = self.score(enrolled, emb)
        return s, s >= self.threshold
