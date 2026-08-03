"""Audio-first end-of-turn detection — Pipecat Smart Turn v3 (ONNX).

Given the audio of the user's turn SO FAR, this predicts whether they've
actually finished a complete thought or are just pausing mid-sentence. It reads
prosody, intonation and rhythm straight from the waveform, so the decision is
LOCAL and needs no transcript — unlike the older text-only heuristic
(``ends_mid_thought``) that had to wait for a speculative transcription.

Model: pipecat-ai/smart-turn-v3 (``smart-turn-v3.2-cpu.onnx``). Input is
Whisper's 80-bin log-mel of the last <=8 s of audio, shape ``[1, 80, 800]``;
output is a single logit, ``sigmoid`` -> P(turn complete). Runs on onnxruntime
(CPU) — no torch.

The Whisper feature front-end is reimplemented here in pure numpy (validated
against ``transformers.WhisperFeatureExtractor`` — see
``scripts/validate_smart_turn_features.py``) so the Pi stays lean: no
transformers, torch or librosa, only numpy + onnxruntime which are already
present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("vad.smart_turn")

_SR = 16000
_N_FFT = 400
_HOP = 160
_N_MELS = 80
_CHUNK_S = 8
_MAX_SAMPLES = _CHUNK_S * _SR          # 128000
_MAX_FRAMES = _MAX_SAMPLES // _HOP     # 800


def _mel_filterbank() -> np.ndarray:
    """80 Slaney-normalised triangular mel filters over a 400-pt FFT (201 bins).
    Matches transformers.WhisperFeatureExtractor / librosa (htk=False)."""
    f_min, f_sp = 0.0, 200.0 / 3
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = (min_log_hz - f_min) / f_sp

    def hz_to_mel(f):
        f = np.atleast_1d(np.asarray(f, dtype=np.float64))
        mel = (f - f_min) / f_sp
        log_t = f >= min_log_hz
        mel[log_t] = min_log_mel + np.log(f[log_t] / min_log_hz) / logstep
        return mel

    def mel_to_hz(m):
        m = np.atleast_1d(np.asarray(m, dtype=np.float64))
        freq = f_min + f_sp * m
        log_t = m >= min_log_mel
        freq[log_t] = min_log_hz * np.exp(logstep * (m[log_t] - min_log_mel))
        return freq

    fft_freqs = np.linspace(0, _SR / 2, 1 + _N_FFT // 2)
    mel_min, mel_max = float(hz_to_mel(0.0)[0]), float(hz_to_mel(_SR / 2)[0])
    mel_pts = np.linspace(mel_min, mel_max, _N_MELS + 2)
    hz_pts = mel_to_hz(mel_pts)

    filt = np.zeros((_N_MELS, fft_freqs.size), dtype=np.float64)
    for i in range(_N_MELS):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        lower = (fft_freqs - left) / (center - left)
        upper = (right - fft_freqs) / (right - center)
        filt[i] = np.maximum(0.0, np.minimum(lower, upper))
    # Slaney area normalisation
    enorm = 2.0 / (hz_pts[2:_N_MELS + 2] - hz_pts[:_N_MELS])
    filt *= enorm[:, None]
    return filt.astype(np.float32)


_MEL = _mel_filterbank()
# Periodic Hann window (torch.hann_window default) — NOT numpy's symmetric one.
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_N_FFT) / _N_FFT)).astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """float32 mono 16 kHz audio -> [80, 800] log-mel, matching Whisper's
    feature extractor with do_normalize=True (waveform zero-mean/unit-var)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = audio[-_MAX_SAMPLES:]  # judge the END of the turn (last <=8 s)
    if audio.size:  # zero-mean unit-variance on the REAL samples (pre-pad)
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
    if audio.size < _MAX_SAMPLES:
        audio = np.pad(audio, (0, _MAX_SAMPLES - audio.size))

    padded = np.pad(audio, (_N_FFT // 2, _N_FFT // 2), mode="reflect")
    n_frames = 1 + (padded.size - _N_FFT) // _HOP
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=(n_frames, _N_FFT),
        strides=(padded.strides[0] * _HOP, padded.strides[0]))
    spec = np.fft.rfft(frames * _WINDOW, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).T          # [201, n_frames]

    mel = _MEL @ power                                    # [80, n_frames]
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[:, :_MAX_FRAMES].astype(np.float32)


class SmartTurn:
    """Smart Turn v3 end-of-turn detector. ``complete(audio)`` returns
    P(the speaker has finished their turn) in [0, 1]. Never raises — on any
    failure it returns None so the caller falls back to its own logic."""

    def __init__(self, model_path: str, threshold: float = 0.5,
                 min_audio_ms: int = 250):
        import onnxruntime as ort

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"smart-turn model missing: {model_path}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 2  # small model; don't hog the Pi's 4 cores
        self._sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self._in = self._sess.get_inputs()[0].name
        self.threshold = float(threshold)
        self._min_samples = int(min_audio_ms * _SR / 1000)
        # Validate the graph now (deaf-insurance): a bad model raises here so
        # the factory can fall back to the text heuristic instead of failing mid-turn.
        self._sess.run(None, {self._in: np.zeros((1, _N_MELS, _MAX_FRAMES),
                                                  dtype=np.float32)})
        log.info("smart-turn v3 loaded (%s, threshold=%.2f)", model_path,
                 self.threshold)

    def probability(self, audio: np.ndarray) -> float | None:
        try:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size < self._min_samples:
                return None  # too little audio to judge — let the caller decide
            feats = log_mel(audio)[None]                  # [1, 80, 800]
            logit = float(np.asarray(
                self._sess.run(None, {self._in: feats})[0]).reshape(-1)[0])
            return 1.0 / (1.0 + np.exp(-logit))
        except Exception as e:
            log.debug("smart-turn inference failed: %s", e)
            return None

    def complete(self, audio: np.ndarray) -> bool | None:
        """True = turn finished (commit), False = mid-thought (hold), None =
        undecided (not enough audio / error) -> caller keeps its own logic."""
        p = self.probability(audio)
        if p is None:
            return None
        return p >= self.threshold
