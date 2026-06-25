"""Speaker verification — "only my voice", the way Siri does it.

A small ONNX speaker-embedding model (WeSpeaker ECAPA-TDNN) turns a clip of speech
into a 192-d "voiceprint". We enroll the owner once (average a few clips), then at
runtime compare each utterance's voiceprint to the enrolled one by cosine
similarity. Above the threshold => it's the owner; below => ignore it.

Runs on onnxruntime on both PC and the Pi (same model file, same code), so it can
be prototyped on a laptop and shipped to the Pi unchanged.

Model: https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM
       (voxceleb_ECAPA512_LM.onnx, ~25 MB)
"""
from __future__ import annotations

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("voiceid")


class SpeakerVerifier:
    def __init__(self, model_path: str, threshold: float = 0.5):
        import onnxruntime as ort  # lazy
        import kaldi_native_fbank as knf  # lazy

        self.threshold = threshold
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name

        # 80-dim Kaldi fbank @16 kHz — matches WeSpeaker's training front-end.
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = False
        opts.mel_opts.num_bins = 80
        self._knf = knf
        self._fbank_opts = opts
        log.info("speaker verifier loaded (%s, threshold=%.2f)", model_path, threshold)

    # -- features -----------------------------------------------------------
    def _fbank(self, audio: np.ndarray) -> np.ndarray:
        """audio: float32 mono [-1, 1] @16 kHz -> [T, 80] mean-normalized fbank."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        fb = self._knf.OnlineFbank(self._fbank_opts)
        # Kaldi convention: samples scaled to int16 range.
        fb.accept_waveform(16000, (audio * 32768.0).tolist())
        fb.input_finished()
        frames = [fb.get_frame(i) for i in range(fb.num_frames_ready)]
        feats = np.asarray(frames, dtype=np.float32)
        if feats.size == 0:
            return feats
        feats -= feats.mean(axis=0, keepdims=True)  # cepstral mean normalization
        return feats

    # -- embeddings ---------------------------------------------------------
    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Return a unit-length voiceprint for a clip, or None if too short."""
        feats = self._fbank(audio)
        if feats.shape[0] < 10:  # < ~0.1s of usable speech
            return None
        out = self._sess.run(None, {self._input: feats[None, :, :]})[0][0]
        emb = np.asarray(out, dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    @staticmethod
    def score(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two unit voiceprints, in [-1, 1]."""
        return float(np.dot(a, b))

    def verify(self, enrolled: np.ndarray, audio: np.ndarray) -> tuple[float, bool]:
        """Score a clip against the enrolled voiceprint -> (score, is_owner)."""
        emb = self.embed(audio)
        if emb is None:
            return 0.0, False
        s = self.score(enrolled, emb)
        return s, s >= self.threshold
