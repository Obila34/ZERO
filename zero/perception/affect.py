"""Multimodal affect — a coarse read of the speaker's state, fused from:

  * voice prosody  — energy (arousal), pitch via autocorrelation (arousal),
  * text sentiment — a small valence lexicon (no model, offline, instant),
  * face           — optionally a FER+ style ONNX (64x64 gray → 8 emotions)
                     when a model file is configured.

Deliberately coarse: the output steers TONE ("the speaker sounds frustrated"),
it is not a diagnosis. Below the confidence floor it says nothing at all —
a wrong emotion read is worse than none.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("perception.affect")

_POS = {
    "great", "good", "awesome", "love", "loved", "nice", "happy", "excited",
    "amazing", "wonderful", "fantastic", "perfect", "thanks", "thank", "cool",
    "fun", "glad", "yay", "beautiful", "excellent", "brilliant", "haha",
}
_NEG = {
    "bad", "terrible", "awful", "hate", "hated", "angry", "annoyed", "annoying",
    "sad", "tired", "sick", "broken", "wrong", "problem", "problems", "worried",
    "worst", "stupid", "ugh", "frustrated", "frustrating", "upset", "stressed",
    "stop", "no", "can't", "cannot", "fail", "failed", "failing", "hurts",
}


@dataclass(frozen=True)
class AffectResult:
    label: str          # excited | frustrated | down | calm | neutral
    arousal: float      # 0..1
    valence: float      # -1..1
    confidence: float   # 0..1

    @property
    def notable(self) -> bool:
        return self.label != "neutral" and self.confidence >= 0.35


NEUTRAL = AffectResult("neutral", 0.5, 0.0, 0.0)


class MoodTracker:
    """Cross-turn mood: an EMA over per-turn affect reads, so a single noisy
    estimate never whipsaws the tone, and a mood that PERSISTS gets said even
    when each individual read is low-confidence."""

    def __init__(self, alpha: float = 0.4):
        self._alpha = float(alpha)
        self._arousal: float | None = None
        self._valence: float | None = None
        self._streak_label = "neutral"
        self._streak = 0

    def reset(self) -> None:
        self._arousal = self._valence = None
        self._streak_label, self._streak = "neutral", 0

    def update(self, r: AffectResult) -> tuple[str, str | None]:
        """Fold one turn's read in. Returns (mood_label, note_or_None) — the
        note is prompt-ready tone guidance for the LLM."""
        if r.confidence > 0.15:
            if self._arousal is None:
                self._arousal, self._valence = r.arousal, r.valence
            else:
                a = self._alpha
                self._arousal = (1 - a) * self._arousal + a * r.arousal
                self._valence = (1 - a) * self._valence + a * r.valence
        label = self._label()
        if label != "neutral" and label == self._streak_label:
            self._streak += 1
        else:
            self._streak_label, self._streak = label, (1 if label != "neutral" else 0)
        note = None
        # "calm" gets NO note: it's the default state of most conversation, and
        # a small model can't resist narrating it ("you sound calm today") —
        # it still reaches the voice's pacing via the returned label.
        if label not in ("neutral", "calm") and (r.notable or self._streak >= 2):
            persists = " — and has for a few turns now" if self._streak >= 2 else ""
            note = (f"(The speaker sounds {label} right now{persists} — "
                    "let that shape your tone, don't comment on it.)")
        return label, note

    def _label(self) -> str:
        if self._arousal is None:
            return "neutral"
        a, v = self._arousal, self._valence or 0.0
        if a >= 0.6 and v >= 0:
            return "excited"
        if a >= 0.6:
            return "frustrated"
        if a <= 0.35 and v < 0:
            return "down"
        if a <= 0.35:
            return "calm"
        return "neutral"


def _pitch_hz(audio_f: np.ndarray, sr: int) -> float:
    """Median pitch of the strongest voiced windows via autocorrelation.
    0.0 when nothing convincingly voiced."""
    win = int(sr * 0.04)
    if audio_f.size < win * 3:
        return 0.0
    hop = win  # non-overlapping is plenty for a median
    lo, hi = int(sr / 350.0), int(sr / 70.0)   # 70–350 Hz human range
    pitches = []
    for start in range(0, audio_f.size - win, hop):
        frame = audio_f[start:start + win]
        if float(np.sqrt(np.mean(frame ** 2))) < 0.02:
            continue  # unvoiced/silence
        frame = frame - frame.mean()
        ac = np.correlate(frame, frame, mode="full")[win - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if seg.size == 0:
            continue
        lag = int(np.argmax(seg)) + lo
        if ac[lag] / ac[0] > 0.30:             # clear periodicity only
            pitches.append(sr / lag)
    return float(np.median(pitches)) if len(pitches) >= 3 else 0.0


def _text_valence(text: str) -> tuple[float, int]:
    words = re.findall(r"[a-z']+", (text or "").lower())
    pos = sum(w in _POS for w in words)
    neg = sum(w in _NEG for w in words)
    hits = pos + neg
    if hits == 0:
        return 0.0, 0
    return (pos - neg) / hits, hits


class FaceExpression:
    """FER+ style ONNX: 64x64 grayscale in, 8 emotion logits out."""

    _LABELS = ("neutral", "happiness", "surprise", "sadness", "anger",
               "disgust", "fear", "contempt")

    def __init__(self, model_path: str):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(model_path,
                                          providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        log.info("face expression model loaded (%s)", model_path)

    def top(self, face_gray_64: np.ndarray) -> tuple[str, float]:
        x = face_gray_64.astype(np.float32).reshape(1, 1, 64, 64)
        logits = self._sess.run(None, {self._input: x})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        i = int(np.argmax(probs))
        return self._LABELS[i], float(probs[i])


class AffectEstimator:
    def __init__(self, face_model_path: str | None = None,
                 face_detector=None):
        self._fer = None
        self._face_det = face_detector   # reuse identity's Haar detect, if any
        if face_model_path and Path(face_model_path).exists():
            try:
                self._fer = FaceExpression(face_model_path)
            except Exception as e:
                log.warning("face expression channel unavailable: %s", e)

    def estimate(self, audio, sr: int, text: str = "",
                 frame_rgb=None) -> AffectResult:
        """Never raises; returns NEUTRAL when there's not enough signal."""
        try:
            return self._estimate(audio, sr, text, frame_rgb)
        except Exception as e:
            log.debug("affect estimate failed: %s", e)
            return NEUTRAL

    def _estimate(self, audio, sr, text, frame_rgb) -> AffectResult:
        evidence = 0.0
        arousal = 0.5
        if audio is not None and getattr(audio, "size", 0) > sr // 4:
            f = np.asarray(audio, dtype=np.float32).reshape(-1) / 32768.0
            energy = float(np.sqrt(np.mean(f ** 2)))
            e_arousal = min(1.0, energy / 0.12)          # ~0.12 RMS = animated
            pitch = _pitch_hz(f, sr)
            if pitch > 0:
                p_arousal = min(1.0, max(0.0, (pitch - 110.0) / 140.0))
                arousal = 0.6 * e_arousal + 0.4 * p_arousal
                evidence += 0.5
            else:
                arousal = e_arousal
                evidence += 0.3

        valence, hits = _text_valence(text)
        if hits:
            evidence += min(0.5, 0.2 * hits)

        # Face channel (optional): maps FER emotion onto valence/arousal.
        if self._fer is not None and frame_rgb is not None and self._face_det is not None:
            try:
                import cv2

                boxes = self._face_det.detect(frame_rgb)
                if boxes:
                    x, y, w, h = boxes[0]
                    gray = cv2.cvtColor(frame_rgb[y:y + h, x:x + w],
                                        cv2.COLOR_RGB2GRAY)
                    gray = cv2.resize(gray, (64, 64))
                    emo, p = self._fer.top(gray)
                    if p > 0.5:
                        evidence += 0.3
                        if emo == "happiness":
                            valence = max(valence, 0.6)
                        elif emo in ("sadness", "fear"):
                            valence = min(valence, -0.4)
                            arousal = min(arousal, 0.4)
                        elif emo in ("anger", "disgust", "contempt"):
                            valence = min(valence, -0.5)
                            arousal = max(arousal, 0.6)
                        elif emo == "surprise":
                            arousal = max(arousal, 0.7)
            except Exception as e:
                log.debug("face expression failed: %s", e)

        confidence = min(1.0, evidence)
        if arousal >= 0.6 and valence >= 0:
            label = "excited"
        elif arousal >= 0.6:
            label = "frustrated"
        elif arousal <= 0.35 and valence < 0:
            label = "down"
        elif arousal <= 0.35:
            label = "calm"
        else:
            label = "neutral"
        # A non-neutral claim needs BOTH channels leaning the same way unless
        # the evidence is strong. "Calm" is the weakest claim of all — pure
        # prosody (one channel) must not produce it, or every quiet flat
        # sentence gets annotated.
        if label in ("excited", "frustrated", "down") and confidence < 0.35:
            label = "neutral"
        if label == "calm" and confidence <= 0.5:
            label = "neutral"
        return AffectResult(label, round(arousal, 3), round(valence, 3),
                            round(confidence, 3))
