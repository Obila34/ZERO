"""Perception: prosody+text affect estimation and speaker-change tracking."""
from __future__ import annotations

import numpy as np

from zero.perception.affect import (
    NEUTRAL, AffectEstimator, _pitch_hz, _text_valence,
)
from zero.perception.diarize import SpeakerTracker

SR = 16000


def _tone(freq: float, seconds: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


# ── building blocks ──────────────────────────────────────────────────────────
def test_pitch_detects_tone():
    audio = _tone(150, 1.0, 0.4).astype(np.float32) / 32768.0
    assert abs(_pitch_hz(audio, SR) - 150) < 10


def test_pitch_silence_is_zero():
    assert _pitch_hz(np.zeros(SR, dtype=np.float32), SR) == 0.0


def test_text_valence():
    v, hits = _text_valence("this is great, I love it")
    assert v > 0 and hits == 2
    v, hits = _text_valence("ugh this is broken and terrible")
    assert v < 0 and hits >= 2
    assert _text_valence("the sky is blue") == (0.0, 0)


# ── the estimator ────────────────────────────────────────────────────────────
def test_loud_positive_is_excited():
    est = AffectEstimator()
    audio = _tone(220, 1.2, 0.5)                       # loud, high pitch
    r = est.estimate(audio, SR, "this is amazing, I love it!")
    assert r.label == "excited" and r.notable


def test_loud_negative_is_frustrated():
    est = AffectEstimator()
    audio = _tone(220, 1.2, 0.5)
    r = est.estimate(audio, SR, "ugh, it's broken again, this is terrible")
    assert r.label == "frustrated" and r.notable


def test_quiet_neutral_stays_unnotable():
    est = AffectEstimator()
    audio = _tone(120, 1.0, 0.05)                      # quiet, low
    r = est.estimate(audio, SR, "the meeting is at three")
    assert not r.notable                               # no evidence, no claim


def test_no_audio_no_text_is_neutral():
    est = AffectEstimator()
    assert est.estimate(None, SR, "") == NEUTRAL


def test_estimator_never_raises():
    est = AffectEstimator()
    r = est.estimate(np.array([1, 2, 3], dtype=np.int16), SR, None)
    assert r.label in ("excited", "frustrated", "down", "calm", "neutral")


# ── speaker tracking ─────────────────────────────────────────────────────────
def _unit(seed, dim=16):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_known_speaker_change_is_named():
    t = SpeakerTracker()
    assert t.update(person_id=1, name="David", voice_emb=_unit(1)) == ""
    note = t.update(person_id=2, name="Jane", voice_emb=_unit(2))
    assert "Jane" in note


def test_same_person_no_note():
    t = SpeakerTracker()
    t.update(person_id=1, name="David", voice_emb=_unit(1))
    assert t.update(person_id=1, name="David", voice_emb=_unit(1)) == ""


def test_unknown_speakers_change_by_voice():
    t = SpeakerTracker(change_threshold=0.40)
    a, b = _unit(1), _unit(9)
    assert float(np.dot(a, b)) < 0.40                  # genuinely different
    t.update(voice_emb=a)
    note = t.update(voice_emb=b)
    assert "different person" in note


def test_unknown_same_voice_no_note():
    t = SpeakerTracker(change_threshold=0.40)
    a = _unit(1)
    t.update(voice_emb=a)
    assert t.update(voice_emb=a) == ""


def test_first_turn_never_flags():
    t = SpeakerTracker()
    assert t.update(person_id=5, name="Ada", voice_emb=_unit(3)) == ""


def test_reset_clears_state():
    t = SpeakerTracker()
    t.update(person_id=1, name="David")
    t.reset()
    assert t.update(person_id=2, name="Jane") == ""    # fresh conversation
