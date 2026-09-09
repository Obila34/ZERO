"""Sign-along: time-budgeted picking, spell fallback, one-shot firing,
tee-forwarding, skip-when-busy, recent-word suppression, ships-dark gate."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.sign.along import SignAlong, build_sign_along  # noqa: E402

SR = 16000


class FakeCfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


class FakeEngine:
    def __init__(self, knows=("water", "book", "help", "yes", "school",
                              "teacher")):
        self._knows = set(knows)
        self.sequences: list = []
        self.busy = False
        self.estopped = False
        self._dictionary = object()

    def knows_sign(self, g):
        return g in self._knows

    def sign_sequence(self, tokens, spell_letter_s=None):
        self.sequences.append(list(tokens))
        return [t[1] if isinstance(t, tuple) else t for t in tokens]


def _along(engine, inner=None, over=None):
    cfg = FakeCfg({})
    cfg.update(over or {})
    return SignAlong(cfg, inner, engine)


def _speak(al, idx, sentence, seconds):
    """Feed a sentence with `seconds` of audio, then its first playout."""
    piece = np.zeros(SR, dtype=np.float32)
    for _ in range(int(seconds)):
        al.on_audio(idx, sentence, piece, SR)
    al.on_playout(idx, SR)


class RecordingInner:
    def __init__(self):
        self.audio = []
        self.playout = []

    def on_audio(self, idx, sentence, piece, sr):
        self.audio.append(idx)

    def on_playout(self, idx, n):
        self.playout.append(idx)


def test_budget_picks_what_fits_and_fires_once():
    eng = FakeEngine()
    inner = RecordingInner()
    al = _along(eng, inner)
    # ~5 s sentence -> budget 6.25 s -> 2 signs fit (2.2 s each), not 3
    _speak(al, 0, "the teacher put the book near the water for school", 5)
    al.on_playout(0, SR)             # second playout: must NOT re-fire
    time.sleep(0.3)
    al.stop()
    assert eng.sequences == [["teacher", "book"]]
    assert inner.playout == [0, 0], "inner must receive every event"


def test_long_sentence_takes_more_signs():
    eng = FakeEngine()
    al = _along(eng)
    _speak(al, 0, "the teacher put the book near the water for school", 8)
    time.sleep(0.3)
    al.stop()
    assert eng.sequences == [["teacher", "book", "water", "school"]]


def test_unknown_name_gets_spelled_once():
    eng = FakeEngine()
    al = _along(eng)
    _speak(al, 0, "maxwell brought the water to garissa", 6)
    time.sleep(0.3)
    al.stop()
    (seq,) = eng.sequences
    assert "water" in seq
    assert ("spell", "maxwell") in seq
    assert ("spell", "garissa") not in seq, "one spelled word per sentence"


def test_name_only_sentence_spells_even_on_tight_budget():
    eng = FakeEngine(knows=())
    al = _along(eng)
    _speak(al, 0, "kamau", 1)
    time.sleep(0.3)
    al.stop()
    assert eng.sequences == [[("spell", "kamau")]]


def test_recent_word_not_resigned():
    eng = FakeEngine()
    al = _along(eng)
    _speak(al, 0, "here is the water", 4)
    time.sleep(0.3)
    _speak(al, 1, "more water is coming", 4)
    time.sleep(0.3)
    al.stop()
    assert eng.sequences == [["water"]], eng.sequences


def test_busy_engine_skips_sentence_never_queues():
    eng = FakeEngine()
    eng.busy = True
    al = _along(eng)
    _speak(al, 0, "water please", 4)
    time.sleep(0.3)
    eng.busy = False
    time.sleep(0.3)
    al.stop()
    assert eng.sequences == []


def test_gate_off_and_no_dictionary():
    assert build_sign_along(FakeCfg({}), FakeEngine()) is None
    eng = FakeEngine()
    eng._dictionary = None
    assert build_sign_along(FakeCfg({"sign.along.enabled": True}),
                            eng) is None


class FakeGloss:
    def __init__(self, out):
        self.out = out
        self.calls = []

    def gloss(self, sentence):
        self.calls.append(sentence)
        return self.out


def test_gloss_order_wins_and_hallucinations_are_gated():
    eng = FakeEngine(knows=("school", "water", "go"))
    al = _along(eng)
    # LLM glosses topic-comment; 'xylophone' is hallucinated (unknown,
    # long) -> becomes at most the single spell fallback, never a sign
    al._gloss = FakeGloss(["school", "water", "go", "xylophone"])
    _speak(al, 0, "I will go to the school to get water", 8)
    time.sleep(0.3)
    al.stop()
    (seq,) = eng.sequences
    assert seq[:3] == ["school", "water", "go"], seq
    assert "xylophone" not in seq
    assert al._gloss.calls == ["I will go to the school to get water"]


def test_gloss_failure_falls_back_to_plain_picker():
    eng = FakeEngine(knows=("water",))
    al = _along(eng)
    al._gloss = FakeGloss(None)
    _speak(al, 0, "please bring the water", 4)
    time.sleep(0.3)
    al.stop()
    assert eng.sequences == [["water"]]
