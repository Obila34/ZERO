"""Sign-along: word picking, one-shot firing per sentence, tee-forwarding,
skip-when-busy, and the ships-dark gate."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.sign.along import SignAlong, build_sign_along  # noqa: E402


class FakeCfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


class FakeEngine:
    def __init__(self, knows=("water", "book", "help", "yes")):
        self._knows = set(knows)
        self.sequences: list = []
        self.busy = False
        self.estopped = False
        self._dictionary = object()

    def knows_sign(self, g):
        return g in self._knows

    def sign_sequence(self, words):
        self.sequences.append(list(words))
        return list(words)


class RecordingInner:
    def __init__(self):
        self.audio = []
        self.playout = []

    def on_audio(self, idx, sentence, piece, sr):
        self.audio.append(idx)

    def on_playout(self, idx, n):
        self.playout.append(idx)


def _along(engine, inner=None, over=None):
    cfg = FakeCfg({"sign.along.max_words_per_sentence": 3})
    cfg.update(over or {})
    return SignAlong(cfg, inner, engine)


def test_picks_dictionary_words_and_fires_once():
    eng = FakeEngine()
    inner = RecordingInner()
    al = _along(eng, inner)
    piece = np.zeros(160, dtype=np.float32)
    al.on_audio(0, "I can help you find the water and a book today", piece,
                16000)
    al.on_playout(0, 160)
    al.on_playout(0, 160)          # second playout: must NOT re-fire
    time.sleep(0.3)
    al.stop()
    assert eng.sequences == [["help", "water", "book"]]
    assert inner.audio == [0] and inner.playout == [0, 0], \
        "inner listener must receive every event untouched"


def test_busy_engine_skips_sentence_never_queues():
    eng = FakeEngine()
    eng.busy = True
    al = _along(eng)
    piece = np.zeros(160, dtype=np.float32)
    al.on_audio(0, "water please", piece, 16000)
    al.on_playout(0, 160)
    time.sleep(0.3)
    eng.busy = False               # freeing later must not replay old text
    time.sleep(0.3)
    al.stop()
    assert eng.sequences == []


def test_stopwords_and_unknown_words_ignored():
    eng = FakeEngine(knows=("yes",))
    al = _along(eng)
    piece = np.zeros(160, dtype=np.float32)
    al.on_audio(1, "well yes that is the thing I was saying", piece, 16000)
    al.on_playout(1, 160)
    time.sleep(0.3)
    al.stop()
    assert eng.sequences == [["yes"]]


def test_gate_off_and_no_dictionary():
    assert build_sign_along(FakeCfg({}), FakeEngine()) is None
    eng = FakeEngine()
    eng._dictionary = None
    assert build_sign_along(FakeCfg({"sign.along.enabled": True}),
                            eng) is None
