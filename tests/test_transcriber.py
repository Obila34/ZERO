"""Transcriber (M4.2a surgery slice): lock/strip semantics, afterthought
expiry, rescue plumbing, live-session gating."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.turn import Transcriber


class FakeCfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


class Engine:
    def __init__(self, text="  hello  "):
        self.text = text
        self.calls = 0

    def transcribe(self, audio, sr):
        self.calls += 1
        return self.text


def test_transcribe_strips_and_locks():
    t = Transcriber(FakeCfg({}), Engine())
    assert t.transcribe(np.zeros(10), 16000) == "hello"
    assert not t.lock.locked()


def test_rescue_only_when_engine_offers_it():
    t = Transcriber(FakeCfg({}), Engine())
    assert t.rescue(np.zeros(10), 16000) is None
    eng = Engine()
    eng.rescue_transcribe = lambda a, sr: " saved "
    t2 = Transcriber(FakeCfg({}), eng)
    assert t2.rescue(np.zeros(10), 16000) == "saved"


def test_afterthoughts_expire():
    t = Transcriber(FakeCfg({"stt.afterthought_max_age_s": 0.1}), Engine())
    t._afterthoughts = [(time.monotonic() - 5.0, "stale words"),
                        (time.monotonic(), "fresh words")]
    assert t.pop_afterthoughts() == "fresh words"
    assert t.pop_afterthoughts() == ""          # one-shot


def test_note_afterthought_records_timestamped():
    t = Transcriber(FakeCfg({}), Engine(text="oh and one more"))
    t.note_afterthought([np.ones(160, dtype=np.int16)])
    assert t.pop_afterthoughts() == "oh and one more"


def test_open_live_gated_and_safe():
    t = Transcriber(FakeCfg({"stt.streaming": False}), Engine())
    assert t.open_live(16000) is None
    t2 = Transcriber(FakeCfg({}), Engine())     # engine has no live_session
    assert t2.open_live(16000) is None
    t2.close_live()                             # idempotent, never raises
