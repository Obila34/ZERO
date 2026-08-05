"""Kyutai TTS prewarm — the socket opened BEFORE the reply text exists.

These test the adoption/fallback logic with fabricated presessions (no real
websocket): the load-bearing properties are that a live unclaimed session is
adopted exactly once, dead/stopped ones are ignored, an adopted session gets
the words and yields the audio, and a prewarm that died falls back to a fresh
inline connection instead of going mute.
"""
import queue
import threading

import numpy as np

from zero.tts.kyutai_engine import KyutaiTTS


def _bare_tts():
    tts = KyutaiTTS.__new__(KyutaiTTS)   # skip __init__: no websocket deps
    tts._pre_lock = threading.Lock()
    tts._pre = None
    tts.timeout = 2.0
    tts.degraded = False
    return tts


def _fake_pre(alive=True):
    release = threading.Event()
    th = threading.Thread(target=release.wait, daemon=True)
    th.start()
    pre = {"thread": th, "taken": threading.Event(), "stop": threading.Event(),
           "text_q": queue.Queue(), "audio_q": queue.Queue(),
           "failed": [False], "_release": release}
    if not alive:
        release.set()
        th.join()
    return pre


class TestPresessionAdoption:
    def test_adopts_a_live_unclaimed_session_exactly_once(self):
        tts = _bare_tts()
        pre = _fake_pre()
        tts._pre = pre
        got = tts._take_presession()
        assert got is pre
        assert pre["taken"].is_set()
        assert tts._pre is None
        assert tts._take_presession() is None   # nothing left on deck
        pre["_release"].set()

    def test_ignores_a_dead_session(self):
        tts = _bare_tts()
        tts._pre = _fake_pre(alive=False)
        assert tts._take_presession() is None

    def test_ignores_a_stopped_session(self):
        tts = _bare_tts()
        pre = _fake_pre()
        pre["stop"].set()
        tts._pre = pre
        assert tts._take_presession() is None
        pre["_release"].set()


class TestPrewarmedStream:
    def test_adopted_session_gets_the_words_and_yields_the_audio(self):
        tts = _bare_tts()
        pre = _fake_pre()
        chunk = np.ones(240, dtype=np.float32)
        pre["audio_q"].put(chunk)
        pre["audio_q"].put(None)
        tts._pre = pre
        out = list(tts.synthesize_stream("hello there friend"))
        assert len(out) == 1 and out[0] is chunk
        words = []
        while True:
            w = pre["text_q"].get_nowait()
            if w is None:
                break
            words.append(w)
        assert words == ["hello", "there", "friend"]
        assert pre["stop"].is_set()   # the session is released afterwards
        pre["_release"].set()

    def test_dead_prewarm_falls_back_to_a_fresh_connection(self, monkeypatch):
        tts = _bare_tts()
        pre = _fake_pre()
        pre["audio_q"].put(None)      # worker died before any audio
        pre["failed"][0] = True
        tts._pre = pre
        marker = np.zeros(3, dtype=np.float32)
        monkeypatch.setattr(KyutaiTTS, "_stream_inline",
                            lambda self, words: iter([marker]))
        out = list(tts.synthesize_stream("hello there"))
        assert len(out) == 1 and out[0] is marker
        pre["_release"].set()

    def test_no_prewarm_uses_the_inline_path(self, monkeypatch):
        tts = _bare_tts()
        seen = {}

        def fake_inline(self, words):
            seen["words"] = words
            yield np.zeros(2, dtype=np.float32)

        monkeypatch.setattr(KyutaiTTS, "_stream_inline", fake_inline)
        out = list(tts.synthesize_stream("good morning"))
        assert len(out) == 1
        assert seen["words"] == ["good", "morning"]
