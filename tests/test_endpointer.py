import numpy as np

from zero.vad.endpointer import _BaseEndpointer

BLOCK_MS = 30
SR = 16000
BLOCK = SR * BLOCK_MS // 1000  # 480 samples


class ScriptedEndpointer(_BaseEndpointer):
    """VAD decisions driven by a script instead of a real model."""

    def __init__(self, speech_flags, **kw):
        defaults = dict(sample_rate=SR, silence_ms=90, max_utterance_ms=3000,
                        speech_pad_ms=30, block_ms=BLOCK_MS)
        defaults.update(kw)
        super().__init__(**defaults)
        self._flags = list(speech_flags)
        self._i = 0

    def _is_speech(self, frame):
        flag = self._flags[self._i] if self._i < len(self._flags) else False
        self._i += 1
        return flag


def _frames(loudness, n):
    return [np.full(BLOCK, loudness, dtype=np.int16) for _ in range(n)]


def test_silence_returns_none_on_idle_timeout():
    ep = ScriptedEndpointer([False] * 100)
    out = ep.capture(iter(_frames(0, 100)), idle_timeout_s=0.3)  # 10 blocks
    assert out is None


def test_speech_then_silence_returns_utterance():
    # 5 speech blocks then enough silence to end the utterance (3 blocks @ 90ms).
    flags = [True] * 5 + [False] * 10
    ep = ScriptedEndpointer(flags)
    out = ep.capture(iter(_frames(2000, 15)))
    assert out is not None
    assert out.dtype == np.float32
    assert np.abs(out).max() <= 1.0
    # speech + pad + trailing silence, in samples
    assert len(out) >= 5 * BLOCK


def test_semantic_hold_waits_for_pending_transcript():
    # should_hold() returning None means "the transcript is still in flight" —
    # the endpointer must WAIT (bounded) instead of committing a half-sentence.
    flags = [True] * 5 + [False] * 40
    calls = {"n": 0}

    def should_hold():
        calls["n"] += 1
        return None if calls["n"] < 4 else False   # arrives on the 4th check

    ep = ScriptedEndpointer(flags, semantic_hold_wait_ms=300)  # 10 blocks
    out = ep.capture(iter(_frames(2000, 45)), should_hold=should_hold)
    assert out is not None
    assert calls["n"] >= 4               # it actually waited for the answer


def test_semantic_hold_wait_is_bounded():
    # A transcript that NEVER arrives must not hold the endpoint forever.
    flags = [True] * 5 + [False] * 40
    ep = ScriptedEndpointer(flags, semantic_hold_wait_ms=90)   # 3 blocks max
    out = ep.capture(iter(_frames(2000, 45)), should_hold=lambda: None)
    assert out is not None               # committed after the bounded wait
    # utterance length stays close to speech + silence + small wait, not 40 blocks
    assert len(out) <= 15 * BLOCK


def test_semantic_hold_true_extends_once():
    flags = [True] * 5 + [False] * 40
    decisions = iter([True, False])      # mid-thought once, then finished

    ep = ScriptedEndpointer(flags)
    out = ep.capture(iter(_frames(2000, 45)),
                     should_hold=lambda: next(decisions, False))
    assert out is not None


def test_quiet_utterance_is_dropped_not_a_timeout():
    # A whole utterance below min_utterance_rms is background — dropped, and the
    # capture keeps listening rather than reporting idle sleep.
    flags = [True] * 5 + [False] * 10          # quiet utterance...
    flags += [True] * 5 + [False] * 10         # ...then a loud one
    quiet = _frames(100, 15)
    loud = _frames(5000, 15)
    ep = ScriptedEndpointer(flags, min_utterance_rms=1000)
    out = ep.capture(iter(quiet + loud))
    assert out is not None
    rms = float(np.sqrt(np.mean((out * 32768.0) ** 2)))
    assert rms > 1000


def test_max_length_cap():
    n = 200
    ep = ScriptedEndpointer([True] * n, max_utterance_ms=300)  # cap = 10 blocks
    out = ep.capture(iter(_frames(2000, n)))
    assert out is not None
    assert len(out) <= (10 + 2) * BLOCK  # cap plus pre-roll slack


def test_exhausted_frames_return_none():
    ep = ScriptedEndpointer([False] * 5)
    assert ep.capture(iter(_frames(0, 5))) is None
