"""Wake-word engine tests.

`openwakeword` is not required to run these — a stub module is injected into
sys.modules before the engine is imported, so the tests pass on any dev box.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest


SR = 16000
CHUNK = 1280  # openWakeWord's native 80 ms window
FRAME = SR * 30 // 1000  # 480 samples — what MicCapture actually hands out


class FakeModel:
    """Stand-in for openwakeword.model.Model.

    Returns a scripted score for each predict() call. Records the chunks it
    received so tests can assert on chunk size and call count.
    """

    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = 0
        self.chunks_seen = []  # sample counts of each chunk fed to predict()
        self.reset_count = 0

    def predict(self, chunk):
        self.chunks_seen.append(len(chunk))
        score = self._scores[self.calls] if self.calls < len(self._scores) else 0.0
        self.calls += 1
        if score is None:            # None = simulate "not enough data" empty dict
            return {}
        return {"hey_jarvis": score}

    def reset(self):
        self.reset_count += 1


@pytest.fixture
def engine_factory(monkeypatch):
    """Build an OpenWakeWordEngine wired to a FakeModel with scripted scores."""
    # Inject a fake openwakeword.model module so the lazy import inside the
    # engine's __init__ resolves without the real package installed.
    def make(scores):
        fake_module = types.ModuleType("openwakeword.model")
        fake = FakeModel(scores)
        fake_module.Model = lambda *a, **kw: fake
        monkeypatch.setitem(sys.modules, "openwakeword", types.ModuleType("openwakeword"))
        monkeypatch.setitem(sys.modules, "openwakeword.model", fake_module)
        # Re-import to pick up the stubbed module.
        sys.modules.pop("zero.wake.openwakeword_engine", None)
        from zero.wake.openwakeword_engine import OpenWakeWordEngine

        engine = OpenWakeWordEngine(model="hey_jarvis", threshold=0.5)
        return engine, fake

    return make


def _frame(value=0):
    return np.full(FRAME, value, dtype=np.int16)


def test_no_predict_until_full_chunk_buffered(engine_factory):
    """Small mic frames must accumulate — predict() should not run yet."""
    engine, fake = engine_factory(scores=[])
    # 2 frames = 960 samples, below the 1280-sample chunk.
    assert engine.process(_frame()) is False
    assert engine.process(_frame()) is False
    assert fake.calls == 0


def test_predict_runs_once_full_chunk_available(engine_factory):
    """Once >=1280 samples are buffered, predict() sees a 1280-sample chunk."""
    # Score below threshold — process() returns False but predict() DID run.
    engine, fake = engine_factory(scores=[0.1])
    for _ in range(3):  # 3 * 480 = 1440 samples -> one full chunk + leftover
        engine.process(_frame())
    assert fake.calls == 1
    assert fake.chunks_seen == [CHUNK]


def test_fires_when_score_meets_threshold(engine_factory):
    """A score >= threshold on any completed chunk must return True."""
    engine, fake = engine_factory(scores=[0.9])
    fired = False
    for _ in range(3):
        if engine.process(_frame()):
            fired = True
            break
    assert fired is True


def test_no_fire_on_empty_scores(engine_factory):
    """Regression: predict() returning {} used to silently mean score=0 forever.
    The engine must handle it, and must NOT fire on it."""
    engine, fake = engine_factory(scores=[None, None, None])
    fired = any(engine.process(_frame()) for _ in range(9))  # 9 * 480 -> 3 chunks
    assert fired is False
    assert fake.calls == 3  # every chunk was still handed to predict()


def test_below_threshold_does_not_fire(engine_factory):
    """Scores under threshold must return False even when predict() ran."""
    engine, fake = engine_factory(scores=[0.4, 0.49])
    fired = any(engine.process(_frame()) for _ in range(6))  # 6 * 480 -> 2 chunks
    assert fired is False
    assert fake.calls == 2


def test_reset_clears_buffer_and_model_state(engine_factory):
    """After reset() a partial buffer must be discarded and the model reset."""
    engine, fake = engine_factory(scores=[0.9])
    engine.process(_frame())        # buffer at 480
    engine.process(_frame())        # buffer at 960 — still no predict
    engine.reset()
    assert fake.reset_count == 1
    # A fresh 480-sample frame must not immediately complete a chunk.
    assert engine.process(_frame()) is False
    assert fake.calls == 0


def test_non_int16_frame_is_coerced(engine_factory):
    """MicCapture always gives int16, but downstream code shouldn't crash if a
    float array leaks through — predict() must still see an int16 chunk."""
    engine, fake = engine_factory(scores=[0.1])
    float_frame = np.zeros(FRAME, dtype=np.float32)
    for _ in range(3):
        engine.process(float_frame)
    assert fake.calls == 1
    assert fake.chunks_seen == [CHUNK]
