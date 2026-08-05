"""Speaker output leveling — the soft-knee limiter behind output_gain.

The venue needed the robot louder. Raw gain with a hard clip flattened every
loud syllable into a buzz; the soft knee keeps everything below the knee
linear (the loudness actually rises) while peaks round smoothly toward the
ceiling instead of shearing.
"""
import sys
import types

import numpy as np

try:  # playback imports sounddevice at module load; these tests never play
    import sounddevice  # noqa: F401
except Exception:  # pragma: no cover — dev box without PortAudio
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

from zero.audio.playback import Speaker


class TestLevelLimiter:
    def test_below_knee_is_pure_gain(self):
        sp = Speaker(output_gain=2.0)
        block = np.full(64, 0.3, dtype=np.float32)
        assert np.allclose(sp._level(block), 0.6)   # 0.6 < knee: untouched

    def test_peaks_are_limited_not_clipped(self):
        sp = Speaker(output_gain=2.2)
        block = np.linspace(-1.0, 1.0, 401).astype(np.float32)
        out = sp._level(block)
        # Never above the ceiling (float32 tanh saturates AT 1.0 for the very
        # loudest inputs — that's the asymptote, not a hard clip) ...
        assert float(np.max(np.abs(out))) <= 1.0
        # ... moderate over-knee peaks are squashed, not flattened flat ...
        mid = sp._level(np.array([0.5], dtype=np.float32))  # 1.1 after gain
        assert 0.9 < float(mid[0]) < 1.0
        # ... and the curve stays monotone (no wraparound, no knee inversion).
        assert np.all(np.diff(out) >= -1e-7)

    def test_gain_actually_makes_it_louder(self):
        block = (0.2 * np.sin(np.linspace(0, 40, 4000))).astype(np.float32)
        quiet = Speaker(output_gain=1.0)._level(block)
        loud = Speaker(output_gain=2.2)._level(block)
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
        assert rms(loud) > 2.0 * rms(quiet)   # the boost survives the limiter

    def test_unity_gain_is_untouched(self):
        sp = Speaker(output_gain=1.0)
        block = np.full(16, 0.9, dtype=np.float32)
        assert sp._level(block) is block
