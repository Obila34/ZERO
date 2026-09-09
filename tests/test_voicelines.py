"""Canned-speech subsystem (M4.1 surgery slice): synthesis resilience,
category fit, selection."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.voicelines import VoiceLines


class FakeCfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


def _clip():
    return np.ones(100, dtype=np.float32)


def test_presynth_and_selection():
    lines = VoiceLines(FakeCfg({"conversation.filler_probability": 1.0}),
                       lambda t: _clip())
    lines.presynth()
    assert lines.pick_filler("what is a robot?") is not None
    assert lines.recovery_clip("lost") is not None
    assert lines.recovery_clip("nonsense-kind") is not None  # retry fallback
    assert len(lines.clips("ack")) == 2


def test_dead_tts_never_blocks_startup():
    calls = []

    def dead(t):
        calls.append(t)
        raise RuntimeError("tts down")
    lines = VoiceLines(FakeCfg({}), dead)
    lines.presynth()          # must not raise
    assert lines.pick_filler("hello there my friend") is None
    assert lines.recovery_clip() is None
    # filler synth gives up after 2 misses instead of hammering a dead TTS
    assert len([c for c in calls if c]) < 20


def test_category_fit():
    assert VoiceLines.filler_category("why is the sky blue") == "question"
    assert VoiceLines.filler_category("ok") == "ack"
    assert VoiceLines.filler_category("bring me the red cup please") == "default"
