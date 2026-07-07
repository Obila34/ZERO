"""The duplex/emotion additions: speech barge-in, mood tracking, adaptive
endpointing, scene-change remarks, and the widened Orpheus cue map."""
import numpy as np

from zero.audio.bargein import SpeechBargeIn
from zero.perception.affect import AffectResult, MoodTracker
from zero.tts.remote_engine import _to_orpheus
from zero.vad.endpointer import _BaseEndpointer

BLOCK_MS = 30
SR = 16000
BLOCK = SR * BLOCK_MS // 1000


def _frame(loudness):
    return np.full(BLOCK, loudness, dtype=np.int16)


class TestSpeechBargeIn:
    def _detector(self, **kw):
        defaults = dict(is_speech=lambda f: True, block_ms=BLOCK_MS,
                        learn_ms=90, trigger_ms=90, ratio=2.0, min_rms=250)
        defaults.update(kw)
        return SpeechBargeIn(**defaults)

    def test_never_triggers_during_learn_window(self):
        d = self._detector(learn_ms=300)
        assert not any(d.update(_frame(5000)) for _ in range(10))

    def test_triggers_on_sustained_loud_speech(self):
        d = self._detector()
        for _ in range(3):
            d.update(_frame(400))       # learn: quiet echo floor
        fired = [d.update(_frame(4000)) for _ in range(3)]
        assert fired[-1]

    def test_echo_level_speech_does_not_trigger(self):
        d = self._detector()
        for _ in range(3):
            d.update(_frame(3000))      # learn: LOUD echo floor
        assert not any(d.update(_frame(3000)) for _ in range(20))

    def test_non_speech_noise_does_not_trigger(self):
        d = self._detector(is_speech=lambda f: False)
        for _ in range(3):
            d.update(_frame(400))
        assert not any(d.update(_frame(4000)) for _ in range(20))

    def test_keeps_the_interrupting_frames(self):
        d = self._detector()
        for _ in range(3):
            d.update(_frame(400))
        while not d.update(_frame(4000)):
            pass
        assert len(d.frames) >= 3


class TestMoodTracker:
    def test_neutral_read_produces_no_note(self):
        label, note = MoodTracker().update(AffectResult("neutral", 0.5, 0.0, 0.0))
        assert label == "neutral" and note is None

    def test_confident_read_notes_immediately(self):
        label, note = MoodTracker().update(
            AffectResult("frustrated", 0.8, -0.5, 0.6))
        assert label == "frustrated" and "frustrated" in note

    def test_persistent_low_confidence_mood_surfaces(self):
        t = MoodTracker()
        note = None
        for _ in range(3):
            _, note = t.update(AffectResult("down", 0.2, -0.4, 0.3))
        assert note is not None and "down" in note

    def test_reset_clears_the_streak(self):
        t = MoodTracker()
        for _ in range(3):
            t.update(AffectResult("down", 0.2, -0.4, 0.3))
        t.reset()
        _, note = t.update(AffectResult("neutral", 0.5, 0.0, 0.0))
        assert note is None


class ScriptedEndpointer(_BaseEndpointer):
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


class TestAdaptiveEndpoint:
    def test_short_utterance_waits_double_silence(self):
        # 2 speech blocks (60ms) < fast-end floor (120ms) -> needs 6 silence
        # blocks, not 3; speech resuming at block 5 must still be captured.
        flags = [True, True] + [False] * 4 + [True, True] + [False] * 6
        ep = ScriptedEndpointer(flags, min_speech_for_fast_end_ms=120)
        utt = ep.capture(iter(_frame(2000) for _ in range(len(flags) + 2)))
        assert utt is not None and utt.size >= 9 * BLOCK

    def test_long_utterance_still_ends_fast(self):
        flags = [True] * 5 + [False] * 3 + [True] * 20   # trailing True ignored
        ep = ScriptedEndpointer(flags, min_speech_for_fast_end_ms=120)
        utt = ep.capture(iter(_frame(2000) for _ in range(len(flags))))
        assert utt is not None and utt.size == 8 * BLOCK

    def test_is_speech_frame_never_raises(self):
        ep = ScriptedEndpointer([])
        ep._is_speech = lambda f: 1 / 0
        assert ep.is_speech_frame(_frame(100)) is False


class TestSceneChanges:
    def _eyes(self):
        from zero.vision.eyes import Eyes

        eyes = Eyes(camera=None, detector=None, color_namer=None,
                    change_note_cooldown_s=0.0)
        eyes._started_at = -1e9          # startup settle window elapsed
        eyes._SETTLE_S = 0.0
        eyes._STABLE_S = 0.0             # instant stability for the test
        return eyes

    class _Det:
        def __init__(self, label):
            self.label = label

    def test_appearance_and_disappearance_are_phrased(self):
        eyes = self._eyes()
        eyes._track_changes([self._Det("guitar")])
        assert eyes.scene_changes() == ["a guitar just came into view"]
        eyes._track_changes([])
        assert eyes.scene_changes() == ["the guitar is no longer in view"]

    def test_cooldown_swallows_rapid_changes(self):
        eyes = self._eyes()
        eyes._change_cooldown_s = 3600.0
        eyes._track_changes([self._Det("cup")])
        assert eyes.scene_changes()      # first note goes out
        eyes._track_changes([self._Det("laptop")])
        assert eyes.scene_changes() == []   # inside cooldown: swallowed


class TestOrpheusCueMap:
    def test_hmm_and_pause_are_performed_not_deleted(self):
        assert _to_orpheus("[hmm] let me think [pause] okay") == \
            "Hmm, let me think ... okay"

    def test_gasp_maps_to_native_tag(self):
        assert _to_orpheus("[gasps] no way") == "<gasp> no way"
