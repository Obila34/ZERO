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


class TestStripAsides:
    def _run(self, chunks):
        from zero.tts.orchestrator import strip_asides

        return "".join(strip_asides(iter(chunks)))

    def test_plain_text_passes_through(self):
        assert self._run(["Hey", " there."]) == "Hey there."

    def test_note_spanning_sentences_and_chunks_is_dropped(self):
        chunks = ["Sure thing. (parentheses: No one is visible",
                  " right now. The speaker sounds calm. )", " Anyway."]
        assert self._run(chunks) == "Sure thing.  Anyway."

    def test_unclosed_paren_suppresses_to_end(self):
        assert self._run(["Okay. (I notice the person is", " sitting there."]) \
            == "Okay. "


class TestBargeInActiveGate:
    def test_no_trigger_before_audio_plays(self):
        d = SpeechBargeIn(is_speech=lambda f: True, block_ms=BLOCK_MS,
                          learn_ms=90, trigger_ms=90, ratio=2.0, min_rms=250)
        assert not any(d.update(_frame(5000), active=False) for _ in range(30))

    def test_silence_does_not_decay_the_echo_floor(self):
        d = SpeechBargeIn(is_speech=lambda f: True, block_ms=BLOCK_MS,
                          learn_ms=90, trigger_ms=90, ratio=2.0, min_rms=250)
        for _ in range(3):
            d.update(_frame(3000))          # learn: loud reply echo
        for _ in range(50):
            d.update(_frame(10))            # inter-sentence silence
        # next sentence's own echo (same level as the floor) must NOT trigger
        assert not any(d.update(_frame(3000)) for _ in range(20))


class TestCalmIsNeverNoted:
    def test_calm_shapes_voice_but_produces_no_note(self):
        t = MoodTracker()
        label = note = None
        for _ in range(4):
            label, note = t.update(AffectResult("calm", 0.2, 0.1, 0.6))
        assert label == "calm" and note is None


class TestOrpheusCueMap:
    def test_hmm_and_pause_are_performed_not_deleted(self):
        assert _to_orpheus("[hmm] let me think [pause] okay") == \
            "Hmm, let me think ... okay"

    def test_gasp_maps_to_native_tag(self):
        assert _to_orpheus("[gasps] no way") == "<gasp> no way"


class TestSpeculativeSTT:
    def test_pause_hook_fires_with_audio_so_far(self):
        calls = []
        flags = [True] * 4 + [False] * 30
        ep = ScriptedEndpointer(flags, silence_ms=600)  # spec fires at 180ms
        ep.capture(iter(_frame(2000) for _ in range(len(flags))),
                   on_speech_pause=calls.append)
        assert len(calls) == 1 and calls[0].size >= 4 * BLOCK

    def test_hook_refires_on_each_pause(self):
        flags = [True] * 4 + [False] * 8 + [True] * 4 + [False] * 30
        calls = []
        ep = ScriptedEndpointer(flags, silence_ms=600)
        ep.capture(iter(_frame(2000) for _ in range(len(flags))),
                   on_speech_pause=calls.append)
        assert len(calls) == 2 and calls[1].size > calls[0].size

    def test_hook_error_never_breaks_capture(self):
        flags = [True] * 4 + [False] * 30
        ep = ScriptedEndpointer(flags, silence_ms=600)
        utt = ep.capture(iter(_frame(2000) for _ in range(len(flags))),
                         on_speech_pause=lambda a: 1 / 0)
        assert utt is not None and utt.size > 0


class TestSalienceHint:
    class _Det:
        def __init__(self, label, area):
            self.label = label
            self.bbox = (0, 0, area, 1)

    def test_biggest_box_outranks_raw_counts(self):
        from zero.vision.phrasing import detector_hint

        dets = [self._Det("chair", 10)] * 5 + [self._Det("guitar", 500)]
        assert detector_hint(dets, max_items=2) == "a guitar, 5 chairs"


class TestPersonCrop:
    def test_no_people_returns_none(self):
        from types import SimpleNamespace

        from zero.vision.eyes import Eyes

        eyes = Eyes(camera=None, detector=None, color_namer=None)
        snap = SimpleNamespace(frame_rgb=np.zeros((100, 100, 3), dtype=np.uint8),
                               detections=[])
        assert eyes._person_crop(snap) is None


class TestSemanticHold:
    def test_mid_thought_words_and_punctuation(self):
        from zero.main import ends_mid_thought

        assert ends_mid_thought("I went to the store and")
        assert ends_mid_thought("so what I mean is,")
        assert ends_mid_thought("could you get me the")
        assert not ends_mid_thought("what time is it")
        assert not ends_mid_thought("")

    def test_hold_extends_endpoint_once(self):
        # 5 speech + long silence; hold=True once -> ends at 2x silence window.
        flags = [True] * 5 + [False] * 30
        calls = []

        def hold():
            calls.append(1)
            return True

        ep = ScriptedEndpointer(flags, silence_ms=90,
                                min_speech_for_fast_end_ms=0)
        utt = ep.capture(iter(_frame(2000) for _ in range(len(flags))),
                         should_hold=hold)
        assert len(calls) == 1                       # asked once, not re-held
        assert utt is not None and utt.size == 11 * BLOCK  # 5 speech + 6 sil

    def test_no_hold_ends_at_normal_window(self):
        flags = [True] * 5 + [False] * 30
        ep = ScriptedEndpointer(flags, silence_ms=90,
                                min_speech_for_fast_end_ms=0)
        utt = ep.capture(iter(_frame(2000) for _ in range(len(flags))),
                         should_hold=lambda: False)
        assert utt is not None and utt.size == 8 * BLOCK   # 5 speech + 3 sil


class TestIouTracker:
    class _Det:
        def __init__(self, label, bbox):
            self.label = label
            self.bbox = bbox

    def _cup(self, x=10):
        return self._Det("cup", (x, 10, 20, 20))

    def test_confirm_then_gone(self):
        from zero.vision.tracker import IouTracker

        tr = IouTracker(confirm_s=1.0, lost_s=1.0)
        assert tr.update([self._cup()], now=0.0) == []      # tentative
        assert ("new", "cup") in tr.update([self._cup()], now=1.5)
        assert ("gone", "cup") in tr.update([], now=3.5)

    def test_same_object_moving_fires_moved_not_new(self):
        from zero.vision.tracker import IouTracker

        tr = IouTracker(confirm_s=0.5, lost_s=2.0, move_frac=0.75,
                        move_cooldown_s=0.0)
        tr.update([self._cup(10)], now=0.0)
        tr.update([self._cup(10)], now=1.0)                 # confirmed
        evs = tr.update([self._cup(35)], now=1.2)           # big jump, IoU ok?
        # 25px shift on a 20x20 box > 0.75*diag(28)=21 -> moved
        assert ("moved", "cup") in evs or ("new", "cup") not in evs

    def test_move_cooldown_suppresses_repeats(self):
        from zero.vision.tracker import IouTracker

        tr = IouTracker(confirm_s=0.0, lost_s=5.0, move_frac=0.5,
                        move_cooldown_s=100.0)
        tr.update([self._cup(10)], now=0.0)
        tr.update([self._cup(10)], now=0.1)                 # confirm
        tr.update([self._cup(25)], now=0.2)                 # moved #1
        evs = tr.update([self._cup(40)], now=0.3)           # inside cooldown
        assert ("moved", "cup") not in evs
