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


class TestLearnedAnnotationCache:
    """Learned-name annotation must NEVER run inline on a turn — it cost
    1.6-2.0s per look at the venue (a remote round trip per detection), paid
    even on turns that asked nothing visual. Turns read a background cache."""

    class _Det:
        def __init__(self, label):
            self.label = label

    class _Snap:
        def __init__(self, dets):
            self.detections = dets
            self.frame_rgb = object()   # non-None: annotation would be legal

    class _ExplodingLearned:
        def annotate(self, frame, dets):  # pragma: no cover — must not run
            raise AssertionError("annotate() ran on the turn path")

    def _eyes(self):
        from zero.vision.eyes import Eyes

        return Eyes(camera=None, detector=None, color_namer=None,
                    learned=self._ExplodingLearned())

    def test_turn_path_never_calls_annotate(self):
        import time

        eyes = self._eyes()
        dets = [self._Det("cup")]
        snap = self._Snap(dets)
        # No cache yet: raw detections come back, annotate() is NOT called.
        assert eyes._with_learned(snap) == dets
        # Fresh cache: the cached annotation is served as-is.
        ann = [self._Det("french press")]
        with eyes._ann_lock:
            eyes._ann_dets, eyes._ann_ts = ann, time.monotonic()
        assert eyes._with_learned(snap) == ann

    def test_stale_cache_falls_back_to_raw_labels(self):
        import time

        eyes = self._eyes()
        dets = [self._Det("cup")]
        with eyes._ann_lock:
            eyes._ann_dets = [self._Det("french press")]
            eyes._ann_ts = time.monotonic() - (eyes._ann_ttl_s + 1.0)
        assert eyes._with_learned(self._Snap(dets)) == dets


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
        # Whisper writes a literal trailing "..." when speech trails off.
        assert ends_mid_thought("Alright, so give me a bit of the...")
        assert ends_mid_thought("Okay, so briefly explain the concept of…")
        assert not ends_mid_thought("what time is it")
        assert not ends_mid_thought("that's all I needed, thanks.")
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


class TestInterruptClassifier:
    def _c(self, text):
        from zero.audio.interrupt import classify_interrupt

        return classify_interrupt(text).value

    def test_corrections_cut_immediately(self):
        assert self._c("no, I meant Tuesday") == "correction"
        assert self._c("wait, that's wrong") == "correction"
        assert self._c("okay stop") == "correction"
        assert self._c("actually, make it three") == "correction"
        assert self._c("hold on") == "correction"

    def test_backchannels_are_waved_through(self):
        assert self._c("yeah") == "backchannel"
        assert self._c("mm-hmm") == "backchannel"
        assert self._c("oh wow") == "backchannel"
        assert self._c("right right") == "backchannel"
        assert self._c("makes sense") == "backchannel"

    def test_new_thoughts_are_queued(self):
        assert self._c("what's the weather tomorrow") == "queue"
        assert self._c("can you also set a timer") == "queue"
        # "no" deep inside a sentence is content, not a correction opener.
        assert self._c("there is no milk left by the way") == "queue"

    def test_noise_is_ignored(self):
        assert self._c("") == "noise"
        assert self._c("   ") == "noise"
        assert self._c(None) == "noise"

    def test_never_raises_on_junk(self):
        assert self._c("...!!!") == "noise"
        assert self._c("🤖🤖") == "noise"


class TestBargeInPercentileFloor:
    def test_one_loud_syllable_cannot_wedge_the_gate(self):
        # Old max-floor: one 3000-RMS blip in the learn window locked the gate
        # at 6000 and the user could never interrupt. Percentile floor ignores
        # the outlier: normal speech still triggers.
        d = SpeechBargeIn(is_speech=lambda f: True, block_ms=BLOCK_MS,
                          learn_ms=300, trigger_ms=90, ratio=2.0, min_rms=250)
        frames = [_frame(400)] * 9 + [_frame(3000)]      # one loud blip
        for f in frames:
            d.update(f)
        fired = [d.update(_frame(1500)) for _ in range(5)]
        assert any(fired)


class TestBargeInEnvelopeCorrelation:
    def test_echo_that_tracks_playback_is_suppressed(self):
        import time as _t

        # The "played" envelope reports exactly what the mic hears (perfect
        # echo, zero lag): correlation must veto the trigger even though the
        # level beats the gate.
        #
        # The played timestamps must be spaced like REAL playback (one chunk
        # per block). An earlier version appended them in a tight loop, so
        # every entry landed microseconds apart while the correlator compares
        # against a block-spaced grid — no overlap, no correlation, and the
        # test only passed because the learned floor happened to suppress the
        # trigger instead. It was not testing correlation at all.
        played = []

        def env():
            return list(played)

        d = SpeechBargeIn(is_speech=lambda f: True, block_ms=BLOCK_MS,
                          learn_ms=90, trigger_ms=90, ratio=2.0, min_rms=250,
                          played_env=env, env_corr_max=0.65)
        rng = np.random.default_rng(7)
        step = BLOCK_MS / 1000.0

        def feed(level):
            # Backdate the played entry onto the same grid the mic frame will
            # land on, then advance real time by one block.
            played.append((_t.monotonic(), float(level)))
            d.update(_frame(level))
            _t.sleep(step)

        for _ in range(12):          # echo phase: mic == played, varying level
            feed(int(rng.uniform(300, 3000)))
        fired = []
        for _ in range(8):           # louder, still tracking playback exactly
            level = int(rng.uniform(2000, 6000))
            played.append((_t.monotonic(), float(level)))
            fired.append(d.update(_frame(level)))
            _t.sleep(step)
        assert not any(fired)

    def test_uncorrelated_speech_still_triggers(self):
        import time as _t

        played = []

        def env():
            return list(played)

        d = SpeechBargeIn(is_speech=lambda f: True, block_ms=BLOCK_MS,
                          learn_ms=90, trigger_ms=90, ratio=2.0, min_rms=250,
                          played_env=env, env_corr_max=0.65)
        # Playback is held CONSTANT so the reference has no variance and no
        # correlation can be computed against it — that is what "uncorrelated"
        # means here, and it makes the test deterministic. Randomising both
        # signals (the earlier version) let them correlate by chance on a short
        # window, so the test passed or failed depending on machine timing.
        step = BLOCK_MS / 1000.0
        for _ in range(12):
            played.append((_t.monotonic(), 400.0))
            d.update(_frame(400))
            _t.sleep(step)
        # Playback unchanged; the mic jumps loud on its own (a person).
        fired = []
        for _ in range(6):
            played.append((_t.monotonic(), 400.0))
            fired.append(d.update(_frame(4000)))
            _t.sleep(step)
        assert any(fired)


class TestBargeInFloorAdaptation:
    def test_user_voice_below_the_gate_cannot_ratchet_the_floor(self):
        import time as _t

        # The venue failure: someone talking UNDER the interrupt gate fed the
        # slow floor-adaptation path frame after frame (it had no correlation
        # guard), the floor climbed to their voice (1279), and the gate rose
        # above anything they could say. Now only frames that CORRELATE with
        # what the speaker is playing may adapt the floor.
        played = []

        def env():
            return list(played)

        d = SpeechBargeIn(is_speech=lambda f: True, block_ms=BLOCK_MS,
                          learn_ms=90, trigger_ms=90, ratio=2.0, min_rms=250,
                          played_env=env, env_corr_max=0.65)
        rng = np.random.default_rng(11)
        step = BLOCK_MS / 1000.0
        # Echo phase: mic tracks playback exactly (varying level) — this is
        # what legitimately builds the floor, via learn window + adaptation.
        # Long enough that the floor window is well populated, as it is in a
        # real reply.
        for _ in range(40):
            level = int(rng.uniform(300, 600))
            played.append((_t.monotonic(), float(level)))
            d.update(_frame(level))
            _t.sleep(step)
        floor_after_echo = d._floor()
        assert floor_after_echo > 0          # echo did feed the floor
        # User phase: playback goes constant-quiet, the mic holds a steady 700
        # (a person talking under the gate). Uncorrelated -> must NOT adapt.
        for _ in range(30):
            played.append((_t.monotonic(), 400.0))
            d.update(_frame(700))
            _t.sleep(step)
        assert d._floor() <= max(floor_after_echo, 600.0), \
            "user's voice was learned into the echo floor"
        # And when they raise their voice above the (honest) gate, it fires.
        fired = []
        for _ in range(6):
            played.append((_t.monotonic(), 400.0))
            fired.append(d.update(_frame(4000)))
            _t.sleep(step)
        assert any(fired)


class TestBargeInGateCeiling:
    def test_gate_is_clamped_and_log_uses_the_same_number(self):
        d = SpeechBargeIn(is_speech=lambda f: True, block_ms=BLOCK_MS,
                          learn_ms=90, trigger_ms=90, ratio=1.6, min_rms=250,
                          gate_ceiling=1200)
        # The venue log printed "gate 2047" while 1200 was actually in force —
        # _gate() is now the single source both the trigger and the log read.
        assert d._gate(1279) == 1200
        assert d._gate(100) == 250        # min_rms floor still applies
        assert d._gate(500) == 800        # ratio path, under the ceiling


class TestBargeInRearm:
    def test_rearm_clears_the_run_but_keeps_the_floor(self):
        d = SpeechBargeIn(is_speech=lambda f: True, block_ms=BLOCK_MS,
                          learn_ms=90, trigger_ms=90, ratio=2.0, min_rms=250)
        for _ in range(3):
            d.update(_frame(400))
        while not d.update(_frame(4000)):
            pass
        d.rearm()
        # Echo-level frames after rearm: no instant retrigger.
        assert not d.update(_frame(400))
        # But sustained loud speech triggers again without re-learning.
        fired = [d.update(_frame(4000)) for _ in range(4)]
        assert any(fired)


class TestEarlyCommit:
    def test_confident_prediction_commits_at_the_spec_pause(self):
        flags = [True] * 5 + [False] * 30
        ep = ScriptedEndpointer(flags, silence_ms=600)   # normal wait: 20 blocks
        utt = ep.capture(iter(_frame(2000) for _ in range(len(flags))),
                         early_commit=lambda audio: True)
        # Committed at the speculative-pause mark (6 blocks of silence), not 20.
        assert utt is not None and utt.size == (5 + ep.spec_blocks) * BLOCK

    def test_unsure_prediction_waits_the_full_window(self):
        flags = [True] * 5 + [False] * 30
        ep = ScriptedEndpointer(flags, silence_ms=600)
        utt = ep.capture(iter(_frame(2000) for _ in range(len(flags))),
                         early_commit=lambda audio: False)
        assert utt is not None and utt.size == 25 * BLOCK  # 5 speech + 20 sil

    def test_early_commit_error_never_breaks_capture(self):
        flags = [True] * 5 + [False] * 30
        ep = ScriptedEndpointer(flags, silence_ms=600)
        utt = ep.capture(iter(_frame(2000) for _ in range(len(flags))),
                         early_commit=lambda audio: 1 / 0)
        assert utt is not None and utt.size > 0


class TestSmileShelf:
    def test_brightens_highs_without_touching_lows(self):
        from zero.tts.orchestrator import _SmileShelf

        sr = 22050
        t = np.arange(sr) / sr
        low = np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.3
        high = np.sin(2 * np.pi * 4000 * t).astype(np.float32) * 0.3

        def gain(x):
            y = _SmileShelf(sr).process(x)
            return float(np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(x**2)))

        assert abs(gain(low) - 1.0) < 0.05          # lows untouched
        assert gain(high) > 1.15                    # ~+2.5dB above the shelf

    def test_streaming_chunks_join_without_a_seam(self):
        from zero.tts.orchestrator import _SmileShelf

        sr = 22050
        x = np.sin(2 * np.pi * 3000 * np.arange(sr) / sr).astype(np.float32)
        whole = _SmileShelf(sr).process(x)
        s = _SmileShelf(sr)
        parts = np.concatenate([s.process(c) for c in np.array_split(x, 7)])
        assert np.allclose(whole, parts, atol=1e-6)


class TestAfterthought:
    def _d(self, **kw):
        defaults = dict(is_speech=lambda f: True, block_ms=BLOCK_MS,
                        learn_ms=90, trigger_ms=90, ratio=2.0, min_rms=250,
                        afterthought_ms=120)
        defaults.update(kw)
        return SpeechBargeIn(**defaults)

    def test_gap_remark_becomes_an_afterthought(self):
        d = self._d()
        for _ in range(6):                      # 180ms of gap speech
            d.update(_frame(2000), active=False)
        for _ in range(12):                     # then quiet: the remark ended
            d.update(_frame(10), active=False)
        frames = d.take_afterthought()
        assert frames is not None and len(frames) >= 6
        assert d.take_afterthought() is None    # one-shot

    def test_too_short_gap_speech_is_not_an_afterthought(self):
        d = self._d(afterthought_ms=300)        # needs 10 blocks
        for _ in range(4):                      # only 4 voiced blocks
            d.update(_frame(2000), active=False)
        for _ in range(12):
            d.update(_frame(10), active=False)
        assert d.take_afterthought() is None

    def test_quiet_gap_speech_below_min_rms_ignored(self):
        d = self._d()
        for _ in range(10):                     # loud enough VAD, too quiet RMS
            d.update(_frame(100), active=False)
        for _ in range(12):
            d.update(_frame(10), active=False)
        assert d.take_afterthought() is None


class TestAmendLastUser:
    def _convo(self):
        from zero.conversation import Conversation

        return Conversation("sys")

    def test_appends_to_last_user_turn(self):
        c = self._convo()
        c.add_user("set a timer for five minutes")
        c.add_assistant("Sure.")
        c.add_user("make me tea")
        c.amend_last_user("oh and make it two")
        msgs = c.messages()
        assert msgs[-1]["content"] == "make me tea oh and make it two"
        assert msgs[-3]["content"] == "set a timer for five minutes"  # untouched

    def test_noop_without_user_turn_or_text(self):
        c = self._convo()
        c.amend_last_user("stray")              # empty history: no crash
        c.add_user("hello")
        c.amend_last_user("")                   # empty extra: no change
        assert c.messages()[-1]["content"] == "hello"


class TestOpenAIMessageConvert:
    def test_plain_messages_pass_through(self):
        from zero.llm.openai_engine import _convert

        msgs = [{"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"}]
        assert _convert(msgs) == msgs

    def test_images_become_data_uri_parts(self):
        from zero.llm.openai_engine import _convert

        out = _convert([{"role": "user", "content": "what is this?",
                         "images": ["QUJD"]}])
        parts = out[0]["content"]
        assert parts[0] == {"type": "text", "text": "what is this?"}
        assert parts[1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,QUJD")


class TestSpeculationGate:
    """The gate that decides whether a reply generated BEFORE the person
    finished may be spoken. Wrong answers here mean ZERO confidently answers a
    question that was never asked, so this is deliberately strict."""

    def _spec(self, text):
        import threading

        from zero.speculate import Speculation

        return Speculation(text, iter(()), threading.Event())

    def test_exact_match_is_kept(self):
        assert self._spec("what time is the meeting").matches(
            "what time is the meeting")

    def test_punctuation_and_case_do_not_matter(self):
        assert self._spec("what time is the meeting").matches(
            "What time is the meeting?")

    def test_extra_words_invalidate_the_bet(self):
        # The reversal case: a prefix match must NOT be accepted.
        s = self._spec("book me a flight")
        assert not s.matches("book me a flight actually no cancel that")

    def test_different_sentence_rejected(self):
        assert not self._spec("call mum").matches("call the plumber")

    def test_empty_never_matches(self):
        assert not self._spec("").matches("")

    def test_abandon_sets_the_stop_event(self):
        s = self._spec("hello there")
        s.abandon()
        assert s.stop.is_set()


class TestWorthSpeculating:
    def _w(self, t):
        from zero.speculate import worth_speculating

        return worth_speculating(t)

    def test_complete_request_is_worth_it(self):
        assert self._w("what is the weather tomorrow")

    def test_mid_sentence_is_not(self):
        assert not self._w("i was thinking that we should")
        assert not self._w("can you get me the")
        assert not self._w("book a table and")

    def test_too_short_is_not(self):
        assert not self._w("okay")
        assert not self._w("")


class TestSpokenVolume:
    def _v(self, t):
        from zero.memory.preferences import parse_volume

        return parse_volume(t)

    def test_quieter_requests(self):
        for t in ("talk quietly", "lower your voice", "keep it down",
                  "not so loud"):
            pref, lvl = self._v(t)
            assert lvl < 1.0 and "quiet" in pref

    def test_louder_requests(self):
        for t in ("speak up", "talk louder", "I can't hear you"):
            pref, lvl = self._v(t)
            assert lvl > 1.0 and "louder" in pref

    def test_whisper_is_quietest(self):
        assert self._v("whisper")[1] < self._v("talk quietly")[1]

    def test_reset_to_normal(self):
        assert self._v("normal volume")[1] == 1.0

    def test_a_story_is_not_an_instruction(self):
        # Long sentences that merely MENTION volume must not change it.
        assert self._v("she whispered a long story to me about that yesterday ok") is None
        assert self._v("what time is it") is None
        # Past the (widened) word cap: still a story, still ignored.
        assert self._v("yesterday he told me to whisper because everyone in the "
                       "library was already asleep by then") is None

    def test_the_venue_phrase_that_was_missed(self):
        # Heard live and misrouted to `-> queue`: 11 words defeated the old
        # 10-word cap, and "a bit" sat between the verb and "louder".
        pref, lvl = self._v(
            "you talk a bit louder than you are talking right now")
        assert lvl > 1.0 and "louder" in pref

    def test_hedged_requests_still_parse(self):
        assert self._v("talk a little louder")[1] > 1.0
        assert self._v("speak a bit more quietly")[1] < 1.0
        assert self._v("could you talk much louder please")[1] > 1.0


class TestRoomSense:
    def _room(self, level, n=30, **kw):
        from zero.audio.room import RoomSense

        r = RoomSense(**kw)
        for _ in range(n):
            r.observe(np.full(BLOCK, level, dtype=np.int16))
        return r

    def test_louder_room_means_louder_voice(self):
        quiet = self._room(80).speech_gain()
        hall = self._room(900).speech_gain()
        assert hall > quiet and hall > 1.0 and quiet < 1.0

    def test_voice_boost_is_capped(self):
        # A very loud room must not drive the output into clipping.
        assert self._room(5000).speech_gain() <= 1.6 + 1e-6

    def test_interrupt_gate_rises_with_noise(self):
        # In a hall the crowd itself must not clear the barge-in gate.
        assert self._room(900).gate(250) > self._room(80).gate(250)

    def test_quiet_room_keeps_the_configured_gate(self):
        assert self._room(40).gate(250) == 250

    def test_observe_never_raises_on_junk(self):
        from zero.audio.room import RoomSense

        r = RoomSense()
        r.observe(None)
        r.observe(np.zeros(0, dtype=np.int16))
        assert r.floor >= 0

    def test_room_label_tracks_level(self):
        assert self._room(60).note() == "quiet"
        assert self._room(1200).note() == "loud"


class TestStripStageDirections:
    def _run(self, *chunks):
        from zero.tts.orchestrator import strip_asides

        return "".join(strip_asides(iter(chunks)))

    def test_multiword_star_span_dropped(self):
        assert self._run("Hey there ", "*smiles slightly* ", "how are you") \
            == "Hey there  how are you"

    def test_single_word_emphasis_kept(self):
        assert self._run("that is *really* good") == "that is *really* good"

    def test_star_span_across_chunks_dropped(self):
        assert self._run("Oh ", "*looks ", "at him* ", "okay") == "Oh  okay"

    def test_parens_still_dropped(self):
        assert self._run("Sure. (a note) done") == "Sure.  done"

    def test_unclosed_star_flushed_as_words(self):
        assert self._run("wait *what") == "wait what"
