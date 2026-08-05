"""The remote->local failover wrappers: primary first, fall back on failure,
recover when the primary comes back."""
import numpy as np

from zero.stt.base import STT
from zero.stt.fallback import FallbackSTT
from zero.tts.base import TTS
from zero.tts.fallback import FallbackTTS

AUDIO = np.zeros(1600, dtype=np.float32)


class FlakySTT(STT):
    def __init__(self):
        self.down = False
        self.calls = 0

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        if self.down:
            raise RuntimeError("tunnel down")
        return "remote text"


class LocalSTT(STT):
    def transcribe(self, audio, sample_rate):
        return "local text"


def test_stt_uses_primary_when_up():
    primary = FlakySTT()
    stt = FallbackSTT(primary, LocalSTT)
    assert stt.transcribe(AUDIO, 16000) == "remote text"


def test_stt_falls_back_and_recovers():
    primary = FlakySTT()
    stt = FallbackSTT(primary, LocalSTT)
    primary.down = True
    assert stt.transcribe(AUDIO, 16000) == "local text"
    primary.down = False  # tunnel restored
    assert stt.transcribe(AUDIO, 16000) == "remote text"


def test_stt_without_fallback_returns_empty():
    primary = FlakySTT()
    primary.down = True
    stt = FallbackSTT(primary, None)
    assert stt.transcribe(AUDIO, 16000) == ""


def test_stt_broken_builder_does_not_retry_every_turn():
    calls = []

    def broken_builder():
        calls.append(1)
        raise RuntimeError("no model file")

    primary = FlakySTT()
    primary.down = True
    stt = FallbackSTT(primary, broken_builder)
    assert stt.transcribe(AUDIO, 16000) == ""
    assert stt.transcribe(AUDIO, 16000) == ""
    assert len(calls) == 1


class FlakyTTS(TTS):
    sample_rate = 24000

    def __init__(self):
        self.down = False

    def synthesize(self, text):
        if self.down:
            return np.zeros(0, dtype=np.float32)  # remote engines fail "empty"
        return np.ones(2400, dtype=np.float32)


class LocalTTS(TTS):
    sample_rate = 22050

    def __init__(self):
        self.spoken = []

    def synthesize(self, text):
        self.spoken.append(text)
        return np.ones(2205, dtype=np.float32)


def test_tts_uses_primary_when_up():
    tts = FallbackTTS(FlakyTTS(), LocalTTS)
    assert tts.synthesize("hi").size == 2400


def test_tts_falls_back_resampled_to_primary_rate():
    primary = FlakyTTS()
    primary.down = True
    tts = FallbackTTS(primary, LocalTTS)
    audio = tts.synthesize("hello there")
    # 2205 samples @ 22050 Hz resampled to the primary's 24000 Hz ≈ 2400 samples.
    assert abs(audio.size - 2400) <= 2
    assert tts.sample_rate == 24000


def test_tts_fallback_strips_cue_tags():
    primary = FlakyTTS()
    primary.down = True
    local = LocalTTS()
    tts = FallbackTTS(primary, lambda: local)
    tts.synthesize("[laughs] that's *so* true")
    assert local.spoken == ["that's so true"]


def test_tts_stream_falls_back_when_primary_yields_nothing():
    primary = FlakyTTS()
    primary.down = True
    tts = FallbackTTS(primary, LocalTTS)
    pieces = list(tts.synthesize_stream("hello"))
    assert len(pieces) == 1
    assert pieces[0].size > 0


def test_tts_empty_text_stays_empty():
    tts = FallbackTTS(FlakyTTS(), LocalTTS)
    assert tts.synthesize("   ").size == 0
    assert list(tts.synthesize_stream("  ")) == []


# 1.0s at int16-scale rms ~1638: audio that PLAINLY contains speech.
SPEECH = np.full(16000, 0.05, dtype=np.float32)


class EmptySTT(STT):
    """Kyutai's live-venue failure mode: the request succeeds, the marker
    comes back, and there are zero words — no exception to catch."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        return ""


def test_stt_empty_on_clear_speech_escalates_to_fallback():
    # The bug that ate live interruptions: '' with no exception was taken as
    # "nothing said", so Whisper never ran and real speech was discarded.
    primary = EmptySTT()
    stt = FallbackSTT(primary, LocalSTT)
    assert stt.transcribe(SPEECH, 16000) == "local text"
    assert stt.degraded is False   # the primary is UP; it just couldn't hear


def test_stt_empty_on_silence_is_taken_at_face_value():
    primary = EmptySTT()
    stt = FallbackSTT(primary, LocalSTT)
    assert stt.transcribe(AUDIO, 16000) == ""   # quiet & short: a real blip
    assert primary.calls == 1


def test_rescue_transcribe_goes_fallback_first():
    # The live streaming session already finalized empty on this very audio —
    # rescue must NOT pay the primary again first.
    primary = FlakySTT()
    stt = FallbackSTT(primary, LocalSTT)
    assert stt.rescue_transcribe(SPEECH, 16000) == "local text"
    assert primary.calls == 0


def test_rescue_transcribe_retries_primary_without_a_fallback():
    stt = FallbackSTT(FlakySTT(), None)
    assert stt.rescue_transcribe(SPEECH, 16000) == "remote text"
