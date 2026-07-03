"""Privacy: bystander gating, forget commands, indicator, self-state flags."""
from __future__ import annotations

import numpy as np
import pytest

from zero.identity.fuser import STRANGER, IdentityResult
from zero.memory.embeddings import HashEmbedder
from zero.memory.sqlite_memory import SqliteMemory
from zero.privacy.guard import PrivacyGuard, parse_forget_command
from zero.privacy.indicator import ListeningIndicator

KNOWN = IdentityResult(1, "David", 0.9, via="face+voice")


# ── bystander modes ──────────────────────────────────────────────────────────
def test_open_mode_everyone_full_service():
    g = PrivacyGuard("open")
    for ident in (KNOWN, STRANGER, None):
        d = g.decide(ident)
        assert d.respond and d.store_memory


def test_guarded_mode_answers_but_never_remembers_strangers():
    g = PrivacyGuard("guarded")
    d = g.decide(STRANGER)
    assert d.respond and not d.store_memory
    d = g.decide(KNOWN)
    assert d.respond and d.store_memory


def test_strict_mode_ignores_strangers():
    g = PrivacyGuard("strict")
    d = g.decide(STRANGER)
    assert not d.respond and not d.store_memory
    d = g.decide(KNOWN)
    assert d.respond and d.store_memory


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        PrivacyGuard("paranoid")


# ── forget commands ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,kind", [
    ("Forget everything about me", "me"),
    ("forget about me", "me"),
    ("please forget me", "me"),
    ("forget that", "that"),
    ("forget what I just said", "that"),
    ("forget it", "that"),
])
def test_parse_forget_accepts(text, kind):
    assert parse_forget_command(text) == kind


@pytest.mark.parametrize("text", [
    "don't forget the milk",
    "I always forget where I put my keys honestly it happens every day",
    "what is it",
])
def test_parse_forget_rejects(text):
    assert parse_forget_command(text) is None


def test_forget_last_and_person(tmp_path):
    m = SqliteMemory(str(tmp_path / "m.sqlite"), embedder=HashEmbedder(64))
    m.remember("secret", "the cake is a lie", person_id=1)
    what = m.forget_last(1)
    assert what is not None and "cake" in what
    assert m.forget_last(1) is None
    m.remember("a", "1", person_id=1)
    m.remember("b", "2", person_id=1)
    assert m.forget_person(1) == 2


# ── indicator (no GPIO here: log path) ───────────────────────────────────────
def test_indicator_without_gpio_is_safe():
    ind = ListeningIndicator(gpio_pin=None)
    for s in ("listening", "thinking", "speaking", "idle", "idle"):
        ind.set_state(s)
    ind.close()


def test_indicator_bad_pin_degrades():
    ind = ListeningIndicator(gpio_pin=17)   # no gpiozero here — must not raise
    ind.set_state("listening")
    ind.close()


# ── self-state (degraded flags on the fallback wrappers) ─────────────────────
def test_tts_fallback_sets_degraded_flag():
    from zero.tts.fallback import FallbackTTS

    class DeadPrimary:
        sample_rate = 24000

        def synthesize(self, text):
            raise RuntimeError("tunnel down")

        def synthesize_stream(self, text):
            raise RuntimeError("tunnel down")

        def close(self):
            pass

    class LocalVoice:
        sample_rate = 22050

        def synthesize(self, text):
            return np.ones(100, dtype=np.float32)

        def close(self):
            pass

    tts = FallbackTTS(DeadPrimary(), lambda: LocalVoice())
    audio = tts.synthesize("hello")
    assert audio.size > 0
    assert tts.degraded is True


def test_stt_fallback_sets_degraded_flag():
    from zero.stt.fallback import FallbackSTT

    class DeadPrimary:
        def transcribe(self, audio, sr):
            raise RuntimeError("tunnel down")

        def close(self):
            pass

    class LocalEar:
        def transcribe(self, audio, sr):
            return "hello"

        def close(self):
            pass

    stt = FallbackSTT(DeadPrimary(), lambda: LocalEar())
    assert stt.transcribe(np.zeros(16000, dtype=np.int16), 16000) == "hello"
    assert stt.degraded is True


def test_remote_stt_language_param():
    from zero.stt.remote_engine import RemoteSTT

    r = RemoteSTT("http://127.0.0.1:9000/transcribe", language="auto")
    assert r.url.endswith("?language=auto")
    r = RemoteSTT("http://127.0.0.1:9000/transcribe")
    assert "language" not in r.url
