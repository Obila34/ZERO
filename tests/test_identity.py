"""Identity: registry storage/matching, fusion logic, conversational enrolment."""
from __future__ import annotations

import numpy as np
import pytest

from zero.identity.fuser import IdentityFuser, STRANGER
from zero.identity.registry import PersonRegistry
from zero.identity.service import (
    ENROLL_NEEDS_NAME, IdentityService, parse_enroll_command, parse_enrollment,
)


def _unit(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def registry(tmp_path):
    return PersonRegistry(str(tmp_path / "id.sqlite"), max_samples_per_kind=3)


# ── registry ──────────────────────────────────────────────────────────────────
def test_enroll_is_idempotent_case_insensitive(registry):
    a = registry.enroll("David")
    b = registry.enroll("david")
    assert a == b
    assert registry.count() == 1


def test_match_returns_best_person(registry):
    d, j = registry.enroll("David"), registry.enroll("Jane")
    dv, jv = _unit(1), _unit(2)
    registry.add_embedding(d, "voice", dv)
    registry.add_embedding(j, "voice", jv)
    pid, name, score = registry.match("voice", dv + 0.05 * _unit(9))
    assert (pid, name) == (d, "David")
    assert score > 0.9


def test_match_empty_and_bad_vectors(registry):
    assert registry.match("voice", _unit(1)) is None
    registry.add_embedding(registry.enroll("A"), "face", _unit(3))
    assert registry.match("face", np.zeros(32, dtype=np.float32)) is None
    # Mismatched dims are skipped, not fatal.
    assert registry.match("face", _unit(4, dim=16)) is None


def test_sample_cap_keeps_newest(registry):
    pid = registry.enroll("David")
    for i in range(6):
        registry.add_embedding(pid, "voice", _unit(i))
    assert registry.sample_count(pid, "voice") == 3


def test_forget_removes_person_and_samples(registry):
    pid = registry.enroll("David")
    registry.add_embedding(pid, "voice", _unit(1))
    assert registry.forget("DAVID") is True
    assert registry.count() == 0
    assert registry.match("voice", _unit(1)) is None
    assert registry.forget("Nobody") is False


# ── fuser ─────────────────────────────────────────────────────────────────────
def test_fuse_agreement_boosts():
    f = IdentityFuser(threshold=0.50, face_threshold=0.42, voice_threshold=0.45)
    # Individually weak-ish, jointly enough.
    r = f.fuse((1, "David", 0.50), (1, "David", 0.52))
    assert r.is_known and r.via == "face+voice"
    assert r.person_id == 1


def test_fuse_disagreement_never_manufactures_confidence():
    f = IdentityFuser()
    # Two conflicting weak matches -> stranger, not a fused accept.
    r = f.fuse((1, "David", 0.30), (2, "Jane", 0.30))
    assert not r.is_known


def test_fuse_single_signals():
    f = IdentityFuser(face_threshold=0.42, voice_threshold=0.45)
    assert f.fuse((1, "David", 0.60), None).via == "face"
    assert f.fuse(None, (2, "Jane", 0.60)).via == "voice"
    assert not f.fuse((1, "David", 0.30), None).is_known
    assert not f.fuse(None, None).is_known
    assert f.fuse(None, None) == STRANGER


# ── enrolment phrase parsing ─────────────────────────────────────────────────
@pytest.mark.parametrize("text,name", [
    ("My name is David.", "David"),
    ("my name is jane doe", "Jane Doe"),
    ("Call me Sam.", "Sam"),
    ("I'm David.", "David"),
    ("Hey, I am Alice", "Alice"),
    ("This is Peter.", "Peter"),
])
def test_parse_enrollment_accepts(text, name):
    assert parse_enrollment(text) == name


@pytest.mark.parametrize("text", [
    "I'm fine.",
    "I'm back home now",
    "I'm not sure about that",
    "I'm really tired",
    "i'm david",                 # "i'm" requires STT capitalization of the name
    "This is the best day",
    "My name is",                # nothing after
    "I met a guy yesterday and he said I'm David and left",  # too long
])
def test_parse_enrollment_rejects(text):
    assert parse_enrollment(text) is None


# ── explicit "remember my face" enrolment commands ──────────────────────────
@pytest.mark.parametrize("text,name", [
    ("remember me as David", "David"),
    ("Please remember my face as Jane", "Jane"),
    ("enroll me as Sam", "Sam"),
    ("register me as Peter", "Peter"),
])
def test_parse_enroll_command_named(text, name):
    assert parse_enroll_command(text) == name


@pytest.mark.parametrize("text", [
    "remember my face",
    "scan my face and remember me",
    "remember me",
    "learn my face",
    "scan me",
    "memorize my face",
])
def test_parse_enroll_command_unnamed(text):
    assert parse_enroll_command(text) == ENROLL_NEEDS_NAME  # "" sentinel


@pytest.mark.parametrize("text", [
    "remember that I like tea",   # memory command, not enrolment
    "remember to buy milk",       # reminder, not enrolment
    "what's your name",
    "I'm David",                  # name intro — handled by parse_enrollment
    "this is a french press",     # object teach
])
def test_parse_enroll_command_rejects(text):
    assert parse_enroll_command(text) is None


def test_enroll_command_and_name_intro_dont_collide():
    # A name intro is NOT an explicit command; an explicit command is NOT a
    # name intro. They're routed separately in main.py.
    assert parse_enroll_command("I'm David") is None
    assert parse_enrollment("remember my face") is None


# ── service (with injected fake embedders) ───────────────────────────────────
class _FakeVoice:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, audio):
        return self.vec


class _FakeFace:
    def __init__(self, vec):
        self.vec = vec

    def embed_faces(self, frame, max_faces=3):
        return [(self.vec, (0, 0, 10, 10))] if self.vec is not None else []


def test_service_enroll_then_identify(registry):
    dv, df = _unit(10), _unit(11)
    svc = IdentityService(registry, IdentityFuser(),
                          voice_embedder=_FakeVoice(dv),
                          face_recognizer=_FakeFace(df))
    audio = np.ones(1600, dtype=np.int16)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    result, kinds = svc.enroll("David", audio=audio, frame_rgb=frame)
    assert result.is_known and sorted(kinds) == ["face", "voice"]
    ident = svc.identify(audio=audio, frame_rgb=frame)
    assert ident.is_known and ident.name == "David" and ident.via == "face+voice"


def test_service_voice_only_channel(registry):
    dv = _unit(20)
    svc = IdentityService(registry, IdentityFuser(),
                          voice_embedder=_FakeVoice(dv), face_recognizer=None)
    audio = np.ones(1600, dtype=np.int16)
    _, kinds = svc.enroll("Jane", audio=audio, frame_rgb=None)
    assert kinds == ["voice"]
    assert svc.identify(audio=audio).name == "Jane"


def test_service_stranger_when_nothing_sensed(registry):
    svc = IdentityService(registry, IdentityFuser(),
                          voice_embedder=_FakeVoice(None), face_recognizer=None)
    assert not svc.identify(audio=None).is_known
    result, kinds = svc.enroll("Ghost", audio=None, frame_rgb=None)
    assert not result.is_known and kinds == []


def test_service_requires_a_channel(registry):
    with pytest.raises(ValueError):
        IdentityService(registry, IdentityFuser())


# ── voice-owned session tracking (identify_speaker) ──────────────────────────
def test_identify_speaker_attributes_by_voice(registry):
    dv, df = _unit(30), _unit(31)
    svc = IdentityService(registry, IdentityFuser(),
                          voice_embedder=_FakeVoice(dv),
                          face_recognizer=_FakeFace(df))
    audio = np.ones(1600, dtype=np.int16)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    svc.enroll("David", audio=audio, frame_rgb=frame)
    speaker, face_name = svc.identify_speaker(audio=audio, frame_rgb=frame)
    assert speaker.is_known and speaker.name == "David" and speaker.via == "voice"
    assert face_name == "David"


def test_identify_speaker_voice_unknown_is_anonymous_face_still_seen(registry):
    # David is enrolled on both channels. A DIFFERENT voice speaks while David's
    # face is on camera: the session is anonymous (voice owns it), but the face
    # is still recognised for the "I can see you" note.
    dv, df = _unit(40), _unit(41)
    voice = _FakeVoice(dv)
    svc = IdentityService(registry, IdentityFuser(),
                          voice_embedder=voice, face_recognizer=_FakeFace(df))
    audio = np.ones(1600, dtype=np.int16)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    svc.enroll("David", audio=audio, frame_rgb=frame)
    voice.vec = _unit(999)  # an unrecognised voice takes the turn
    speaker, face_name = svc.identify_speaker(audio=audio, frame_rgb=frame)
    assert not speaker.is_known     # anonymous — the voice didn't match
    assert face_name == "David"     # face still recognised (perception only)


def test_identify_speaker_handover_switches_person(registry):
    dv, jv = _unit(50), _unit(51)
    d = registry.enroll("David")
    registry.add_embedding(d, "voice", dv)
    j = registry.enroll("Jane")
    registry.add_embedding(j, "voice", jv)
    voice = _FakeVoice(dv)
    svc = IdentityService(registry, IdentityFuser(),
                          voice_embedder=voice, face_recognizer=None)
    audio = np.ones(1600, dtype=np.int16)
    first, _ = svc.identify_speaker(audio=audio)
    assert first.name == "David"
    voice.vec = jv                  # a different enrolled voice takes over
    second, _ = svc.identify_speaker(audio=audio)
    assert second.name == "Jane" and second.person_id != first.person_id
