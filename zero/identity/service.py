"""IdentityService — resolve "who is this?" each turn, and enrol people
conversationally ("I'm David" → capture face + voice → remember them).

Degrades gracefully per channel: no face model → voice-only; no voice model →
face-only; neither → the factory returns None and ZERO runs anonymous, exactly
as before. On a confident two-signal match the fresh embeddings are fed back
into the registry (capped), so recognition adapts as people change hairstyles,
positions and microphones.
"""
from __future__ import annotations

import re

import numpy as np

from zero.identity.fuser import IdentityFuser, IdentityResult, STRANGER
from zero.identity.registry import PersonRegistry
from zero.utils.logging import get_logger

log = get_logger("identity")

# ── conversational enrolment ──────────────────────────────────────────────────
# "my name is X" / "call me X" are explicit; "i'm X" / "i am X" / "this is X"
# also fire but are filtered hard: STT capitalizes real names, and everyday
# predicates ("I'm fine", "I'm back", "I'm not sure") are blocklisted.
_ENROLL_RES = (
    re.compile(r"\bmy name is ([A-Za-z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)?)", re.IGNORECASE),
    re.compile(r"\bcall me ([A-Za-z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)?)", re.IGNORECASE),
    # Prefix is case-insensitive (STT writes "I'm"); the NAME must stay
    # capitalized — that's the signal it's a name and not a predicate.
    re.compile(r"(?i:\b(?:i'm|i am|this is)\s)([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)?)"),
)
_NOT_NAMES = {
    "fine", "good", "great", "okay", "ok", "back", "here", "home", "not",
    "so", "just", "really", "sure", "sorry", "done", "tired", "hungry",
    "happy", "sad", "busy", "ready", "listening", "serious", "kidding",
    "afraid", "glad", "lost", "late", "early", "leaving", "going", "trying",
    "thinking", "wondering", "asking", "talking", "sick", "cold", "hot",
    "bored", "curious", "confused", "right", "wrong", "new", "old", "still",
    "the", "a", "an", "your", "his", "her", "my", "sam's",
}
_STOPPY_MAX_WORDS = 8  # enrolment is a short utterance, not buried mid-story


def parse_enrollment(text: str) -> str | None:
    """The name being introduced, or None. Deliberately conservative."""
    t = text.strip()
    if len(t.split()) > _STOPPY_MAX_WORDS:
        return None
    for rx in _ENROLL_RES:
        m = rx.search(t)
        if not m:
            continue
        name = " ".join(m.group(1).split())
        parts = name.split()
        if any(p.lower().strip("'-") in _NOT_NAMES for p in parts):
            continue
        if not all(p[:1].isalpha() and 2 <= len(p) <= 20 for p in parts):
            continue
        return " ".join(p[:1].upper() + p[1:] for p in parts)
    return None


# Explicit "learn my face" style commands — distinct from a name introduction.
# NAMED forms carry the name ("remember me as David"); UNNAMED ones ask ZERO to
# enrol the current speaker without giving a name ("remember my face", "scan
# me"). Careful NOT to collide with the memory command "remember that ..." or a
# reminder "remember to ..." — those have no "me"/"my face"/"scan".
ENROLL_NEEDS_NAME = ""  # sentinel: it's an enrol command, but no name was given

_ENROLL_CMD_NAMED = (
    re.compile(r"\b(?:remember|enroll|enrol|register|learn)\s+me\s+as\s+"
               r"([A-Za-z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)?)", re.IGNORECASE),
    re.compile(r"\bremember\s+my\s+face\s+as\s+"
               r"([A-Za-z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)?)", re.IGNORECASE),
)
_ENROLL_CMD_UNNAMED = re.compile(
    r"\b(?:remember|learn|memori[sz]e|scan)\s+my\s+face\b"
    r"|\bscan\s+me\b"
    r"|\bremember\s+me\b(?!\s+as)"
    r"|\b(?:learn|enroll|enrol|register)\s+me\b(?!\s+as)",
    re.IGNORECASE)


def parse_enroll_command(text: str) -> str | None:
    """Explicit face-enrolment request. Returns the NAME if one was given, the
    ``ENROLL_NEEDS_NAME`` sentinel ('') if it's an enrol request with no name,
    or None if this isn't an enrolment command at all."""
    t = text.strip().rstrip(".!?")
    if len(t.split()) > 12:
        return None
    for rx in _ENROLL_CMD_NAMED:
        m = rx.search(t)
        if m:
            name = " ".join(m.group(1).split())
            parts = name.split()
            if any(p.lower().strip("'-") in _NOT_NAMES for p in parts):
                return ENROLL_NEEDS_NAME
            return " ".join(p[:1].upper() + p[1:] for p in parts)
    if _ENROLL_CMD_UNNAMED.search(t):
        return ENROLL_NEEDS_NAME
    return None


class IdentityService:
    def __init__(self, registry: PersonRegistry, fuser: IdentityFuser,
                 voice_embedder=None, face_recognizer=None,
                 reinforce_threshold: float = 0.70):
        if voice_embedder is None and face_recognizer is None:
            raise ValueError("identity needs at least one of face/voice models")
        self.registry = registry
        self.fuser = fuser
        self._voice = voice_embedder       # exposes .embed(audio) -> unit vec | None
        self._face = face_recognizer       # exposes .embed_faces(frame) -> [(vec, box)]
        self.reinforce_threshold = float(reinforce_threshold)
        log.info("identity active (face=%s, voice=%s, %d people enrolled)",
                 face_recognizer is not None, voice_embedder is not None,
                 registry.count())

    # ── embeddings for the current moment ────────────────────────────────────
    def _voice_emb(self, audio) -> np.ndarray | None:
        if self._voice is None or audio is None or getattr(audio, "size", 0) == 0:
            return None
        try:
            return self._voice.embed(audio)
        except Exception as e:  # identity must never break a turn
            log.debug("voice embed failed: %s", e)
            return None

    def _face_emb(self, frame_rgb) -> np.ndarray | None:
        if self._face is None or frame_rgb is None:
            return None
        try:
            faces = self._face.embed_faces(frame_rgb, max_faces=1)
        except Exception as e:
            log.debug("face embed failed: %s", e)
            return None
        return faces[0][0] if faces else None

    # ── the two public operations ─────────────────────────────────────────────
    def identify(self, audio=None, frame_rgb=None) -> IdentityResult:
        """Who is speaking right now? Never raises; STRANGER when unsure.

        Fuses face+voice — used for enrolment re-resolution. For deciding whose
        SESSION (and memories) this turn belongs to, use ``identify_speaker``,
        which is voice-owned by design.
        """
        v_emb = self._voice_emb(audio)
        f_emb = self._face_emb(frame_rgb)
        v_match = self.registry.match("voice", v_emb) if v_emb is not None else None
        f_match = self.registry.match("face", f_emb) if f_emb is not None else None
        result = self.fuser.fuse(f_match, v_match)
        if result.is_known:
            log.info("recognised %s (%.2f via %s)", result.name, result.score, result.via)
            # Continual adaptation: a confident joint match refreshes samples.
            if result.via == "face+voice" and result.score >= self.reinforce_threshold:
                if v_emb is not None:
                    self.registry.add_embedding(result.person_id, "voice", v_emb)
                if f_emb is not None:
                    self.registry.add_embedding(result.person_id, "face", f_emb)
        return result

    def identify_speaker(self, audio=None, frame_rgb=None
                         ) -> tuple[IdentityResult, str | None]:
        """Who is *speaking* right now — decided by VOICE only.

        The session (and every memory keyed to it) belongs to whoever the voice
        names; the face channel is kept purely for logging, the "I can see you"
        note, and continual reinforcement — it NEVER decides whose session this
        is. Because this runs every turn, a different enrolled voice taking over
        mid-conversation simply becomes the new session owner (live handover).

        Returns ``(speaker, face_name)``: ``speaker`` (STRANGER when the voice is
        unsure) keys per-person memory; ``face_name`` is who the camera
        recognises this instant (may differ, or be None) — for perception notes
        only. Never raises.
        """
        v_emb = self._voice_emb(audio)
        f_emb = self._face_emb(frame_rgb)
        v_match = self.registry.match("voice", v_emb) if v_emb is not None else None
        f_match = self.registry.match("face", f_emb) if f_emb is not None else None

        # Session owner: voice, gated by the voice threshold. Face never decides it.
        if v_match is not None and v_match[2] >= self.fuser.voice_threshold:
            speaker = IdentityResult(v_match[0], v_match[1], v_match[2], via="voice")
        else:
            speaker = STRANGER

        # Face is a perception signal only — surfaced in the log and returned for
        # the "I can see you" note, but kept out of memory attribution.
        face_name = None
        if f_match is not None:
            f_pid, f_name, f_score = f_match
            if f_score >= self.fuser.face_threshold:
                face_name = f_name
                log.info("face sees %s (%.2f)", f_name, f_score)
            else:
                log.debug("face inconclusive (best %s %.2f)", f_name, f_score)

        if speaker.is_known:
            log.info("speaker is %s (voice %.2f)", speaker.name, speaker.score)
            # Continual adaptation: when the face AGREES with the voice, refresh
            # both channels' samples (same behaviour as identify()).
            if (f_match is not None and f_match[0] == speaker.person_id
                    and f_match[2] >= self.reinforce_threshold):
                if v_emb is not None:
                    self.registry.add_embedding(speaker.person_id, "voice", v_emb)
                if f_emb is not None:
                    self.registry.add_embedding(speaker.person_id, "face", f_emb)
        return speaker, face_name

    def enroll(self, name: str, audio=None, frame_rgb=None) -> tuple[IdentityResult, list[str]]:
        """Enrol (or refresh) ``name`` from this moment's audio + frame.
        Returns (result, kinds_captured); kinds empty = nothing usable sensed."""
        v_emb = self._voice_emb(audio)
        f_emb = self._face_emb(frame_rgb)
        kinds: list[str] = []
        if v_emb is None and f_emb is None:
            return STRANGER, kinds
        pid = self.registry.enroll(name)
        if v_emb is not None:
            self.registry.add_embedding(pid, "voice", v_emb)
            kinds.append("voice")
        if f_emb is not None:
            self.registry.add_embedding(pid, "face", f_emb)
            kinds.append("face")
        via = "face+voice" if len(kinds) == 2 else kinds[0]
        return IdentityResult(pid, self.registry.name_of(pid), 1.0, via=via), kinds

    def forget(self, name: str) -> bool:
        return self.registry.forget(name)

    def voice_embedding(self, audio) -> np.ndarray | None:
        """This utterance's voice embedding (used by diarization). None-safe."""
        return self._voice_emb(audio)

    @property
    def face_detector(self):
        """The face channel's detector (Haar), shared with affect. May be None."""
        return self._face
