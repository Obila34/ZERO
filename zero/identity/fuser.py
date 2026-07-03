"""IdentityFuser — combine a face match and a voice match into one answer.

fused = w_face·face_cos + w_voice·voice_cos when both signals name the SAME
person; a single signal falls back to its own (stricter) threshold. When the
two signals disagree, the higher weighted score competes alone — two weak,
conflicting matches must never manufacture a confident identification.
"""
from __future__ import annotations

from dataclasses import dataclass

Match = tuple[int, str, float] | None  # (person_id, name, score)


@dataclass(frozen=True)
class IdentityResult:
    person_id: int | None
    name: str | None
    score: float
    via: str  # "face" | "voice" | "face+voice" | "none"

    @property
    def is_known(self) -> bool:
        return self.person_id is not None


STRANGER = IdentityResult(person_id=None, name=None, score=0.0, via="none")


class IdentityFuser:
    def __init__(self, w_face: float = 0.55, w_voice: float = 0.45,
                 threshold: float = 0.50, face_threshold: float = 0.42,
                 voice_threshold: float = 0.45):
        total = float(w_face) + float(w_voice)
        if total <= 0:
            raise ValueError("fusion weights must be positive")
        self.w_face = float(w_face) / total
        self.w_voice = float(w_voice) / total
        self.threshold = float(threshold)
        self.face_threshold = float(face_threshold)
        self.voice_threshold = float(voice_threshold)

    def _single(self, match: Match, kind: str, threshold: float) -> IdentityResult:
        if match is None:
            return STRANGER
        pid, name, score = match
        if score >= threshold:
            return IdentityResult(pid, name, score, via=kind)
        return STRANGER

    def fuse(self, face: Match, voice: Match) -> IdentityResult:
        if face is not None and voice is not None:
            if face[0] == voice[0]:
                fused = self.w_face * face[2] + self.w_voice * voice[2]
                if fused >= self.threshold:
                    return IdentityResult(face[0], face[1], fused, via="face+voice")
                return STRANGER
            # Disagreement: strongest weighted signal competes on its own merits.
            if self.w_face * face[2] >= self.w_voice * voice[2]:
                return self._single(face, "face", self.face_threshold)
            return self._single(voice, "voice", self.voice_threshold)
        if face is not None:
            return self._single(face, "face", self.face_threshold)
        if voice is not None:
            return self._single(voice, "voice", self.voice_threshold)
        return STRANGER
