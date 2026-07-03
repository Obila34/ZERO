"""Identity — "who is this?" from face and voice together.

Replaces the single-owner voice gate with a multi-person registry. Face and
voice are both turned into unit embeddings and matched by cosine similarity —
the exact pattern zero/voiceid/speaker.py already uses — then fused, so two
weak signals (a turned-away face + a noisy clip) confirm each other and either
alone still works.
"""
from zero.identity.fuser import IdentityFuser, IdentityResult, STRANGER
from zero.identity.registry import PersonRegistry
from zero.identity.service import IdentityService, parse_enrollment

__all__ = [
    "IdentityFuser", "IdentityResult", "STRANGER",
    "PersonRegistry", "IdentityService", "parse_enrollment",
]
