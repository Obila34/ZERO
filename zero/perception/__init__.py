"""Perception polish — reading HOW things are said, and by whom.

affect.py  — multimodal emotion estimate: voice prosody + text sentiment
             (+ facial expression when an FER model is configured).
diarize.py — turn-level speaker tracking: notices when a different person
             takes over the conversation.
"""
from zero.perception.affect import AffectEstimator, AffectResult
from zero.perception.diarize import SpeakerTracker

__all__ = ["AffectEstimator", "AffectResult", "SpeakerTracker"]
