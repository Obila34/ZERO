"""Persona + the emotion-tagging contract.

This is the single source of truth for HOW the model should talk and WHICH cue
tags it may emit. The TTS orchestrator consumes the very same vocabulary, so the
prompt stays engine-agnostic: the LLM always writes `[laughs]`, and whichever
engine is active translates it (Fish -> `(laughing)`, Piper -> spliced clip).
"""
from __future__ import annotations

# Shared cue vocabulary. Keep in sync with zero/tts/orchestrator.py::CUE_MAP.
CUES = ["[laughs]", "[chuckles]", "[sighs]", "[hmm]", "[pause]"]

SYSTEM_TEMPLATE = """You are {name}, {description}

Rules:
- Reply in two or three natural spoken sentences — warm and conversational, like
  a friend talking out loud. Not clipped or robotic, but never a lecture or a list.
- No emoji, no markdown, no bullet points.
- If you don't know, say so briefly. Never invent facts, names, or numbers.
"""


def build_system_prompt(name: str, description: str) -> str:
    return SYSTEM_TEMPLATE.format(
        name=name,
        description=description.strip(),
        cues=" ".join(CUES),
    )
