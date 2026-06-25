"""The conversation state machine.

    IDLE ──(wake word)──> LISTENING ──(end of speech)──> THINKING
      ▲                                                     │
      └──────────── SPEAKING <──────────── (reply) ─────────┘

Kept deliberately tiny — main.py drives the transitions; this just names the
states and the legal moves so the loop stays readable.
"""
from __future__ import annotations

from enum import Enum


class State(str, Enum):
    IDLE = "idle"            # waiting for the wake word
    LISTENING = "listening"  # capturing the user's utterance
    THINKING = "thinking"    # STT + LLM running
    SPEAKING = "speaking"    # TTS playback (interruptible via barge-in)


# Legal transitions; main.py asserts against this to catch wiring bugs early.
TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING},
    State.LISTENING: {State.THINKING, State.IDLE},     # IDLE = nothing said
    State.THINKING: {State.SPEAKING, State.IDLE, State.LISTENING},  # LISTENING = empty reply, stay in convo
    State.SPEAKING: {State.IDLE, State.LISTENING},     # LISTENING = barge-in
}


def can_transition(src: State, dst: State) -> bool:
    return dst in TRANSITIONS.get(src, set())
