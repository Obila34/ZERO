"""Conversation memory — system prompt + a rolling window of recent turns.

Kept small on purpose: a short history keeps the local LLM fast and stops the
context from ballooning past the model's comfort zone on the Pi.
"""
from __future__ import annotations

from zero.llm.base import Message


class Conversation:
    def __init__(self, system_prompt: str, history_turns: int = 6):
        self.system_prompt = system_prompt
        # history_turns counts user+assistant PAIRS to retain.
        self.max_messages = history_turns * 2
        self._history: list[Message] = []

    def add_user(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})
        self._trim()

    def messages(self) -> list[Message]:
        return [{"role": "system", "content": self.system_prompt}, *self._history]

    def _trim(self) -> None:
        if len(self._history) > self.max_messages:
            self._history = self._history[-self.max_messages :]

    def reset(self) -> None:
        self._history.clear()
