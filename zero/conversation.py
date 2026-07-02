"""Conversation memory — system prompt + the recent turns of the chat.

Two design points that matter on the Pi:

1. Cache-friendly trimming. Ollama caches the prompt prefix, so as long as we only
   APPEND messages each turn, it reprocesses just the new ones (fast, ~2s). The
   moment we drop the oldest message the prefix shifts and the whole context is
   re-read (slow, ~10-15s). So instead of trimming every turn, we let history grow
   to `trim_at` turns and only then trim back to `target` — most turns stay
   append-only (cache hit), and the expensive re-read happens rarely.

2. Long-term memory block. Durable facts loaded from SQLite are injected into the
   system message as a short block, so ZERO "remembers" across sessions without
   replaying whole past conversations (which would bloat the prompt).
"""
from __future__ import annotations

from zero.llm.base import Message


class Conversation:
    def __init__(self, system_prompt: str, history_turns: int = 3,
                 trim_at_turns: int = 8):
        self.system_prompt = system_prompt
        # Counts are in user+assistant PAIRS.
        self.target_messages = history_turns * 2          # size we trim back DOWN to
        self.trim_at_messages = max(trim_at_turns, history_turns) * 2  # trigger
        self._history: list[Message] = []
        self._memory_block = ""  # durable facts injected per-session (set once at start)

    def set_memory(self, block: str) -> None:
        """Inject long-term memory into the system prompt for this conversation.
        Set once at conversation start so the prefix stays stable (cache-friendly).
        """
        self._memory_block = block or ""

    def add_user(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})
        self._maybe_trim()

    def add_assistant(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})
        self._maybe_trim()

    def messages(self) -> list[Message]:
        system = self.system_prompt
        if self._memory_block:
            system = f"{system}\n\n{self._memory_block}"
        return [{"role": "system", "content": system}, *self._history]

    def transcript(self) -> str:
        """Plain-text dump of the chat, for the end-of-session memory extraction."""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self._history)

    def _maybe_trim(self) -> None:
        # Only trim once we've grown past the trigger — keeps most turns append-only
        # (cache hit) and pays the expensive re-read only occasionally.
        if len(self._history) > self.trim_at_messages:
            trimmed = self._history[-self.target_messages:]
            # Keep the window aligned to a user turn — a history that opens with a
            # dangling assistant message reads as replying to nothing.
            if trimmed and trimmed[0]["role"] == "assistant":
                trimmed = trimmed[1:]
            self._history = trimmed

    def reset(self) -> None:
        self._history.clear()
        # memory_block is reloaded from SQLite at the next conversation start
