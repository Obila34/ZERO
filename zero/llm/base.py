"""LLM interface: chat messages in, reply text out (streaming by sentence)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

# A message is {"role": "system"|"user"|"assistant", "content": str}.
# A user turn MAY also carry "images": [base64_jpeg, ...] for a multimodal model
# (e.g. a vision-capable Gemma) — Ollama accepts these on the message directly.
Message = dict[str, Any]


class LLM(ABC):
    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield reply text chunks as they are generated (for early TTS start)."""
        raise NotImplementedError

    def complete(self, messages: list[Message]) -> str:
        """Convenience: collect the stream into a single string."""
        return "".join(self.stream(messages))

    def close(self) -> None:
        """Release resources. Optional."""
