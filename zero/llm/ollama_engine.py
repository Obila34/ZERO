"""Ollama LLM engine — talks to a local Ollama server over HTTP.

Streams tokens so the pipeline can start TTS on the first complete sentence
instead of waiting for the whole reply (big latency win on the Pi).
"""
from __future__ import annotations

import json
from typing import Iterator

import requests

from zero.llm.base import LLM, Message
from zero.utils.logging import get_logger

log = get_logger("llm.ollama")


class OllamaLLM(LLM):
    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "llama3.2:3b",
        temperature: float = 0.7,
        max_tokens: int = 160,
        keep_alive: str | int = -1,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # -1 pins the model in RAM indefinitely so it never cold-reloads (~28s on Pi).
        self.keep_alive = keep_alive

    def warmup(self) -> None:
        """Load the model into RAM at startup so the first real query is fast."""
        try:
            log.info("warming up %s (loading into RAM)...", self.model)
            requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"num_predict": 1},
                },
                timeout=180,
            ).raise_for_status()
            log.info("LLM warm and pinned in RAM")
        except requests.RequestException as e:
            log.warning("LLM warmup failed (will load on first query): %s", e)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            # Disable "thinking" on models that support it (e.g. Gemma): we want a
            # direct spoken answer, not hidden reasoning that empties `content`.
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=payload, stream=True, timeout=120
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error("Ollama request failed: %s", e)
            return

        emitted = False
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in obj:
                    log.error("Ollama error: %s", obj["error"])
                    return
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    emitted = True
                    yield chunk
                if obj.get("done"):
                    if not emitted:
                        # No content came back — log why so empty replies are visible.
                        log.warning("empty reply (done_reason=%s)", obj.get("done_reason"))
                    break
        finally:
            # Runs on normal exit AND when the consumer abandons the generator
            # (barge-in) — closes the HTTP stream so Ollama stops generating.
            resp.close()
