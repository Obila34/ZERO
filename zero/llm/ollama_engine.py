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
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def stream(self, messages: list[Message]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
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

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = obj.get("message", {}).get("content", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break
