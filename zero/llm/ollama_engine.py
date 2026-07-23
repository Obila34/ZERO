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
        num_ctx: int = 8192,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # -1 pins the model in RAM indefinitely so it never cold-reloads (~28s on Pi).
        self.keep_alive = keep_alive
        # Explicit context size: Ollama's default can silently truncate the
        # system prompt + memory + history, which also shifts the cached prefix.
        self.num_ctx = int(num_ctx)
        # Last turn's prompt token count, read from Ollama's `done` message so the
        # caller can see how full the window actually is (0 until the first reply).
        self.last_prompt_tokens = 0
        self._session = requests.Session()  # keep-alive across turns

    def warmup(self, messages: list[Message] | None = None) -> None:
        """Load the model into RAM at startup so the first real query is fast.

        Must use the SAME options as stream() — a different num_ctx makes
        Ollama rebuild the runner on the first real request (a multi-second
        first-token stall). Passing the real system prompt as ``messages``
        also pre-fills the prefix cache, so the first turn skips the big
        persona+memory prefill entirely.
        """
        try:
            log.info("warming up %s (loading into RAM)...", self.model)
            self._session.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages or [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "think": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 1,
                        "num_ctx": self.num_ctx,
                    },
                },
                timeout=180,
            ).raise_for_status()
            log.info("LLM warm and pinned in RAM (prefix cached)")
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
                "num_ctx": self.num_ctx,
            },
        }
        try:
            resp = self._session.post(
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
                    self._log_usage(obj)
                    if not emitted:
                        # No content came back — log why so empty replies are visible.
                        log.warning("empty reply (done_reason=%s)", obj.get("done_reason"))
                    break
        finally:
            # Runs on normal exit AND when the consumer abandons the generator
            # (barge-in) — closes the HTTP stream so Ollama stops generating.
            resp.close()

    def _log_usage(self, done: dict) -> None:
        """Record and log the prompt/output token counts from Ollama's final
        `done` message. This runs AFTER the last token is streamed, so it never
        sits on the first-token critical path. A prompt that fills most of the
        window means older turns or the memory block are about to be truncated —
        warn so compaction (or a larger num_ctx) can be applied deliberately."""
        pt = done.get("prompt_eval_count")
        et = done.get("eval_count")
        if pt is None:
            return
        pt = int(pt)
        self.last_prompt_tokens = pt
        pct = (100.0 * pt / self.num_ctx) if self.num_ctx else 0.0
        log.info("context: prompt=%d tok (%.0f%% of %d), output=%s tok",
                 pt, pct, self.num_ctx, et)
        if self.num_ctx and pt >= 0.9 * self.num_ctx:
            log.warning("context near limit: prompt %d tok is >=90%% of num_ctx "
                        "%d — Ollama truncates from the FRONT, so persona/memory "
                        "can be silently dropped. Compact sooner or raise num_ctx.",
                        pt, self.num_ctx)
