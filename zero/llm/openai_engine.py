"""OpenAI-compatible chat engine — vLLM, llama.cpp server, TGI, or any other
server speaking /v1/chat/completions.

This is the cutover target for the GPU-node migration (vLLM serving a large
multilingual instruct model): flip ``llm.engine: openai`` and point ``host`` at
the new endpoint; the rest of ZERO — tools, memory, vision notes, streaming —
is engine-agnostic and doesn't change.

Mirrors OllamaLLM's surface exactly where main.py relies on it:
  * stream(messages)  — SSE deltas, closes the connection on abandonment so
    the server stops generating on a barge-in;
  * warmup(messages)  — pin/prepare the model at startup;
  * prefill(messages) — speculative KV warm during the endpoint wait
    (max_tokens=1, own connection, silent failure).

Vision: Ollama carries images as ``message["images"] = [b64, ...]``; the
OpenAI dialect wants content-part arrays with data URIs. _convert() translates
so the rest of the pipeline can keep the Ollama shape everywhere.
"""
from __future__ import annotations

import json
import re
from typing import Iterator

import requests

from zero.llm.base import LLM, Message
from zero.utils.logging import get_logger

log = get_logger("llm.openai")


# A reasoning pass that leaks into `content` shows up welded to the first real
# word ("thoughtIt's looking warm..."), which then gets SPOKEN. Narrow on
# purpose: only a bare marker immediately followed by a capital letter, so
# ordinary words ("thoughtful", "thinking about it") are untouched.
_REASON_PREFIX_RE = re.compile(r"^\s*(?:thought|thinking|reasoning)(?=[A-Z])")


def _strip_reasoning_prefix(chunk: str) -> str:
    cleaned = _REASON_PREFIX_RE.sub("", chunk)
    if cleaned != chunk:
        log.warning("stripped a leaked reasoning prefix from the reply")
    return cleaned


def _convert(messages: list[Message]) -> list[dict]:
    """Ollama-shaped messages -> OpenAI-shaped (images become data-URI parts)."""
    out: list[dict] = []
    for m in messages:
        images = m.get("images")
        if not images:
            out.append({"role": m["role"], "content": m["content"]})
            continue
        parts: list[dict] = [{"type": "text", "text": m["content"]}]
        for b64 in images:
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"}})
        out.append({"role": m["role"], "content": parts})
    return out


class OpenAICompatLLM(LLM):
    def __init__(self, host: str, model: str, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 160):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._session = requests.Session()  # keep-alive across turns

    def _payload(self, messages: list[Message], stream: bool,
                 max_tokens: int) -> dict:
        return {
            "model": self.model,
            "messages": _convert(messages),
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            # Gemma's chat template can emit a reasoning pass; spoken aloud it
            # comes out as a "thought" prefix glued to the answer. The Ollama
            # engine sent "think": false — this is the OpenAI-dialect
            # equivalent. Servers that don't know the key ignore it.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def warmup(self, messages: list[Message] | None = None) -> None:
        """First request loads/compiles graphs server-side; sending the real
        prefix also seeds vLLM's prefix cache so turn one starts hot."""
        try:
            log.info("warming up %s ...", self.model)
            self._session.post(
                f"{self.host}/v1/chat/completions", headers=self._headers,
                json=self._payload(
                    messages or [{"role": "user", "content": "hi"}],
                    stream=False, max_tokens=1),
                timeout=180,
            ).raise_for_status()
            log.info("LLM warm (prefix cached)")
        except requests.RequestException as e:
            log.warning("LLM warmup failed (will load on first query): %s", e)

    def prefill(self, messages: list[Message]) -> None:
        """Speculative prefix warm during the endpoint silence — same contract
        as OllamaLLM.prefill: one throwaway token, own connection (the real
        stream may start while this is in flight), never raises."""
        try:
            requests.post(
                f"{self.host}/v1/chat/completions", headers=self._headers,
                json=self._payload(messages, stream=False, max_tokens=1),
                timeout=20,
            ).raise_for_status()
        except requests.RequestException as e:
            log.debug("speculative prefill failed (harmless): %s", e)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            resp = self._session.post(
                f"{self.host}/v1/chat/completions", headers=self._headers,
                json=self._payload(messages, stream=True,
                                   max_tokens=self.max_tokens),
                stream=True, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error("LLM request failed: %s", e)
            return
        first = True
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data:"):
                    line = line[5:].strip()
                if line == b"[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                # Only ever take `content`. Servers with a reasoning parser put
                # the model's private thinking in `reasoning_content`, and that
                # must never reach the speaker.
                chunk = (choices[0].get("delta") or {}).get("content", "")
                if chunk:
                    if first:
                        first = False
                        chunk = _strip_reasoning_prefix(chunk)
                        if not chunk:
                            continue
                    yield chunk
        finally:
            # Runs on normal exit AND when the consumer abandons the generator
            # (barge-in) — closing the response makes the server stop generating.
            resp.close()
