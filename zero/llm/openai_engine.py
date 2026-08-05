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
import time
from typing import Iterator

import requests

from zero.llm.base import LLM, Message
from zero.utils.logging import get_logger

log = get_logger("llm.openai")


# A reasoning pass that leaks into `content` shows up as a bare marker welded
# (or loosely joined) to the first real word: "thoughtIt's looking warm...",
# "thought Heythere!" — which then gets SPOKEN. The first version required the
# capital IMMEDIATELY after the marker, so the spaced form slipped through and
# was said aloud at a live venue. Now optional space/colon is allowed before
# the capital. Still narrow on purpose: the reply must START with the bare
# lowercase marker and the next real character must be a capital, so ordinary
# words ("thoughtful", "thinking about it") are untouched.
_REASON_PREFIX_RE = re.compile(
    r"^\s*(?:thought|thinking|reasoning):?\s*(?=[A-Z])")
# The marker alone (the capital arrives in the NEXT stream chunk): drop the
# whole chunk and keep treating the next one as the reply's first.
_REASON_BARE_RE = re.compile(r"^\s*(?:thought|thinking|reasoning):?\s*$")


def _strip_reasoning_prefix(chunk: str) -> str:
    if _REASON_BARE_RE.match(chunk):
        log.warning("dropped a bare leaked reasoning marker (%r)", chunk)
        return ""
    cleaned = _REASON_PREFIX_RE.sub("", chunk)
    if cleaned != chunk:
        log.warning("stripped a leaked reasoning prefix from the reply")
    return cleaned


class _DegenerateGuard:
    """Cuts a reply that has collapsed into repetition.

    A live session produced `thought thought thought thought ...` eleven times
    and SPOKE it. Stripping a leading marker doesn't help once the model is
    looping. This watches the token stream and stops it the moment the same
    short token repeats, or a short phrase cycles — the reply is already
    ruined at that point, and saying nothing is far better than saying that in
    front of an audience.

    Deliberately narrow so ordinary speech survives: real replies repeat words
    ("very very"), but not 4 times in a row, and legitimate long words are
    excluded by the length cap.
    """

    def __init__(self, max_repeats: int = 4, window: int = 24):
        self.max_repeats = max_repeats
        self._last = None
        self._run = 0
        self._recent: list[str] = []
        self._window = window
        self.tripped = False

    def feed(self, chunk: str) -> bool:
        """True = the stream has gone degenerate; stop consuming it."""
        tok = chunk.strip().lower()
        if not tok or len(tok) > 14:
            self._last, self._run = None, 0
            return False
        if tok == self._last:
            self._run += 1
            if self._run >= self.max_repeats:
                self.tripped = True
                log.error("degenerate reply detected (%r x%d) — cutting the "
                          "stream so it is never spoken", tok, self._run + 1)
                return True
        else:
            self._last, self._run = tok, 0
        # Also catch a two-token cycle ("a b a b a b ...").
        self._recent.append(tok)
        if len(self._recent) > self._window:
            self._recent.pop(0)
        if len(self._recent) >= 8:
            tail = self._recent[-8:]
            if len(set(tail)) <= 2 and tail[0] != tail[1]:
                self.tripped = True
                log.error("degenerate reply detected (cycling %s) — cutting",
                          sorted(set(tail)))
                return True
        return False


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
                 temperature: float = 0.7, max_tokens: int = 160,
                 connect_retries: int = 3, connect_timeout: float = 5.0,
                 read_timeout: float = 60.0):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.connect_retries = max(1, int(connect_retries))
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
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
        # Connect with bounded retries. On exhibition wifi the first attempt
        # can fail on a transient drop; retrying costs a few hundred ms and
        # saves the turn. Only the CONNECT is retried — once tokens are
        # flowing a mid-stream failure is handled by the caller, because
        # replaying a half-spoken reply would repeat words aloud.
        resp = None
        last_err = None
        for attempt in range(1, self.connect_retries + 1):
            try:
                resp = self._session.post(
                    f"{self.host}/v1/chat/completions", headers=self._headers,
                    json=self._payload(messages, stream=True,
                                       max_tokens=self.max_tokens),
                    stream=True, timeout=(self.connect_timeout, self.read_timeout))
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                last_err = e
                resp = None
                if attempt < self.connect_retries:
                    delay = 0.4 * attempt
                    log.warning("LLM connect failed (%s) — retry %d/%d in %.1fs",
                                e, attempt, self.connect_retries - 1, delay)
                    time.sleep(delay)
        if resp is None:
            log.error("LLM unreachable after %d attempts: %s",
                      self.connect_retries, last_err)
            return
        first = True
        guard = _DegenerateGuard()
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
                        chunk = _strip_reasoning_prefix(chunk)
                        if not chunk:
                            continue  # still first: a bare marker was dropped
                        first = False
                    if guard.feed(chunk):
                        return   # degenerate: stop before it reaches the voice
                    yield chunk
        except requests.RequestException as e:
            # Network died mid-reply. Yield nothing further; the caller speaks
            # a holding line rather than trailing off into silence.
            log.warning("LLM stream dropped mid-reply: %s", e)
        finally:
            # Runs on normal exit AND when the consumer abandons the generator
            # (barge-in) — closing the response makes the server stop generating.
            resp.close()
