"""Kyutai TTS (moshi-server) — streaming text-to-speech over a WebSocket.

Protocol (validated against dsm/scripts/tts_rust_server.py):
  * ``ws://host:port/api/tts_streaming?voice=<v>&format=PcmMessagePack``,
    header ``kyutai-api-key``
  * send  msgpack ``{"type": "Text", "text": "<word>"}`` per word, then
    ``{"type": "Eos"}``
  * recv  ``{"type": "Audio", "pcm": [float32...]}`` at 24 kHz until close

Why this matters for ZERO: ``synthesize_stream`` yields audio chunks the
moment they arrive (measured ~213 ms to first audio), and main.py's playback
consumer starts writing to the speaker on the first chunk — so the reply
begins speaking while the rest is still being generated.

Cue tags: the shared cue vocabulary ([laughs], [pause], ...) is Orpheus/Piper
syntax that Kyutai does not perform. They are stripped rather than spoken, so
a tag never leaks out as the literal word "laughs" — the one thing worse than
losing the cue.
"""
from __future__ import annotations

import queue
import re
import threading
from typing import Iterator
from urllib.parse import urlencode

import numpy as np

from zero.tts.base import TTS
from zero.utils.logging import get_logger

log = get_logger("tts.kyutai")

_CUE_RE = re.compile(r"\[[a-z]+\]")
_STAR_RE = re.compile(r"\*(\w+)\*")


def _clean(text: str) -> str:
    """Drop cue tags Kyutai can't perform and emphasis stars; keep the words."""
    return _STAR_RE.sub(r"\1", _CUE_RE.sub(" ", text)).strip()


class KyutaiTTS(TTS):
    sample_rate = 24000

    def __init__(self, url: str, voice: str, api_key: str = "public_token",
                 timeout: float = 30.0):
        base = url.rstrip("/")
        self._base = (base if base.endswith("/api/tts_streaming")
                      else f"{base}/api/tts_streaming")
        self.voice = voice
        self.api_key = api_key
        self.timeout = float(timeout)
        self.degraded = False
        import msgpack  # noqa: F401  — fail at build, not mid-reply
        import websockets  # noqa: F401

        log.info("Kyutai TTS -> %s (voice=%s)", self._base, voice)

    def _uri(self) -> str:
        return f"{self._base}?{urlencode({'voice': self.voice, 'format': 'PcmMessagePack'})}"

    def synthesize(self, text: str) -> np.ndarray:
        pieces = [c for c in self.synthesize_stream(text) if getattr(c, "size", 0)]
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces)

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 chunks at 24 kHz as the server produces them.

        The socket is driven on a worker thread feeding a queue, so this stays
        a plain synchronous generator — main.py's TTS producer thread consumes
        it exactly like the Orpheus path, no asyncio in the audio pipeline.
        """
        words = _clean(text).split()
        if not words:
            return
        q: "queue.Queue" = queue.Queue(maxsize=256)
        stop = threading.Event()
        emitted = {"n": 0}

        def worker():
            import asyncio

            async def _run_once():
                import msgpack
                import websockets

                async with websockets.connect(
                        self._uri(),
                        additional_headers={"kyutai-api-key": self.api_key},
                        open_timeout=self.timeout, close_timeout=2.0) as ws:

                    async def send():
                        for w in words:
                            if stop.is_set():
                                return
                            await ws.send(msgpack.packb({"type": "Text",
                                                         "text": w}))
                        await ws.send(msgpack.packb({"type": "Eos"}))

                    send_task = asyncio.create_task(send())
                    try:
                        async for message in ws:
                            if stop.is_set():
                                break
                            msg = msgpack.unpackb(message, raw=False)
                            if msg.get("type") == "Audio":
                                emitted["n"] += 1
                                q.put(np.asarray(msg["pcm"], dtype=np.float32))
                    finally:
                        send_task.cancel()

            async def run():
                # One reconnect attempt on a CONNECT failure. Exhibition wifi
                # drops sockets; a single retry turns a mute reply into a
                # slightly late one. Only retry when nothing was emitted yet —
                # reconnecting mid-sentence would repeat words aloud.
                try:
                    await _run_once()
                    return
                except Exception as e:
                    if emitted["n"]:
                        raise
                    log.warning("Kyutai TTS connect failed (%s) — one retry", e)
                await asyncio.sleep(0.3)
                await _run_once()

            try:
                asyncio.run(run())
                self.degraded = False
            except Exception as e:
                self.degraded = True
                log.warning("Kyutai TTS failed: %s", e)
            finally:
                q.put(None)  # sentinel — always delivered, even on failure

        t = threading.Thread(target=worker, name="kyutai-tts", daemon=True)
        t.start()
        try:
            while True:
                item = q.get(timeout=self.timeout)
                if item is None:
                    return
                yield item
        except queue.Empty:
            self.degraded = True
            log.warning("Kyutai TTS stalled (no audio within %.0fs)", self.timeout)
        finally:
            # A barge-in abandons this generator: tell the worker to stop so we
            # don't keep synthesizing a reply nobody is listening to.
            stop.set()
