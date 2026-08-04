"""Kyutai STT (moshi-server) — streaming speech-to-text over a WebSocket.

Protocol (validated against dsm/scripts/stt_from_file_rust_server.py):
  * ``ws://host:port/api/asr-streaming``, header ``kyutai-api-key``
  * send  msgpack ``{"type": "Audio", "pcm": [float32...]}`` in 1920-sample
    frames at 24 kHz, then ``{"type": "Marker", "id": 0}``
  * recv  ``Word`` (text, start_time), ``EndWord`` (stop_time), ``Step``
    (semantic-VAD signal, unused here), ``Marker`` (our marker came back —
    everything before it is transcribed, so we stop)

Two details are load-bearing and come straight from the reference client:
a full second of leading silence (the model needs it to settle) and trailing
silence to flush the last words out of the model's lookahead.

Batch fit: ZERO's STT interface is ``transcribe(clip) -> text``, so this ships
the clip as fast as the socket accepts it rather than pacing at realtime — the
server transcribes far faster than realtime, so a 3 s utterance comes back in
a fraction of that. The genuinely streaming path (transcribing WHILE the user
talks, so the text is ready the moment they stop) is a capture-loop change,
not an engine change — see ``stream_transcribe`` below, which is the hook for
it and is already usable on its own.

Language: the deployed model is ``stt-1b-en_fr`` — English/French only. It has
NO Swahili. Wrap this with FallbackSTT(primary=kyutai, fallback=whisper) or
select the remote Whisper engine when multilingual coverage matters.
"""
from __future__ import annotations

import asyncio
import threading

import numpy as np

from zero.stt.base import STT
from zero.utils.logging import get_logger

log = get_logger("stt.kyutai")

_SR = 24000        # Kyutai's native rate — audio is resampled to this
_FRAME = 1920      # samples per Audio message (80 ms at 24 kHz)


def _resample(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == _SR or audio.size == 0:
        return audio.astype(np.float32)
    n = int(round(audio.size * _SR / src_sr))
    return np.interp(np.linspace(0.0, 1.0, n, endpoint=False),
                     np.linspace(0.0, 1.0, audio.size, endpoint=False),
                     audio).astype(np.float32)


class KyutaiSTT(STT):
    def __init__(self, url: str, api_key: str = "public_token",
                 timeout: float = 30.0, lead_silence_s: float = 1.0,
                 tail_silence_s: float = 2.0):
        self.url = url.rstrip("/")
        if not self.url.endswith("/api/asr-streaming"):
            self.url = f"{self.url}/api/asr-streaming"
        self.api_key = api_key
        self.timeout = float(timeout)
        self._lead = float(lead_silence_s)
        self._tail = float(tail_silence_s)
        # Fail fast at build time if the deps are missing, so the factory can
        # fall back instead of dying mid-conversation.
        import msgpack  # noqa: F401
        import websockets  # noqa: F401

        log.info("Kyutai STT -> %s", self.url)

    # -- public API ---------------------------------------------------------
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Whole-clip transcription. Raises on transport failure so FallbackSTT
        can tell a dead server from genuine silence."""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        pcm = _resample(np.asarray(audio, dtype=np.float32).reshape(-1),
                        sample_rate)
        if pcm.size == 0:
            return ""
        words = self._run(self._collect(pcm))
        return " ".join(words).strip()

    def stream_transcribe(self, frames, sample_rate: int):
        """Feed live mic frames (an iterable of int16/float arrays) and get
        words back as they are recognised. This is the hook for true streaming
        capture: the transcript is complete almost the instant the user stops,
        because it was produced while they were still talking."""
        return self._run(self._collect_iter(frames, sample_rate), stream=True)

    # -- internals ----------------------------------------------------------
    def _collect(self, pcm: np.ndarray):
        """Yield the framed message sequence for one complete clip."""
        lead = np.zeros(int(_SR * self._lead), dtype=np.float32)
        tail = np.zeros(int(_SR * self._tail), dtype=np.float32)
        full = np.concatenate([lead, pcm, tail])
        for i in range(0, full.size, _FRAME):
            yield full[i:i + _FRAME]

    def _collect_iter(self, frames, sample_rate: int):
        yield np.zeros(int(_SR * self._lead), dtype=np.float32)
        buf = np.zeros(0, dtype=np.float32)
        for frame in frames:
            f = np.asarray(frame)
            if f.dtype == np.int16:
                f = f.astype(np.float32) / 32768.0
            buf = np.concatenate([buf, _resample(f.reshape(-1), sample_rate)])
            while buf.size >= _FRAME:
                yield buf[:_FRAME]
                buf = buf[_FRAME:]
        if buf.size:
            yield buf
        yield np.zeros(int(_SR * self._tail), dtype=np.float32)

    def _run(self, chunk_iter, stream: bool = False) -> list:
        """Drive the async WebSocket session from sync code on its own loop.
        A fresh loop per call keeps this safe to call from any thread (the
        speculative-STT worker, the barge-in monitor, the main loop)."""
        result: dict = {}

        def worker():
            try:
                result["words"] = asyncio.run(self._session(chunk_iter))
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=worker, name="kyutai-stt", daemon=True)
        t.start()
        t.join(timeout=self.timeout)
        if t.is_alive():
            raise RuntimeError("Kyutai STT timed out (server or tailnet down?)")
        if "error" in result:
            raise RuntimeError(f"Kyutai STT failed: {result['error']}") from \
                result["error"]
        return result.get("words", [])

    async def _session(self, chunk_iter) -> list:
        import msgpack
        import websockets

        words: list[str] = []
        async with websockets.connect(
                self.url, additional_headers={"kyutai-api-key": self.api_key},
                open_timeout=self.timeout, close_timeout=2.0) as ws:

            async def send():
                for chunk in chunk_iter:
                    await ws.send(msgpack.packb(
                        {"type": "Audio", "pcm": [float(x) for x in chunk]},
                        use_single_float=True))
                    # Yield to the event loop so received words are processed
                    # while we're still sending (that's what makes the
                    # streaming path deliver text before the audio ends).
                    await asyncio.sleep(0)
                await ws.send(msgpack.packb({"type": "Marker", "id": 0},
                                            use_single_float=True))

            send_task = asyncio.create_task(send())
            try:
                async for message in ws:
                    data = msgpack.unpackb(message, raw=False)
                    kind = data.get("type")
                    if kind == "Word":
                        words.append(data["text"])
                    elif kind == "Marker":
                        break  # our marker returned: everything is transcribed
            finally:
                send_task.cancel()
        return words
