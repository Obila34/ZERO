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
import queue
import threading
import time

import numpy as np

from zero.stt.base import STT
from zero.utils.logging import get_logger

log = get_logger("stt.kyutai")

_SR = 24000        # Kyutai's native rate — audio is resampled to this
_FRAME = 1920      # samples per Audio message (80 ms at 24 kHz)
_SILENCE = np.zeros(_FRAME, dtype=np.float32)


def _resample(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == _SR or audio.size == 0:
        return audio.astype(np.float32)
    n = int(round(audio.size * _SR / src_sr))
    return np.interp(np.linspace(0.0, 1.0, n, endpoint=False),
                     np.linspace(0.0, 1.0, audio.size, endpoint=False),
                     audio).astype(np.float32)


class KyutaiStreamSession:
    """A LIVE recognition session: audio goes in while the person is still
    talking, words come back as they are recognised.

    This is the whole point of a streaming ASR model. The batch wrapper below
    pays ~1.2 s per utterance because it feeds a finished clip and the model
    can only advance one 80 ms step at a time. Here the stepping happens
    *during* speech, where it is free — so when the person stops, the
    transcript is already written except for the model's small lookahead.

    Threading: ``push()`` is called from the capture loop and must never block
    or raise. A worker thread owns the socket; audio crosses on a queue, words
    come back under a lock. The model only advances when it is fed, so the
    sender emits silence whenever the mic hasn't supplied anything — that
    keeps the recogniser moving through its lookahead during pauses instead of
    stalling right when we need the last word.
    """

    def __init__(self, url: str, api_key: str = "public_token",
                 sample_rate: int = 16000, timeout: float = 20.0):
        self.url = url if url.endswith("/api/asr-streaming") else \
            f"{url.rstrip('/')}/api/asr-streaming"
        self.api_key = api_key
        self.sample_rate = int(sample_rate)
        self.timeout = float(timeout)
        self._q: "queue.Queue" = queue.Queue(maxsize=512)
        self._words: list[str] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._marker = threading.Event()   # set when the server echoes our marker
        self._closing = threading.Event()  # finalize() asked for the marker
        self._failed: Exception | None = None
        self._buf = np.zeros(0, dtype=np.float32)

    # -- capture-side API (must never block or raise) -----------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="kyutai-live",
                                        daemon=True)
        self._thread.start()

    def push(self, frame) -> None:
        """Feed one mic frame (int16 or float). Resamples and re-frames to the
        24 kHz/1920-sample grid the server expects. Drops audio rather than
        blocking the capture loop if the sender falls behind."""
        try:
            f = np.asarray(frame)
            if f.dtype == np.int16:
                f = f.astype(np.float32) / 32768.0
            self._buf = np.concatenate([self._buf,
                                        _resample(f.reshape(-1), self.sample_rate)])
            while self._buf.size >= _FRAME:
                chunk, self._buf = self._buf[:_FRAME], self._buf[_FRAME:]
                try:
                    self._q.put_nowait(chunk)
                except queue.Full:
                    pass  # sender is behind; losing a frame beats stalling the mic
        except Exception as e:
            log.debug("live STT push failed: %s", e)

    def text(self) -> str:
        """The transcript SO FAR — safe to read at any moment."""
        with self._lock:
            return " ".join(self._words).strip()

    def finalize(self, timeout: float = 3.0) -> str:
        """End of turn: ask for the marker and wait for the tail to flush.
        Returns the complete transcript. Fast, because everything except the
        model's lookahead was already transcribed while they were speaking."""
        if self._thread is None:
            return self.text()
        self._closing.set()
        self._marker.wait(timeout=timeout)
        return self.text()

    def close(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=2.0)

    @property
    def failed(self) -> Exception | None:
        return self._failed

    # -- worker -------------------------------------------------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._session())
        except Exception as e:
            self._failed = e
            log.warning("live STT session ended: %s", e)
        finally:
            self._marker.set()  # never leave finalize() waiting on a dead socket

    async def _session(self) -> None:
        import msgpack
        import websockets

        async with websockets.connect(
                self.url, additional_headers={"kyutai-api-key": self.api_key},
                open_timeout=self.timeout, close_timeout=2.0) as ws:

            async def send():
                # A second of silence up front: the model needs it to settle.
                for _ in range(int(_SR / _FRAME)):
                    await ws.send(msgpack.packb(
                        {"type": "Audio", "pcm": [0.0] * _FRAME},
                        use_single_float=True))
                sent_marker = False
                period = _FRAME / _SR            # 80 ms of audio per frame
                next_at = time.monotonic()
                while not self._stop.is_set():
                    # REALTIME PACING while listening. Each frame is 80 ms of
                    # audio, so emitting one every 5 ms would push ~16x
                    # realtime and bury the speech in a flood of padding —
                    # which is exactly what produced an empty transcript. Send
                    # at most one frame per 80 ms of wall clock, using mic
                    # audio when it's there and silence when it isn't (the
                    # model only advances when fed).
                    if not sent_marker:
                        now = time.monotonic()
                        if now < next_at:
                            await asyncio.sleep(min(0.005, next_at - now))
                            continue
                        next_at = max(next_at + period, now - period)
                    try:
                        chunk = self._q.get_nowait()
                    except queue.Empty:
                        chunk = _SILENCE
                    await ws.send(msgpack.packb(
                        {"type": "Audio", "pcm": [float(x) for x in chunk]},
                        use_single_float=True))
                    if self._closing.is_set() and not sent_marker and self._q.empty():
                        await ws.send(msgpack.packb({"type": "Marker", "id": 0},
                                                    use_single_float=True))
                        # Past the marker, drop the pacing: pumping silence as
                        # fast as the socket takes it is what makes the tail
                        # short instead of realtime-long.
                        sent_marker = True
                    await asyncio.sleep(0)

            send_task = asyncio.create_task(send())
            try:
                async for message in ws:
                    data = msgpack.unpackb(message, raw=False)
                    kind = data.get("type")
                    if kind == "Word":
                        with self._lock:
                            self._words.append(data["text"])
                    elif kind == "Marker":
                        self._marker.set()
                        break
                    if self._stop.is_set():
                        break
            finally:
                send_task.cancel()


class KyutaiSTT(STT):
    def __init__(self, url: str, api_key: str = "public_token",
                 timeout: float = 30.0, lead_silence_s: float = 1.0,
                 tail_silence_s: float = 0.5, flush_silence_s: float = 35.0):
        self.url = url.rstrip("/")
        if not self.url.endswith("/api/asr-streaming"):
            self.url = f"{self.url}/api/asr-streaming"
        self.api_key = api_key
        self.timeout = float(timeout)
        self._lead = float(lead_silence_s)
        self._tail = float(tail_silence_s)
        # Upper bound on post-marker silence. The send loop is cancelled the
        # moment the marker returns, so this is only a runaway guard.
        self._flush = float(flush_silence_s)
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

    def live_session(self, sample_rate: int) -> "KyutaiStreamSession":
        """A live session bound to this engine's endpoint — the streaming path
        used by the capture loop."""
        return KyutaiStreamSession(self.url, self.api_key,
                                   sample_rate=sample_rate, timeout=self.timeout)

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

            async def send_audio(chunk):
                await ws.send(msgpack.packb(
                    {"type": "Audio", "pcm": [float(x) for x in chunk]},
                    use_single_float=True))

            async def send():
                for chunk in chunk_iter:
                    await send_audio(chunk)
                    # Yield to the event loop so received words are processed
                    # while we're still sending (that's what makes the
                    # streaming path deliver text before the audio ends).
                    await asyncio.sleep(0)
                await ws.send(msgpack.packb({"type": "Marker", "id": 0},
                                            use_single_float=True))
                # The model only advances when it is fed audio, so the marker
                # does not come back until enough further audio has been
                # pushed through its lookahead delay. Keep pumping silence
                # until the receive loop sees the marker and cancels us —
                # bounded so a dead server can't spin here forever.
                for _ in range(int(_SR * self._flush / _FRAME)):
                    await send_audio(_SILENCE)
                    await asyncio.sleep(0.001)

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
