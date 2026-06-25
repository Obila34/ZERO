"""Microphone capture: a continuous 16 kHz mono int16 frame stream.

The stream runs in the background (PortAudio callback) and pushes frames onto a
queue. Wake-word and VAD stages pull frames from `frames()`. One shared capture
feeds the whole pipeline so we never fight over the mic device.
"""
from __future__ import annotations

import queue
from typing import Iterator

import numpy as np
import sounddevice as sd

from zero.utils.logging import get_logger

log = get_logger("audio.capture")


class MicCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        block_ms: int = 30,
        device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_ms / 1000)
        self.device = device
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._dropped = 0  # frames dropped since last warning (throttled logging)
        self._paused = False  # gate capture while ZERO thinks/speaks (echo guard)

    def pause(self) -> None:
        """Stop enqueueing frames — used while ZERO speaks so it can't hear itself."""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            log.debug("input status: %s", status)
        if self._paused:
            return  # drop audio captured while speaking/thinking
        # indata is float32 [-1, 1]; convert to int16 mono frame.
        mono = indata[:, 0] if indata.ndim > 1 else indata
        pcm16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16)
        try:
            self._q.put_nowait(pcm16.copy())
        except queue.Full:
            # Expected while a downstream stage (STT/LLM/TTS) blocks the consumer.
            # Throttle so it doesn't drown the log — one line per ~1s of drops.
            self._dropped += 1
            if self._dropped % 33 == 1:
                log.debug("capture queue full — dropping frames (x%d)", self._dropped)

    def start(self) -> None:
        if self._stream is not None:
            return
        # PortAudio's ALSA->Pulse bridge sometimes fails to start its realtime
        # callback thread on the first try (PaErrorCode -9987, "Wait timed out").
        # 'high' latency uses larger buffers (less RT pressure) and we retry a
        # few times, closing the half-open stream between attempts.
        import time as _time

        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    device=self.device,
                    channels=1,
                    dtype="float32",
                    latency="high",
                    callback=self._callback,
                )
                self._stream.start()
                log.info("mic started (%d Hz, %d-sample frames)",
                         self.sample_rate, self.block_size)
                return
            except sd.PortAudioError as err:
                last_err = err
                log.warning("mic start attempt %d/3 failed: %s", attempt, err)
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                _time.sleep(0.5)
        raise RuntimeError(f"could not start mic after 3 attempts: {last_err}")

    def frames(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        """Yield int16 mono frames as they arrive. Blocks up to `timeout` each."""
        while True:
            try:
                yield self._q.get(timeout=timeout)
            except queue.Empty:
                continue

    def drain(self) -> None:
        """Discard buffered frames (call before LISTENING to drop stale audio)."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("mic stopped")

    def __enter__(self) -> "MicCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
