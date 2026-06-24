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

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            log.debug("input status: %s", status)
        # indata is float32 [-1, 1]; convert to int16 mono frame.
        mono = indata[:, 0] if indata.ndim > 1 else indata
        pcm16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16)
        try:
            self._q.put_nowait(pcm16.copy())
        except queue.Full:
            log.warning("capture queue full — dropping frame")

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        log.info("mic started (%d Hz, %d-sample frames)", self.sample_rate, self.block_size)

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
