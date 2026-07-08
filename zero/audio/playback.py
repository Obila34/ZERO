"""Speaker playback with barge-in support.

play() streams a float32 waveform to the output device in small chunks and
checks a `should_stop` callback between chunks, so the main loop can cut
playback short the instant a new wake word fires (barge-in).
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import sounddevice as sd

from zero.utils.logging import get_logger

log = get_logger("audio.playback")


class Speaker:
    def __init__(self, device: int | str | None = None, chunk_ms: int = 50,
                 echo_ref=None, prebuffer_ms: int = 0):
        self.device = device
        self.chunk_ms = chunk_ms
        # AEC far-end feed: every chunk written to the device is also pushed
        # here so the mic side can subtract it (zero/audio/aec.py).
        self.echo_ref = echo_ref
        # Jitter buffer: hold this much audio before starting play_stream, so a
        # transient TTS generation slowdown can't underrun the device mid-word.
        self.prebuffer_ms = int(prebuffer_ms)

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> bool:
        """Play a float32 mono waveform. Returns True if it finished, False if
        interrupted by should_stop()."""
        if audio.size == 0:
            return True
        audio = np.asarray(audio, dtype=np.float32)
        chunk = max(1, int(sample_rate * self.chunk_ms / 1000))

        with sd.OutputStream(
            samplerate=sample_rate,
            device=self.device,
            channels=1,
            dtype="float32",
        ) as stream:
            for start in range(0, audio.size, chunk):
                if should_stop is not None and should_stop():
                    log.info("playback interrupted (barge-in)")
                    return False
                block = audio[start : start + chunk]
                stream.write(block.reshape(-1, 1))
                if self.echo_ref is not None:
                    self.echo_ref.push(block, sample_rate)
        return True

    def play_stream(
        self,
        chunks: Iterable[np.ndarray],
        sample_rate: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> bool:
        """Play audio from an iterator of float32 chunks on a SINGLE open stream
        (gapless). Returns True if finished, False if interrupted by should_stop().
        Used for streaming TTS — first audio plays the moment the first chunk lands.
        """
        sub = max(1, int(sample_rate * self.chunk_ms / 1000))
        stream: sd.OutputStream | None = None
        # Jitter buffer: gather at least this many samples before the FIRST
        # write, then let the device buffer ride out generation slowdowns.
        need = int(sample_rate * self.prebuffer_ms / 1000)
        primed = self.prebuffer_ms <= 0
        pending: list[np.ndarray] = []
        buffered = 0

        def write(block: np.ndarray) -> bool:
            nonlocal stream
            if should_stop is not None and should_stop():
                log.info("playback interrupted (barge-in)")
                return False
            if stream is None:
                stream = sd.OutputStream(samplerate=sample_rate,
                                         device=self.device, channels=1,
                                         dtype="float32")
                stream.start()
            stream.write(block.reshape(-1, 1))
            if self.echo_ref is not None:
                self.echo_ref.push(block, sample_rate)
            return True

        try:
            for audio in chunks:
                if audio is None or getattr(audio, "size", 0) == 0:
                    continue
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if not primed:
                    pending.append(audio)
                    buffered += audio.size
                    if buffered < need:
                        continue
                    audio = np.concatenate(pending)  # prime reached — flush all
                    pending, primed = [], True
                for start in range(0, audio.size, sub):
                    if not write(audio[start : start + sub]):
                        return False
            if pending:  # reply shorter than the prebuffer — play what we have
                leftover = np.concatenate(pending)
                for start in range(0, leftover.size, sub):
                    if not write(leftover[start : start + sub]):
                        return False
            return True
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
