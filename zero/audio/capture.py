"""Microphone capture: a continuous 16 kHz mono int16 frame stream.

The stream runs in the background (PortAudio callback) and pushes frames onto a
queue. Wake-word and VAD stages pull frames from `frames()`. One shared capture
feeds the whole pipeline so we never fight over the mic device.

Sample-rate resilience: many USB audio devices (e.g. C-Media CM108 dongles) only
support 44.1/48 kHz — asking for 16 kHz makes PortAudio raise `paInvalidSampleRate`.
When that happens we open the stream at the device's native rate and resample
each block down to 16 kHz on the fly, so downstream stages always see the mono
int16 16 kHz frames they expect.
"""
from __future__ import annotations

import queue
from math import gcd
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
        gain: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.block_size = int(sample_rate * block_ms / 1000)
        self.device = device
        # Software input gain. Some USB/webcam mics (e.g. the Logitech BRIO)
        # capture very quietly (peak ~0.03), too low for the wake word / STT.
        # Multiply each frame by this before int16 conversion. 1.0 = unchanged.
        self.gain = float(gain) if gain else 1.0
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._dropped = 0  # frames dropped since last warning (throttled logging)
        self._paused = False  # gate capture while ZERO thinks/speaks (echo guard)
        self.aec = None  # optional echo canceller (zero/audio/aec.py)
        # Resample state — populated only when the device rejects our target rate.
        self._native_rate: int | None = None
        self._resample_up: int | None = None
        self._resample_down: int | None = None

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
        if self._native_rate is not None:
            # Device runs at native rate (e.g. 48 kHz); downsample per block so
            # downstream code always sees `self.sample_rate` samples.
            from scipy.signal import resample_poly

            mono = resample_poly(mono, up=self._resample_up, down=self._resample_down)
        if self.gain != 1.0:
            mono = mono * self.gain  # boost a quiet mic (clipped on the next line)
        pcm16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16)
        if self.aec is not None:
            pcm16 = self.aec.process(pcm16)  # subtract ZERO's own playback
        try:
            self._q.put_nowait(pcm16.copy())
        except queue.Full:
            # Expected while a downstream stage (STT/LLM/TTS) blocks the consumer.
            # Throttle so it doesn't drown the log — one line per ~1s of drops.
            self._dropped += 1
            if self._dropped % 33 == 1:
                log.debug("capture queue full — dropping frames (x%d)", self._dropped)

    def _open_stream(self, rate: int, blocksize: int):
        return sd.InputStream(
            samplerate=rate,
            blocksize=blocksize,
            device=self.device,
            channels=1,
            dtype="float32",
            latency="high",
            callback=self._callback,
        )

    def _configure_resample(self, native_rate: int) -> int:
        """Set _resample_up/_down and return the native block size to open with."""
        self._native_rate = native_rate
        g = gcd(self.sample_rate, native_rate)
        self._resample_up = self.sample_rate // g
        self._resample_down = native_rate // g
        return int(native_rate * self.block_ms / 1000)

    @staticmethod
    def _is_invalid_sample_rate(err: sd.PortAudioError) -> bool:
        # PaErrorCode -9997 = paInvalidSampleRate. On some builds `err.args` has
        # the code as the second element; fall back to string matching.
        code = err.args[1] if len(err.args) >= 2 else None
        if code == -9997:
            return True
        return "sample rate" in str(err).lower()

    def start(self) -> None:
        if self._stream is not None:
            return
        # PortAudio's ALSA->Pulse bridge sometimes fails to start its realtime
        # callback thread on the first try (PaErrorCode -9987, "Wait timed out").
        # 'high' latency uses larger buffers (less RT pressure) and we retry a
        # few times, closing the half-open stream between attempts.
        import time as _time

        rate = self.sample_rate
        blocksize = self.block_size
        last_err: Exception | None = None

        for attempt in range(1, 4):
            try:
                self._stream = self._open_stream(rate, blocksize)
                self._stream.start()
                if self._native_rate is not None:
                    log.info("mic started (native %d Hz -> resample to %d Hz, "
                             "%d-sample frames)",
                             self._native_rate, self.sample_rate, self.block_size)
                else:
                    log.info("mic started (%d Hz, %d-sample frames)",
                             self.sample_rate, self.block_size)
                return
            except sd.PortAudioError as err:
                last_err = err
                if self._is_invalid_sample_rate(err) and self._native_rate is None:
                    # Device won't do our target rate — fall back to its native
                    # rate and resample on the fly. Retry immediately, no sleep.
                    try:
                        info = sd.query_devices(self.device)
                        native = int(info.get("default_samplerate") or 0)
                    except Exception:  # noqa: BLE001
                        native = 0
                    if not native or native == self.sample_rate:
                        native = 48000  # last-resort common default
                    log.info("mic device rejected %d Hz — falling back to native "
                             "%d Hz with resampling", self.sample_rate, native)
                    blocksize = self._configure_resample(native)
                    rate = native
                    if self._stream is not None:
                        try:
                            self._stream.close()
                        except Exception:
                            pass
                        self._stream = None
                    continue  # retry with the new rate right away
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
