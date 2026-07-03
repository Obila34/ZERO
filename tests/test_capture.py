"""Tests for MicCapture — especially the sample-rate fallback path used when
a USB device (e.g. C-Media CM108) refuses 16 kHz."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest


class _FakePortAudioError(Exception):
    pass


class _FakeStream:
    """Records what samplerate/blocksize it was opened with. `start()` succeeds
    unless the caller pre-loads it with an error."""

    def __init__(self, samplerate, blocksize, device, channels, dtype, latency, callback):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.channels = channels
        self.callback = callback
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        pass


def _install_fake_sd(*, default_rate=48000, fail_at_16k=False):
    """Wire a minimal sounddevice stub into sys.modules BEFORE importing capture."""
    sys.modules.pop("zero.audio.capture", None)

    opens = []

    def input_stream(*, samplerate, blocksize, device, channels, dtype, latency, callback):
        opens.append(samplerate)
        if fail_at_16k and samplerate == 16000:
            raise _FakePortAudioError("Invalid sample rate", -9997)
        return _FakeStream(samplerate, blocksize, device, channels, dtype, latency, callback)

    fake = types.ModuleType("sounddevice")
    fake.PortAudioError = _FakePortAudioError
    fake.InputStream = input_stream
    fake.query_devices = lambda idx=None: {
        "name": "USB Audio Device",
        "max_input_channels": 1,
        "default_samplerate": default_rate,
    }
    sys.modules["sounddevice"] = fake
    return fake, opens


def test_start_opens_at_target_rate_by_default():
    _, opens = _install_fake_sd(fail_at_16k=False)
    from zero.audio.capture import MicCapture

    mic = MicCapture(sample_rate=16000, block_ms=30, device=0)
    mic.start()

    assert opens == [16000]
    assert mic._native_rate is None
    assert mic._resample_up is None


def test_start_falls_back_to_native_rate_on_paInvalidSampleRate():
    _, opens = _install_fake_sd(default_rate=48000, fail_at_16k=True)
    from zero.audio.capture import MicCapture

    mic = MicCapture(sample_rate=16000, block_ms=30, device=0)
    mic.start()

    # 16 kHz tried first, then the device's default 48 kHz.
    assert opens == [16000, 48000]
    assert mic._native_rate == 48000
    # 16000/48000 reduces to 1/3.
    assert mic._resample_up == 1
    assert mic._resample_down == 3


def test_start_falls_back_to_hardcoded_48k_if_query_gives_no_rate():
    """Guard the last-resort path: no default_samplerate in the device info."""
    _install_fake_sd(fail_at_16k=True)
    import sounddevice as sd

    sd.query_devices = lambda idx=None: {
        "name": "USB Audio Device", "max_input_channels": 1,
    }
    from zero.audio.capture import MicCapture

    mic = MicCapture(sample_rate=16000, block_ms=30, device=0)
    mic.start()
    assert mic._native_rate == 48000


def test_callback_resamples_native_block_down_to_target_rate():
    """The most important behaviour: downstream code sees 16 kHz frames even
    when the mic is being opened at 48 kHz."""
    _install_fake_sd(default_rate=48000, fail_at_16k=True)
    from zero.audio.capture import MicCapture

    mic = MicCapture(sample_rate=16000, block_ms=30, device=0, gain=1.0)
    mic.start()

    # 30 ms at 48 kHz = 1440 samples. Feed a synthetic input block.
    native_block = 1440
    indata = np.full((native_block, 1), 0.5, dtype=np.float32)
    mic._callback(indata, native_block, None, None)

    # The queue should now hold one frame at 16 kHz.
    frame = mic._q.get_nowait()
    # 30 ms at 16 kHz = 480 samples.
    assert 470 <= frame.shape[0] <= 490
    assert frame.dtype == np.int16


def test_callback_passthrough_when_no_resample_needed():
    _install_fake_sd(fail_at_16k=False)
    from zero.audio.capture import MicCapture

    mic = MicCapture(sample_rate=16000, block_ms=30, device=0, gain=1.0)
    mic.start()

    indata = np.full((480, 1), 0.5, dtype=np.float32)
    mic._callback(indata, 480, None, None)

    frame = mic._q.get_nowait()
    assert frame.shape[0] == 480
    assert frame.dtype == np.int16


def test_callback_still_drops_frames_when_paused():
    _install_fake_sd(fail_at_16k=False)
    from zero.audio.capture import MicCapture

    mic = MicCapture(sample_rate=16000, block_ms=30, device=0)
    mic.start()
    mic.pause()

    indata = np.zeros((480, 1), dtype=np.float32)
    mic._callback(indata, 480, None, None)
    assert mic._q.empty()
