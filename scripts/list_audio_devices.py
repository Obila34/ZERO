#!/usr/bin/env python3
"""List audio devices and (optionally) run a record->playback loopback test.

    python scripts/list_audio_devices.py            # list devices
    python scripts/list_audio_devices.py --loopback # record 3s, play it back

Use the printed indices to set audio.input_device / audio.output_device in
config.yaml (or config.local.yaml).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sounddevice as sd


def list_devices() -> None:
    print(sd.query_devices())
    try:
        din, dout = sd.default.device
        print(f"\nDefault input : {din}")
        print(f"Default output: {dout}")
    except Exception:  # noqa: BLE001
        pass


def loopback(seconds: float = 3.0, sample_rate: int = 16000) -> None:
    print(f"Recording {seconds:.0f}s — speak now...")
    rec = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    peak = float(np.max(np.abs(rec)))
    print(f"Captured. peak level = {peak:.3f} ({'OK' if peak > 0.01 else 'TOO QUIET — check mic'})")
    print("Playing back...")
    sd.play(rec, samplerate=sample_rate)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loopback", action="store_true", help="record then play back")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    list_devices()
    if args.loopback:
        print()
        loopback(args.seconds)
    sys.exit(0)
