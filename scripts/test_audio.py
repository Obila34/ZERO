#!/usr/bin/env python3
"""Audio device tester — find the right input/output indices for config.yaml.

Run with no arguments to see a clean table of every device:

    python scripts/test_audio.py

Then test the ones you care about:

    python scripts/test_audio.py --output 1          # play a test tone on device 1
    python scripts/test_audio.py --input 0           # record 3s on device 0, show level
    python scripts/test_audio.py --loopback -i 0 -o 1 # record on 0, play back on 1

Whatever indices give you a clear tone (output) and a healthy level (input) are
the numbers to put under `audio.input_device` / `audio.output_device`.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import sounddevice as sd
except Exception as e:  # noqa: BLE001
    print("ERROR: could not import sounddevice — is the venv active and deps installed?")
    print(f"       {e}")
    sys.exit(1)


def print_table() -> None:
    """Pretty table of all devices, marking which can record / play."""
    devices = sd.query_devices()
    try:
        default_in, default_out = sd.default.device
    except Exception:  # noqa: BLE001
        default_in, default_out = -1, -1

    print()
    print(f"{'idx':>3}  {'in':>3} {'out':>3}  {'rate':>7}  {'use':<8} name")
    print("-" * 70)
    for i, d in enumerate(devices):
        nin = d["max_input_channels"]
        nout = d["max_output_channels"]
        use = []
        if nin > 0:
            use.append("INPUT")
        if nout > 0:
            use.append("OUTPUT")
        marker = ""
        if i == default_in:
            marker += "*in "
        if i == default_out:
            marker += "*out"
        print(
            f"{i:>3}  {nin:>3} {nout:>3}  {int(d['default_samplerate']):>7}  "
            f"{'/'.join(use):<8} {d['name']} {marker}"
        )
    print("-" * 70)
    print("* = current system default.  'in' = mic channels, 'out' = speaker channels.")
    print("A microphone needs in >= 1; a speaker needs out >= 1.\n")
    print("Put the chosen indices in config.yaml:")
    print("  audio:")
    print(f"    input_device:  {default_in if default_in >= 0 else '<idx>'}")
    print(f"    output_device: {default_out if default_out >= 0 else '<idx>'}")
    print()


def test_output(device: int | None, seconds: float, sample_rate: int = 44100) -> None:
    """Play a short 440 Hz tone so you can confirm the speaker works."""
    print(f"Playing a {seconds:.0f}s tone on output device {device}... (listen)")
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    # gentle fade in/out to avoid clicks
    fade = int(0.02 * sample_rate)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    sd.play(tone, samplerate=sample_rate, device=device)
    sd.wait()
    print("Done. Heard a clear beep? Then that's your output_device.\n")


def test_input(device: int | None, seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """Record from a mic and report the peak level with a tiny meter."""
    print(f"Recording {seconds:.0f}s from input device {device} — speak now...")
    rec = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                 channels=1, dtype="float32", device=device)
    sd.wait()
    peak = float(np.max(np.abs(rec)))
    bars = int(peak * 40)
    meter = "#" * bars + "-" * (40 - bars)
    print(f"  level [{meter}] peak={peak:.3f}")
    if peak < 0.01:
        print("  -> TOO QUIET. Wrong device, muted, or gain too low.\n")
    elif peak > 0.98:
        print("  -> CLIPPING. Lower the mic gain a touch.\n")
    else:
        print("  -> Good signal. That's your input_device.\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Find/test audio device indices.")
    ap.add_argument("-i", "--input", type=int, default=None, help="input device index to test")
    ap.add_argument("-o", "--output", type=int, default=None, help="output device index to test")
    ap.add_argument("--loopback", action="store_true",
                    help="record on --input then play it back on --output")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    # Always show the table first — it's the main thing the user needs.
    print_table()

    if args.output is not None and not args.loopback:
        test_output(args.output, args.seconds)

    if args.input is not None and not args.loopback:
        test_input(args.input, args.seconds)

    if args.loopback:
        rec = test_input(args.input, args.seconds)
        print("Playing back what was recorded...")
        sd.play(rec, samplerate=16000, device=args.output)
        sd.wait()
        print("Done.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
