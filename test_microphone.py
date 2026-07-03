"""Standalone microphone test — pick a device, hear how it sounds, size the gain.

Use this to figure out what to set for `audio.input_device` and `audio.input_gain`
in config.yaml. It uses the SAME sounddevice + int16 conversion the assistant does,
so what you measure here is what the wake word + STT will see.

Setup:
    pip install sounddevice numpy soundfile

Use:
    python test_microphone.py list                        # show all devices
    python test_microphone.py record                      # 3s from default input
    python test_microphone.py record --device Brio        # match by name substring
    python test_microphone.py record --device 3           # match by index
    python test_microphone.py record --gain 12            # test with software gain
    python test_microphone.py record --save mic.wav       # keep the recording
    python test_microphone.py meter                       # live level meter (10s)
    python test_microphone.py meter --device Brio --gain 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

SR = 16000
BLOCK_MS = 30           # matches audio.block_ms in config.yaml
BLOCK = SR * BLOCK_MS // 1000  # 480 samples

# Rough guide for reading the peak of a normalized float32 recording:
#   < 0.02  = way too quiet, wake word almost certainly won't fire
#   < 0.10  = quiet, boost gain (or raise ALSA hardware gain)
#   0.10-0.60 = healthy speech level
#   > 0.85  = risking clipping
QUIET_PEAK = 0.10
GOOD_PEAK = 0.60


def resolve_device(spec):
    """Return a sounddevice-accepted device (int index or exact name string), or None
    to use the system default. Accepts a numeric string, an int, or a name substring."""
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    if spec.isdigit():
        return int(spec)
    # Substring match against input-capable devices, case-insensitive.
    needle = spec.lower()
    matches = []
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0 and needle in dev["name"].lower():
            matches.append((i, dev["name"]))
    if not matches:
        print(f"  ! no input device name contains {spec!r}. Try `list`.")
        sys.exit(2)
    if len(matches) > 1:
        print(f"  ! {spec!r} matches multiple devices — be more specific or use the index:")
        for i, name in matches:
            print(f"    [{i}] {name}")
        sys.exit(2)
    return matches[0][0]


def cmd_list(_args) -> None:
    print(sd.query_devices())
    try:
        din, dout = sd.default.device
        print(f"\nDefault input : {din}")
        print(f"Default output: {dout}")
    except Exception:  # noqa: BLE001
        pass


def _suggest_gain(peak: float) -> str:
    if peak >= GOOD_PEAK:
        return "input_gain: 1.0  (mic level is healthy — leave gain off)"
    if peak >= QUIET_PEAK:
        target = 0.4
        return f"input_gain: {target / max(peak, 1e-4):.1f}  (nudge to a comfortable level)"
    # very quiet — bump toward ~0.4 peak
    target = 0.4
    boost = target / max(peak, 1e-4)
    return (f"input_gain: {boost:.1f}  (BRIO-style quiet mic — this brings peak "
            f"~{peak:.2f} up to ~{target:.2f})")


def cmd_record(args) -> None:
    device = resolve_device(args.device)
    label = sd.query_devices(device)["name"] if device is not None else "default input"
    print(f"Recording {args.seconds:.0f}s from: {label}")
    print("  -> speak now (say the wake word)...")
    rec = sd.rec(int(args.seconds * SR), samplerate=SR, channels=1,
                 dtype="float32", device=device)
    sd.wait()
    mono = rec[:, 0]

    raw_peak = float(np.max(np.abs(mono)))
    raw_rms = float(np.sqrt(np.mean(mono ** 2)))

    # Apply software gain the same way MicCapture does (multiply, then clip to int16).
    boosted = mono * args.gain if args.gain != 1.0 else mono
    pcm16 = np.clip(boosted * 32768.0, -32768, 32767).astype(np.int16)
    int16_peak = int(np.max(np.abs(pcm16)))
    int16_rms = float(np.sqrt(np.mean(pcm16.astype(np.float64) ** 2)))
    # Fraction of samples that ended up clipped after the boost (want ~0).
    clipped = float(np.mean(np.abs(boosted) >= 1.0)) if args.gain != 1.0 else 0.0

    print()
    print("  RAW (before gain):")
    print(f"    peak = {raw_peak:.3f}  rms = {raw_rms:.4f}")
    verdict = "TOO QUIET" if raw_peak < QUIET_PEAK else "healthy"
    print(f"    verdict: {verdict}")
    if args.gain != 1.0:
        print(f"  AFTER gain x{args.gain}:")
        print(f"    int16 peak = {int16_peak}   int16 rms = {int16_rms:.0f}")
        print(f"    clipped samples: {clipped * 100:.2f}%"
              f"{'  (LOWER the gain)' if clipped > 0.01 else ''}")
    else:
        print(f"  int16 peak = {int16_peak}   int16 rms = {int16_rms:.0f}")

    print()
    print("  suggestion for config.yaml:")
    print(f"    {_suggest_gain(raw_peak)}")
    if device is not None:
        # Pull just the recognizable part of the device name for input_device:.
        name = sd.query_devices(device)["name"].split(":")[0].split("(")[0].strip()
        print(f"    input_device: {name}   # or the numeric index {device}")

    if args.save:
        try:
            import soundfile as sf  # optional dep

            sf.write(args.save, pcm16, SR, subtype="PCM_16")
            print(f"  saved: {args.save}")
        except ImportError:
            print("  ! `soundfile` not installed — skipping --save")

    if not args.no_playback:
        print("\n  playing back what the wake word will hear (post-gain int16)...")
        sd.play(pcm16.astype(np.float32) / 32768.0, samplerate=SR)
        sd.wait()


def cmd_meter(args) -> None:
    """Real-time level meter — hold the mic where you'd sit, watch the bar."""
    device = resolve_device(args.device)
    label = sd.query_devices(device)["name"] if device is not None else "default input"
    print(f"Live meter from: {label}   (gain x{args.gain})   {args.seconds:.0f}s")
    print("  bars per frame; H = would clip; . = below wake-word floor")
    print()

    peaks = []
    end = time.monotonic() + args.seconds

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"  [status: {status}]")
        mono = indata[:, 0]
        boosted = mono * args.gain if args.gain != 1.0 else mono
        p = float(np.max(np.abs(boosted)))
        peaks.append(p)
        bars = int(min(p, 1.0) * 40)
        flag = "H" if p >= 1.0 else ("." if p < 0.05 else " ")
        print(f"  {flag} {'#' * bars:<40}  peak={p:.2f}")

    try:
        with sd.InputStream(samplerate=SR, blocksize=BLOCK, channels=1,
                            dtype="float32", device=device, callback=callback):
            while time.monotonic() < end:
                sd.sleep(100)
    except KeyboardInterrupt:
        pass

    if peaks:
        overall_peak = max(peaks)
        overall_rms = float(np.sqrt(np.mean(np.array(peaks) ** 2)))
        print()
        print(f"  session peak = {overall_peak:.3f}   avg-of-peaks = {overall_rms:.3f}")
        # Suggest based on the RAW peak (undo the gain we applied above).
        raw = overall_peak / args.gain
        print(f"  {_suggest_gain(raw)}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="test_microphone",
                                 description="pick a mic, size the gain")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show all audio devices")

    r = sub.add_parser("record", help="record a few seconds and analyse")
    r.add_argument("--device", default=None,
                   help="index (e.g. 3) or name substring (e.g. Brio)")
    r.add_argument("--seconds", type=float, default=3.0)
    r.add_argument("--gain", type=float, default=1.0,
                   help="software gain applied like audio.input_gain")
    r.add_argument("--save", type=Path, default=None,
                   help="write the (post-gain int16) capture to a WAV file")
    r.add_argument("--no-playback", action="store_true",
                   help="skip playing the recording back")

    m = sub.add_parser("meter", help="live level meter")
    m.add_argument("--device", default=None,
                   help="index or name substring")
    m.add_argument("--seconds", type=float, default=10.0)
    m.add_argument("--gain", type=float, default=1.0)

    args = ap.parse_args()
    {"list": cmd_list, "record": cmd_record, "meter": cmd_meter}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
