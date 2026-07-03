"""Standalone microphone test — pick a device, hear how it sounds, size the gain.

Use this to figure out what to set for `audio.input_device` and `audio.input_gain`
in config.yaml. It uses the SAME sounddevice + int16 conversion the assistant does,
so what you measure here is what the wake word + STT will see.

Setup:
    pip install sounddevice numpy soundfile

Use:
    python test_microphone.py list                        # show ALL audio devices
    python test_microphone.py mics                        # show only INPUT-capable devices
    python test_microphone.py set 2                       # write config.yaml -> index 2
    python test_microphone.py set "Logitech BRIO"         # or match by name substring
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


def _input_devices():
    """Yield (index, device_info) for every device with at least one input channel."""
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            yield i, dev


def cmd_mics(_args) -> None:
    """List ONLY devices PortAudio will accept as an input — this is exactly the
    set of devices `sd.InputStream(device=..., channels=1)` will bind to."""
    rows = list(_input_devices())
    if not rows:
        print("  ! no input-capable devices found. On Linux try:")
        print("      alsamixer -c <card>   # F4 (capture), unmute the mic")
        print("      arecord -l            # confirm capture is enabled")
        return
    print(f"  {'idx':>3}  {'ch':>3}  hostapi         name")
    print(f"  {'-'*3}  {'-'*3}  {'-'*14}  {'-'*40}")
    try:
        hostapis = sd.query_hostapis()
    except Exception:  # noqa: BLE001
        hostapis = []
    for idx, dev in rows:
        ch = dev.get("max_input_channels", 0)
        hostapi_idx = dev.get("hostapi", -1)
        api = hostapis[hostapi_idx]["name"] if 0 <= hostapi_idx < len(hostapis) else "?"
        print(f"  {idx:>3}  {ch:>3}  {api:<14}  {dev['name']}")
    print()
    print("  next step: python test_microphone.py set <idx-or-name-substring>")


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _rewrite_input_device(config_text: str, new_value: str) -> tuple[str, str]:
    """Return (new_text, old_value_line). Rewrites the `input_device:` line under
    the top-level `audio:` block, preserving indentation and trailing comment."""
    import re

    lines = config_text.splitlines(keepends=True)
    in_audio = False
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if re.match(r"^audio:\s*(#.*)?$", stripped):
            in_audio = True
            continue
        # Left the block when a new top-level key starts (no leading space).
        if in_audio and stripped and not line[0].isspace() and not stripped.startswith("#"):
            break
        if in_audio:
            m = re.match(r"^(\s+)input_device:\s*([^#\n]*?)(\s*#.*)?$", stripped)
            if m:
                indent, _old_val, comment = m.group(1), m.group(2), m.group(3) or ""
                new_line = f"{indent}input_device: {new_value}{comment}\n"
                old_line = line
                lines[i] = new_line
                return "".join(lines), old_line
    raise KeyError("could not find `audio.input_device:` in config.yaml")


def _device_display_name(device) -> str:
    """PortAudio-friendly name for writing into config.yaml — strips the ALSA
    " - (hw:X,Y)" suffix and the "(Realtek)"-style parenthetical so the substring
    match at runtime stays robust."""
    name = sd.query_devices(device)["name"]
    return name.split(":")[0].split("(")[0].strip()


def cmd_set(args) -> None:
    device = resolve_device(args.device)
    if device is None:
        print("  ! no --device given. usage: set <index-or-name>")
        sys.exit(2)
    info = sd.query_devices(device)
    if info.get("max_input_channels", 0) < 1:
        print(f"  ! device {device} ({info['name']!r}) has 0 input channels — "
              "PortAudio will refuse to bind it as a mic.")
        print("    unmute capture at the ALSA layer first (alsamixer -c <card>, F4).")
        sys.exit(3)

    if args.by == "index":
        new_value = str(device)
    else:
        new_value = _device_display_name(device)

    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    new_text, old_line = _rewrite_input_device(config_text, new_value)

    new_line = next(l for l in new_text.splitlines() if "input_device:" in l)
    if args.dry_run:
        print("  (dry run — config.yaml NOT written)")
        print(f"  would replace: {old_line.rstrip()}")
        print(f"           with: {new_line}")
        return

    CONFIG_PATH.write_text(new_text, encoding="utf-8")
    print(f"  updated {CONFIG_PATH.name}")
    print(f"    was: {old_line.rstrip()}")
    for line in new_text.splitlines():
        if "input_device:" in line:
            print(f"    now: {line}")
            break
    print()
    print("  test it:  python test_microphone.py record --device "
          f"{new_value!r} --gain 12")


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


def _is_invalid_sample_rate(err) -> bool:
    """PaErrorCode -9997 = paInvalidSampleRate. Match by code and by string
    (some builds only give the message)."""
    code = err.args[1] if len(err.args) >= 2 else None
    return code == -9997 or "sample rate" in str(err).lower()


def _record_with_resample(device, seconds: float) -> np.ndarray:
    """Return `seconds` of mono float32 audio at SR, resampling if the device
    refuses 16 kHz. Mirrors the fallback in zero/audio/capture.py so what you
    hear here is what the assistant will hear."""
    try:
        rec = sd.rec(int(seconds * SR), samplerate=SR, channels=1,
                     dtype="float32", device=device)
        sd.wait()
        return rec[:, 0]
    except sd.PortAudioError as err:
        if not _is_invalid_sample_rate(err):
            raise
    # Fall back to the device's native rate + resample down to SR.
    info = sd.query_devices(device) if device is not None else sd.query_devices(kind="input")
    native = int(info.get("default_samplerate") or 48000) or 48000
    print(f"  ! device rejected {SR} Hz; recording at native {native} Hz "
          "and resampling.")
    rec = sd.rec(int(seconds * native), samplerate=native, channels=1,
                 dtype="float32", device=device)
    sd.wait()
    mono_native = rec[:, 0]
    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(SR, native)
    return resample_poly(mono_native, up=SR // g, down=native // g)


def cmd_record(args) -> None:
    device = resolve_device(args.device)
    label = sd.query_devices(device)["name"] if device is not None else "default input"
    print(f"Recording {args.seconds:.0f}s from: {label}")
    print("  -> speak now (say the wake word)...")
    mono = _record_with_resample(device, args.seconds)

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

    # Probe the device: try SR first; on paInvalidSampleRate, fall back to
    # native rate. Same behaviour as MicCapture in the assistant.
    stream_rate = SR
    stream_block = BLOCK
    try:
        sd.check_input_settings(device=device, samplerate=SR, channels=1,
                                dtype="float32")
    except sd.PortAudioError as err:
        if not _is_invalid_sample_rate(err):
            raise
        info = sd.query_devices(device) if device is not None else sd.query_devices(kind="input")
        stream_rate = int(info.get("default_samplerate") or 48000) or 48000
        stream_block = int(stream_rate * BLOCK_MS / 1000)
        print(f"  ! device rejected {SR} Hz; opening at native {stream_rate} Hz.")

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
        with sd.InputStream(samplerate=stream_rate, blocksize=stream_block, channels=1,
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
    sub.add_parser("mics", help="show only INPUT-capable devices, with indices")

    s = sub.add_parser("set", help="rewrite audio.input_device in config.yaml")
    s.add_argument("device", help="index (e.g. 2) or name substring (e.g. Brio)")
    s.add_argument("--by", choices=("name", "index"), default="name",
                   help="write the device NAME (default; robust across reboots) or the numeric INDEX")
    s.add_argument("--dry-run", action="store_true",
                   help="show what would change without writing config.yaml")

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
    dispatch = {
        "list": cmd_list, "mics": cmd_mics, "set": cmd_set,
        "record": cmd_record, "meter": cmd_meter,
    }
    dispatch[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
