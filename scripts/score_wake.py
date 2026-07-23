"""Score a WAV file against the configured wake-word model, offline.

Closes the diagnostic loop with test_microphone.py: record what the pipeline
would hear, then check what the wake model thinks of it — without running the
whole assistant.

    python test_microphone.py record --device pipewire --gain 6 --save probe.wav
    python scripts/score_wake.py probe.wav

Reads wake.model / wake.threshold from config.yaml (override with --model /
--threshold). A top score >= threshold means this exact audio WOULD wake ZERO —
so if the live app still doesn't fire on the same mic, the fault is in the live
capture stream, not the model or the audio.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHUNK = 1280  # openWakeWord's 80 ms @ 16 kHz window — same as the live engine


def load_wav(path: str) -> np.ndarray:
    w = wave.open(path)
    if w.getsampwidth() != 2:
        sys.exit(f"  ! {path}: need 16-bit PCM (got sampwidth {w.getsampwidth()})")
    data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if w.getnchannels() > 1:
        data = data.reshape(-1, w.getnchannels())[:, 0]
    rate = w.getframerate()
    if rate != 16000:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(16000, rate)
        data = resample_poly(data.astype(np.float64), 16000 // g, rate // g)
        data = np.clip(data, -32768, 32767).astype(np.int16)
        print(f"  (resampled {rate} Hz -> 16000 Hz)")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="score a wav against the wake model")
    ap.add_argument("wav", help="16 kHz mono 16-bit WAV (e.g. from test_microphone.py record --save)")
    ap.add_argument("--model", default=None, help="wake model name/path (default: config.yaml wake.model)")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    from zero.config import load_config

    cfg = load_config()
    model_name = args.model or cfg.get("wake.model", "hey_jarvis")
    threshold = args.threshold if args.threshold is not None else cfg.get("wake.threshold", 0.5)

    from openwakeword.model import Model

    model = Model(wakeword_models=[model_name], inference_framework="onnx")
    data = load_wav(args.wav)
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    peak = int(np.max(np.abs(data))) if data.size else 0
    print(f"  {args.wav}: {len(data) / 16000:.1f}s  rms={rms:.0f}  peak={peak}")

    top, top_t = 0.0, 0.0
    for i in range(0, len(data) - CHUNK, CHUNK):
        scores = model.predict(data[i:i + CHUNK])
        s = max(scores.values()) if scores else 0.0
        if s > top:
            top, top_t = s, i / 16000
    fired = top >= threshold
    print(f"  top {model_name} score: {top:.3f} at t={top_t:.1f}s  "
          f"(threshold {threshold:.2f}) -> {'WOULD WAKE' if fired else 'would NOT wake'}")
    if not fired and top >= 0.5 * threshold:
        print("  near miss — try speaking closer/clearer, raising audio.input_gain, "
              "or lowering wake.threshold a notch.")
    elif not fired and rms < 50:
        print("  audio is near-silent — wrong input device or muted capture. "
              "Run: python test_microphone.py mics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
