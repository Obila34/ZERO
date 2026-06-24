#!/usr/bin/env python3
"""Live wake-word tester — see the mic level AND the wake score in real time.

    python scripts/test_wake.py

It uses the SAME mic device and wake engine as the app (from config.yaml), then
prints a continuously-updating line:

    mic [########----] 0.21   wake [##----------] 0.13

  * mic  = input level. If this stays near 0.00 when you talk, the MIC is the
           problem (wrong device / muted / too quiet) — fix that first.
  * wake = wake-word confidence. Say the wake word; if this jumps toward 1.00
           you're good. The default trigger threshold is wake.threshold in config.

Ctrl-C to quit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zero.audio.capture import MicCapture       # noqa: E402
from zero.config import load_config             # noqa: E402
from zero.factory import build_wake             # noqa: E402
from zero.utils.logging import setup_logging    # noqa: E402


def _meter(value: float, width: int = 12) -> str:
    n = max(0, min(width, int(value * width)))
    return "#" * n + "-" * (width - n)


def main() -> int:
    setup_logging("WARNING")  # quiet the engine logs; we want the live line
    cfg = load_config()

    device = cfg.get("audio.input_device")
    threshold = cfg.get("wake.threshold", 0.5)
    mic = MicCapture(
        sample_rate=cfg.get("audio.sample_rate", 16000),
        block_ms=cfg.get("audio.block_ms", 30),
        device=device,
    )
    wake = build_wake(cfg)

    print(f"\nmic device : {device if device is not None else 'system default'}")
    print(f"wake model : {cfg.get('wake.model')}  (threshold {threshold:.2f})")
    print("Talk to check the mic, then say the wake word. Ctrl-C to quit.\n")

    mic.start()
    peak_seen = 0.0
    try:
        for frame in mic.frames():
            level = float(np.max(np.abs(frame))) / 32768.0
            peak_seen = max(peak_seen, level)
            sc = wake.score(frame)
            hit = "  <== DETECTED!" if sc >= threshold else ""
            print(
                f"\rmic [{_meter(level)}] {level:0.2f}   "
                f"wake [{_meter(sc)}] {sc:0.2f}{hit}   ",
                end="", flush=True,
            )
    except KeyboardInterrupt:
        print(f"\n\nPeak mic level seen: {peak_seen:.2f}")
        if peak_seen < 0.02:
            print("-> Mic looks SILENT. Fix the input device before the wake word.")
        else:
            print("-> Mic is receiving audio.")
    finally:
        mic.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
