#!/usr/bin/env python3
"""Text-to-speech tester — type text, hear it on your output device.

Uses the SAME engine the robot uses (TTS engine + orchestrator + Speaker, all
driven by config.yaml), so a success here means your output device and voice
are correctly configured.

    python scripts/test_tts.py                       # interactive: type lines
    python scripts/test_tts.py --text "Hello there"  # one-shot
    python scripts/test_tts.py --output 4            # override output device index

Try the cue tags the LLM uses, e.g.:
    Oh wow [chuckles], that's amazing!
    Hmm [pause] let me think about that [sighs].
(With Piper these splice in clips from zero/tts/nonverbals/; with Fish they are
performed natively.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/test_tts.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zero.audio.playback import Speaker          # noqa: E402
from zero.config import load_config              # noqa: E402
from zero.factory import build_voice             # noqa: E402
from zero.utils.logging import setup_logging     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Type text, hear it spoken.")
    ap.add_argument("--text", help="speak this once and exit (skip interactive mode)")
    ap.add_argument("--output", type=int, default=None,
                    help="override audio output device index for this test")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    # Build the real voice pipeline (engine chosen by config: piper / fish).
    try:
        voice = build_voice(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR building TTS engine: {e}")
        print("If this mentions 'piper': install the piper binary and a voice model")
        print("(see scripts/setup_pi.sh), or set tts.piper.binary / tts.piper.voice.")
        return 1

    out_device = args.output if args.output is not None else cfg.get("audio.output_device")
    speaker = Speaker(device=out_device)
    print(f"TTS engine : {cfg.get('tts.engine')}")
    print(f"Output dev : {out_device if out_device is not None else 'system default'}")
    print(f"Voice rate : {voice.sample_rate} Hz\n")

    def speak(text: str) -> None:
        text = text.strip()
        if not text:
            return
        audio = voice.synthesize(text)
        if audio.size == 0:
            print("  (no audio produced — check the engine logs above)")
            return
        print(f"  speaking {audio.size / voice.sample_rate:.1f}s...")
        speaker.play(audio, voice.sample_rate)

    if args.text:
        speak(args.text)
        return 0

    print("Type text and press Enter to hear it. Empty line or Ctrl-C to quit.\n")
    try:
        while True:
            line = input("say> ")
            if not line.strip():
                break
            speak(line)
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
