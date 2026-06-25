"""Standalone speaker-verification test — run on your PC (or the Pi).

Build your voiceprint, then test that YOUR voice matches and OTHER voices don't.
Tune the threshold here before wiring it into the assistant.

Setup:
    pip install onnxruntime sounddevice numpy
    # download the model (~25 MB):
    #   models/voiceid/voxceleb_ECAPA512_LM.onnx
    #   from https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM

Use:
    python test_voiceid.py enroll     # record yourself 3x -> saves voiceprint.npy
    python test_voiceid.py verify      # record once -> prints match score
    python test_voiceid.py verify --loop   # keep testing (try other people!)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

from zero.voiceid.speaker import SpeakerVerifier

MODEL = "models/voiceid/voxceleb_ECAPA512_LM.onnx"
PROFILE = Path("voiceprint.npy")
SR = 16000
THRESHOLD = 0.45  # tuned: owner ~0.55-0.70, others ~0.25-0.35


def record(seconds: float, prompt: str) -> np.ndarray:
    input(f"\n{prompt}\n  -> press Enter, then speak for {seconds:.0f}s...")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    a = audio[:, 0]
    print(f"  captured (peak {np.abs(a).max():.2f})")
    return a


def enroll(v: SpeakerVerifier) -> None:
    embs = []
    for i in range(3):
        a = record(4, f"Enrollment {i + 1}/3 — say a full sentence in your normal voice")
        e = v.embed(a)
        if e is None:
            print("  too short / silent, try again")
            continue
        embs.append(e)
    if not embs:
        print("enrollment failed — no usable audio")
        return
    profile = np.mean(embs, axis=0)
    profile /= np.linalg.norm(profile) + 1e-9
    np.save(PROFILE, profile)
    print(f"\nVoiceprint saved -> {PROFILE}  (from {len(embs)} clips)")


def verify(v: SpeakerVerifier, loop: bool) -> None:
    if not PROFILE.exists():
        print("No voiceprint yet — run:  python test_voiceid.py enroll")
        return
    profile = np.load(PROFILE)
    while True:
        a = record(4, "Verify — say something (try your voice, then someone else's)")
        score, is_owner = v.verify(profile, a)
        verdict = "MATCH (it's you)" if is_owner else "REJECT (not you)"
        print(f"  score = {score:.3f}   threshold = {v.threshold:.2f}   -> {verdict}")
        if not loop:
            break


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    loop = "--loop" in sys.argv
    v = SpeakerVerifier(MODEL, threshold=THRESHOLD)
    if cmd == "enroll":
        enroll(v)
    else:
        verify(v, loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
