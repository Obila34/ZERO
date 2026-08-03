"""Prove zero/vad/smart_turn.py's numpy log-mel matches the real Whisper
feature extractor Smart Turn v3 was trained with. Run on the DEV box (which can
have transformers); the Pi ships only the validated numpy path.

    python scripts/validate_smart_turn_features.py

Passes if max abs diff over several random + real-ish clips is < 1e-3, and the
end-to-end ONNX completion probabilities agree to < 1e-3.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from zero.vad.smart_turn import log_mel  # noqa: E402


def reference(audio):
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(chunk_length=8, feature_size=80)
    out = fe(audio, sampling_rate=16000, padding="max_length",
             max_length=8 * 16000, truncation=True, do_normalize=True,
             return_tensors="np")
    return out.input_features.squeeze(0).astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    clips = {
        "1s noise": rng.standard_normal(16000).astype(np.float32) * 0.1,
        "3s noise": rng.standard_normal(48000).astype(np.float32) * 0.1,
        "0.5s tone": (np.sin(2 * np.pi * 180 * np.arange(8000) / 16000)
                      ).astype(np.float32) * 0.3,
        "8s sweep": (np.sin(2 * np.pi * np.cumsum(
            np.linspace(80, 300, 128000)) / 16000)).astype(np.float32) * 0.2,
    }
    worst = 0.0
    for name, a in clips.items():
        mine, ref = log_mel(a), reference(a)
        d = float(np.abs(mine - ref).max())
        worst = max(worst, d)
        print(f"  {name:10s} shape {mine.shape} vs {ref.shape}  max|Δ|={d:.2e}")
    print(f"WORST feature diff: {worst:.2e}  -> {'PASS' if worst < 1e-3 else 'FAIL'}")

    # End-to-end: do the probabilities agree?
    try:
        import onnxruntime as ort
        m = "models/turn/smart-turn-v3.2-cpu.onnx"
        s = ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0].name
        sig = lambda x: 1 / (1 + np.exp(-x))
        wd = 0.0
        for name, a in clips.items():
            pm = sig(float(np.ravel(s.run(None, {inp: log_mel(a)[None]})[0])[0]))
            pr = sig(float(np.ravel(s.run(None, {inp: reference(a)[None]})[0])[0]))
            wd = max(wd, abs(pm - pr))
            print(f"  {name:10s} P_mine={pm:.4f} P_ref={pr:.4f}")
        print(f"WORST prob diff: {wd:.2e} -> {'PASS' if wd < 1e-3 else 'FAIL'}")
    except FileNotFoundError:
        print("(model not present here — feature check only)")


if __name__ == "__main__":
    main()
