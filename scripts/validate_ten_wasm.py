"""Prove the WASM TEN VAD (zero/vad/ten_wasm.py) matches the NATIVE library
frame-for-frame from a common start state. Run on the DEV box (x86, where the
native lib runs, with libc++ installed). Both start fresh -> outputs should be
numerically identical.

    python scripts/validate_ten_wasm.py path/to/ten_vad.wasm
"""
import glob
import sys
import wave

import numpy as np

sys.path.insert(0, ".")
from zero.vad.ten_wasm import TenVadWasm  # noqa: E402

HOP = 256
WASM = sys.argv[1] if len(sys.argv) > 1 else "models/vad/ten_vad.wasm"


def load(p):
    with wave.open(p) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        ch = w.getnchannels()
    if ch == 2:
        a = a[::2]
    if sr != 16000:
        n = int(len(a) * 16000 / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n),
                      np.arange(len(a)), a.astype(np.float32)).astype(np.int16)
    return a


from ten_vad import TenVad  # native  # noqa: E402

worst = 0.0
wavs = sorted(glob.glob("*.wav"))[:3]
for wav in wavs:
    a = load(wav)
    native = TenVad(hop_size=HOP, threshold=0.5)   # fresh
    wasm = TenVadWasm(WASM, hop_size=HOP, threshold=0.5)  # fresh
    nf = len(a) // HOP
    dn, dw = [], []
    for i in range(nf):
        f = a[i * HOP:(i + 1) * HOP]
        dn.append(native.process(f)[0])
        dw.append(wasm.process(f)[0])
    dn, dw = np.array(dn), np.array(dw)
    d = float(np.abs(dn - dw).max())
    worst = max(worst, d)
    print(f"  {wav:20s} {nf} frames  max|native-wasm|={d:.2e}  "
          f"voiced native={int((dn>=0.5).sum())} wasm={int((dw>=0.5).sum())}")
print(f"WORST native-vs-wasm: {worst:.2e} -> "
      f"{'PASS (identical VAD)' if worst < 1e-3 else 'FAIL'}")
