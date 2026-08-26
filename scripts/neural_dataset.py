#!/usr/bin/env python3
"""Build gesture-training shards from BEAT2-format clips (Phase E, N2).

Input: a directory of paired files —
    <name>.wav            speech audio (any sample rate, mono)
    <name>.smplx.npz      with arrays:
        left_hand_pose  (T, 45)   axis-angle, SMPL-X/MANO order
        right_hand_pose (T, 45)
        [optional] left_wrist (T, 3), right_wrist (T, 3)
        mocap_frame_rate  scalar
Output: shards/<name>.npz with
    feats   (N, FEAT_DIM)  audio features at 20 Hz (zero/expr/features.py)
    targets (N, 12)        closure/wrist targets (zero/expr/retarget.py)
aligned by resampling mocap to the 20 Hz feature clock.

Fetching BEAT2: https://pantomatrix.github.io/BEAT/ (BEAT2/SMPL-X release,
speaker folders contain wav + smplx npz). Pull a few speakers' folders —
the texture model needs hours, not the full corpus. This script never
downloads anything itself.

    python scripts/neural_dataset.py <clips_dir> --out data/gesture_shards
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from zero.expr.features import FRAME_HZ, extract  # noqa: E402
from zero.expr.retarget import sequence_to_targets  # noqa: E402


def _resample_rows(arr: np.ndarray, src_hz: float, n_out: int) -> np.ndarray:
    """Nearest-frame resample of (T, ...) mocap rows onto the 20 Hz clock."""
    idx = np.clip((np.arange(n_out) / FRAME_HZ * src_hz).round().astype(int),
                  0, len(arr) - 1)
    return arr[idx]


def build_clip(wav_path: Path, npz_path: Path):
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    feats = extract(audio, sr)
    if len(feats) < 40:                    # < 2 s: not worth a shard
        return None
    m = np.load(npz_path)
    fps = float(m["mocap_frame_rate"]) if "mocap_frame_rate" in m else 30.0
    lh = np.asarray(m["left_hand_pose"], dtype=np.float32).reshape(-1, 45)
    rh = np.asarray(m["right_hand_pose"], dtype=np.float32).reshape(-1, 45)
    lw = (np.asarray(m["left_wrist"], dtype=np.float32).reshape(-1, 3)
          if "left_wrist" in m else None)
    rw = (np.asarray(m["right_wrist"], dtype=np.float32).reshape(-1, 3)
          if "right_wrist" in m else None)
    n = min(len(feats), int(len(lh) / fps * FRAME_HZ))
    feats = feats[:n]
    lh = _resample_rows(lh, fps, n).reshape(n, 15, 3)
    rh = _resample_rows(rh, fps, n).reshape(n, 15, 3)
    lw = _resample_rows(lw, fps, n) if lw is not None else None
    rw = _resample_rows(rw, fps, n) if rw is not None else None
    targets = sequence_to_targets(lh, rh, lw, rw)
    return feats, targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir")
    ap.add_argument("--out", default="data/gesture_shards")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(Path(a.clips_dir).rglob("*.wav"))
    built = skipped = 0
    for wav in wavs:
        npz = wav.with_suffix("").with_suffix(".smplx.npz") \
            if wav.suffix == ".wav" else None
        npz = Path(str(wav)[:-4] + ".smplx.npz")
        if not npz.exists():
            skipped += 1
            continue
        try:
            res = build_clip(wav, npz)
        except Exception as e:
            print(f"  SKIP {wav.name}: {e}")
            skipped += 1
            continue
        if res is None:
            skipped += 1
            continue
        feats, targets = res
        np.savez_compressed(out / f"{wav.stem}.npz",
                            feats=feats, targets=targets)
        built += 1
        print(f"  ok {wav.stem}: {len(feats)} frames "
              f"({len(feats)/FRAME_HZ:.0f} s)")
    print(f"\n{built} shard(s) built, {skipped} skipped -> {out}")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
