"""Rokoko mocap converter: a synthetic take end-to-end, plus the refusals
that keep bad recordings out of training."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mocap_dataset import convert, parse_bvh  # noqa: E402

SR = 24000
FPS = 60.0
CALIB = {"open": (4.0, 6.0), "fist": (7.0, 9.0), "wrist": (10.0, 14.0)}

_JOINTS = []
for side in ("Left", "Right"):
    for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for seg in ("1", "2"):
            _JOINTS.append(f"{side}Hand{f}{seg}")
_JOINTS += ["LeftHand", "RightHand"]


def _bvh(frames: np.ndarray, path):
    """Minimal BVH: flat chain, 3 rotation channels per joint."""
    lines = ["HIERARCHY", "ROOT Hips", "{", "\tOFFSET 0 0 0",
             "\tCHANNELS 3 Xrotation Yrotation Zrotation"]
    for j in _JOINTS:
        lines += [f"\tJOINT {j}", "\t{", "\t\tOFFSET 0 0 0",
                  "\t\tCHANNELS 3 Xrotation Yrotation Zrotation"]
    lines += ["\t\tEnd Site", "\t\t{", "\t\t\tOFFSET 0 0 1", "\t\t}"]
    lines += ["\t}"] * len(_JOINTS) + ["}"]
    lines += ["MOTION", f"Frames: {len(frames)}",
              f"Frame Time: {1.0/FPS:.6f}"]
    for row in frames:
        lines.append(" ".join(f"{v:.4f}" for v in row))
    path.write_text("\n".join(lines))


def _take(tmp_path, *, claps=True, fist_curl=60.0, dur=30.0):
    """Synthetic take following the capture protocol."""
    n = int(dur * FPS)
    n_ch = 3 * (1 + len(_JOINTS))            # root + joints
    fr = np.zeros((n, n_ch), dtype=np.float64)

    def ch(joint, k):                        # channel index of joint's k-th rot
        return 3 * (1 + _JOINTS.index(joint)) + k

    t = np.arange(n) / FPS
    if claps:                                # three sharp wiggles at 1/1.5/2 s
        for tc in (1.0, 1.5, 2.0):
            i = int(tc * FPS)
            for j in _JOINTS:
                fr[i:i+2, ch(j, 0)] += 25.0
    # fist segment: finger X channels curl; open segment stays at 0
    a, b = int(CALIB["fist"][0] * FPS), int(CALIB["fist"][1] * FPS)
    for j in _JOINTS:
        if "Hand" in j and j[-1] in "12":
            fr[a:b, ch(j, 0)] = fist_curl / 2.0   # 2 segs -> total = fist_curl
    # wrist roll segment: hands' Z channel sweeps
    a, b = int(CALIB["wrist"][0] * FPS), int(CALIB["wrist"][1] * FPS)
    sweep = 40.0 * np.sin(np.linspace(0, 4 * np.pi, b - a))
    fr[a:b, ch("LeftHand", 2)] = sweep
    fr[a:b, ch("RightHand", 2)] = -sweep
    # content: right index curls rhythmically, left stays open
    a = int(16.0 * FPS)
    wave = (fist_curl / 4.0) * (1 + np.sin(2 * np.pi * 1.5 * t[a:]))
    fr[a:, ch("RightHandIndex1", 0)] = wave

    bvh = tmp_path / "take.bvh"
    _bvh(fr, bvh)

    audio = 0.01 * np.random.randn(int(dur * SR)).astype(np.float32)
    if claps:
        for tc in (1.0, 1.5, 2.0):           # same instants: zero offset
            i = int(tc * SR)
            audio[i:i + 200] += 0.9
    audio[int(16 * SR):] += 0.05 * np.sin(
        2 * np.pi * 150 * t.repeat(int(SR / FPS))[:len(audio) - int(16 * SR)]
    ).astype(np.float32)
    import soundfile as sf
    wav = tmp_path / "take.wav"
    sf.write(wav, audio, SR)
    return str(bvh), str(wav)


def test_parse_bvh_roundtrip(tmp_path):
    fr = np.random.randn(10, 3 * (1 + len(_JOINTS)))
    _bvh(fr, tmp_path / "x.bvh")
    channels, frames, dt = parse_bvh(str(tmp_path / "x.bvh"))
    assert frames.shape == fr.shape
    assert abs(dt - 1.0 / FPS) < 1e-6
    assert "RightHandIndex1:Xrotation" in channels


def test_convert_full_take(tmp_path):
    bvh, wav = _take(tmp_path)
    feats, targets, meta = convert(bvh, wav, calib=CALIB)
    assert targets.shape[1] == 12
    assert len(feats) == len(targets)
    assert abs(meta["sync_offset_s"]) < 0.1, "claps recorded simultaneous"
    # right index (col 6) oscillates in content; left index (col 1) rests
    assert targets[:, 6].max() > 0.4, "content curl lost in conversion"
    assert targets[:, 6].std() > 0.05
    assert targets[:, 1].max() < 0.1, "resting finger contaminated"
    # closure is normalised: nothing outside [0, 1]
    assert targets[:, :10].min() >= 0.0 and targets[:, :10].max() <= 1.0
    assert np.abs(targets[:, 10:]).max() <= 1.0
    # the calibration sandwich itself must NOT be in the training data
    assert meta["content_s"] < 16.0, "calibration prologue leaked into shards"


def test_reject_take_without_claps(tmp_path):
    bvh, wav = _take(tmp_path, claps=False)
    with pytest.raises(ValueError, match="clap"):
        convert(bvh, wav, calib=CALIB)


def test_reject_dead_calibration(tmp_path):
    bvh, wav = _take(tmp_path, fist_curl=5.0)   # fingers barely move
    with pytest.raises(ValueError, match="calibration dead"):
        convert(bvh, wav, calib=CALIB)
