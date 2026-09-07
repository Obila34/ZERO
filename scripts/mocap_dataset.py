#!/usr/bin/env python3
"""Build gesture-training shards from OUR OWN mocap (Rokoko suit + gloves).

Input: one take = a BVH export from Rokoko Studio (Smartsuit + Smartgloves,
finger joints included) plus the WAV recorded alongside it.

    python scripts/mocap_dataset.py take1.bvh take1.wav --out data/gesture_shards

Protocol (docs/MOCAP_CAPTURE.md) — every take begins with:
    0-3 s    three sharp CLAPS (sync: visible in audio AND motion at once)
    ~4-6 s   hands fully OPEN, held still
    ~7-9 s   both fists fully CLOSED, held still
    ~10-14 s wrists rolled end-to-end, a few times
    then     the actual content (talking / signing)

The calibration sandwich is what makes the rig's conventions irrelevant:
for each finger we pick, per joint, the rotation channel that moved most
between OPEN and FIST, and normalise this person's curl into the robot's
0..1 closure scale. Wrists likewise from the roll segment. No fixed joint
axis assumptions — Rokoko's skeleton can change under us and this still
holds, because the person's own extremes define the scale.

Output: shards/<take>.npz with
    feats   (N, FEAT_DIM)  audio features at 20 Hz (zero/expr/features.py)
    targets (N, 12)        [L thumb..pinky, R thumb..pinky, Lwrist/45,
                            Rwrist/45] — byte-compatible with the BEAT2
                            shards; scripts/neural_train.py eats them as-is.

The converter REFUSES bad takes loudly (no claps found, dead calibration,
audio/motion drift) — a rejected take costs minutes to re-record; silent
bad data poisons a training run.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from zero.expr.features import FRAME_HZ, extract  # noqa: E402

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


# ── BVH parsing (stdlib only — the format is plain text) ────────────────────

def parse_bvh(path: str):
    """-> (channel_names ['JointName:Xrotation', ...], frames (T, C), dt)."""
    text = open(path).read()
    if "MOTION" not in text:
        raise ValueError("not a BVH file (no MOTION section)")
    hier, motion = text.split("MOTION", 1)
    channels: list[str] = []
    joint = None
    for line in hier.splitlines():
        line = line.strip()
        m = re.match(r"(?:ROOT|JOINT)\s+(\S+)", line)
        if m:
            joint = m.group(1)
        elif line.startswith("CHANNELS"):
            parts = line.split()
            for ch in parts[2:]:
                channels.append(f"{joint}:{ch}")
    lines = [ln for ln in motion.splitlines() if ln.strip()]
    n_frames = dt = None
    data_start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Frames:"):
            n_frames = int(ln.split(":")[1])
        elif ln.strip().startswith("Frame Time:"):
            dt = float(ln.split(":")[1])
            data_start = i + 1
            break
    if not n_frames or not dt:
        raise ValueError("BVH missing Frames/Frame Time")
    rows = []
    for ln in lines[data_start:data_start + n_frames]:
        rows.append(np.fromstring(ln, sep=" ") if hasattr(np, "fromstring")
                    else np.array(ln.split(), dtype=np.float64))
    frames = np.asarray(rows, dtype=np.float64)
    if frames.shape[1] != len(channels):
        raise ValueError(f"channel mismatch: hierarchy declares "
                         f"{len(channels)}, motion rows have "
                         f"{frames.shape[1]}")
    return channels, frames, dt


def _side_of(name: str) -> str | None:
    n = name.lower()
    if "left" in n or re.match(r"^l(hand|arm|_)", n):
        return "left"
    if "right" in n or re.match(r"^r(hand|arm|_)", n):
        return "right"
    return None


def finger_channel_idx(channels: list[str]) -> dict[tuple, list[int]]:
    """(side, finger) -> indices of that finger's ROTATION channels."""
    out: dict[tuple, list[int]] = {}
    for i, ch in enumerate(channels):
        joint, kind = ch.split(":")
        if "rotation" not in kind.lower():
            continue
        side = _side_of(joint)
        if side is None:
            continue
        for f in FINGERS:
            if f in joint.lower():
                out.setdefault((side, f), []).append(i)
    return out


def wrist_channel_idx(channels: list[str]) -> dict[str, list[int]]:
    """side -> rotation-channel indices of the hand/wrist joint itself."""
    out: dict[str, list[int]] = {}
    for i, ch in enumerate(channels):
        joint, kind = ch.split(":")
        if "rotation" not in kind.lower():
            continue
        jl = joint.lower()
        if not re.search(r"(hand|wrist)$", jl):
            continue
        side = _side_of(joint)
        if side:
            out.setdefault(side, []).append(i)
    return out


# ── sync + calibration ──────────────────────────────────────────────────────

def find_claps_audio(audio: np.ndarray, sr: int, window_s: float = 15.0):
    """Times of the sharpest onsets in the opening window."""
    n = min(len(audio), int(window_s * sr))
    x = np.abs(audio[:n])
    hop = max(1, sr // 200)
    env = x[: (n // hop) * hop].reshape(-1, hop).max(axis=1)
    thr = env.max() * 0.5
    times, last = [], -1.0
    for i, v in enumerate(env):
        t = i * hop / sr
        if v >= thr and t - last > 0.15:
            times.append(t)
            last = t
    return times


def find_claps_motion(frames: np.ndarray, idx: list[int], dt: float,
                      window_s: float = 3.5):
    """Times of peak hand angular speed in the opening window — a clap is
    the fastest thing hands do in the first seconds of a take. The window
    ends BEFORE the calibration sandwich (protocol: claps at 0-3 s), or
    the fist transition gets mistaken for a clap."""
    n = min(len(frames), int(window_s / dt))
    speed = np.abs(np.diff(frames[:n][:, idx], axis=0)).sum(axis=1)
    thr = speed.max() * 0.5
    times, last = [], -1.0
    for i, v in enumerate(speed):
        t = i * dt
        if v >= thr and t - last > 0.15:
            times.append(t)
            last = t
    return times


def _segment(frames: np.ndarray, dt: float, t0: float, t1: float):
    a, b = int(t0 / dt), int(t1 / dt)
    if b <= a or b > len(frames):
        raise ValueError(f"calibration segment {t0}-{t1}s outside take")
    return frames[a:b]


# ── the converter ───────────────────────────────────────────────────────────

def convert(bvh_path: str, wav_path: str, *, calib: dict[str, tuple],
            min_curl_range_deg: float = 20.0,
            min_wrist_range_deg: float = 30.0):
    import soundfile as sf

    channels, frames, dt = parse_bvh(bvh_path)
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    fidx = finger_channel_idx(channels)
    widx = wrist_channel_idx(channels)
    missing = [k for s in ("left", "right") for k in
               [(s, f) for f in FINGERS] if k not in fidx]
    if missing:
        raise ValueError(f"REJECT: finger joints absent from BVH (gloves "
                         f"not exported?): {missing[:4]}...")
    if set(widx) != {"left", "right"}:
        raise ValueError("REJECT: hand/wrist joints not found in BVH")

    # sync: claps seen by both instruments
    hand_rot_idx = [i for lst in fidx.values() for i in lst] + \
                   [i for lst in widx.values() for i in lst]
    a_claps = find_claps_audio(audio, sr)
    m_claps = find_claps_motion(frames, hand_rot_idx, dt)
    if len(a_claps) < 2 or len(m_claps) < 2:
        raise ValueError(f"REJECT: claps not found (audio {len(a_claps)}, "
                         f"motion {len(m_claps)}) — re-record with three "
                         "sharp claps in the first seconds")
    offset = a_claps[0] - m_claps[0]        # motion time + offset = audio time
    if len(a_claps) >= 2 and len(m_claps) >= 2:
        drift = abs((a_claps[-1] - a_claps[0]) - (m_claps[-1] - m_claps[0]))
        if drift > 0.08:
            raise ValueError(f"REJECT: clap spacing disagrees by "
                             f"{drift*1000:.0f} ms — clocks drifting")

    # calibration: per finger, pick each joint's most-moved rotation channel
    # between OPEN and FIST, in that channel's own sign
    open_seg = _segment(frames, dt, *calib["open"])
    fist_seg = _segment(frames, dt, *calib["fist"])
    roll_seg = _segment(frames, dt, *calib["wrist"])

    curl_of: dict[tuple, tuple] = {}
    for key, idxs in fidx.items():
        deltas = fist_seg[:, idxs].mean(0) - open_seg[:, idxs].mean(0)
        curl_open = float(np.sum(np.abs(deltas) * 0))          # 0 by def
        curl_fist = float(np.sum(np.abs(deltas)))
        if curl_fist < min_curl_range_deg:
            raise ValueError(f"REJECT: {key} moved only {curl_fist:.0f} deg "
                             "between open and fist — calibration dead")
        signs = np.sign(deltas)
        base = open_seg[:, idxs].mean(0)
        curl_of[key] = (np.array(idxs), signs, base, curl_fist, curl_open)

    wrist_of: dict[str, tuple] = {}
    for side, idxs in widx.items():
        rng = roll_seg[:, idxs].max(0) - roll_seg[:, idxs].min(0)
        k = int(np.argmax(rng))
        if rng[k] < min_wrist_range_deg:
            raise ValueError(f"REJECT: {side} wrist rolled only "
                             f"{rng[k]:.0f} deg in calibration")
        ch = idxs[k]
        mid = float(roll_seg[:, ch].mean())
        half = float(rng[k]) / 2.0
        wrist_of[side] = (ch, mid, half)

    # per-frame targets on the mocap clock
    T = len(frames)
    tgt = np.zeros((T, 12), dtype=np.float32)
    col = 0
    for side in ("left", "right"):
        for f in FINGERS:
            idxs, signs, base, span, _ = curl_of[(side, f)]
            curl = ((frames[:, idxs] - base) * signs).sum(axis=1)
            tgt[:, col] = np.clip(curl / span, 0.0, 1.0)
            col += 1
    for j, side in enumerate(("left", "right")):
        ch, mid, half = wrist_of[side]
        # person's own roll extreme maps to the model's 45-deg unit
        tgt[:, 10 + j] = np.clip((frames[:, ch] - mid) / half, -1, 1)

    # resample onto the 20 Hz audio-feature clock, in AUDIO time
    feats = extract(audio, sr)
    n = len(feats)
    t_audio = np.arange(n) / FRAME_HZ
    t_motion = np.clip(((t_audio - offset) / dt).round().astype(int),
                       0, T - 1)
    targets = tgt[t_motion]
    # drop the calibration prologue — the model must not learn fists
    # from the sandwich
    content_start = int((calib["wrist"][1] + offset + 1.0) * FRAME_HZ)
    feats = feats[content_start:]
    targets = targets[content_start:]
    if len(feats) < 200:                     # < 10 s of actual content
        raise ValueError("REJECT: under 10 s of content after calibration")
    return feats, targets, {"sync_offset_s": round(offset, 3),
                            "content_s": round(len(feats) / FRAME_HZ, 1)}


def _parse_calib(s: str) -> dict[str, tuple]:
    out = {}
    for part in s.split(","):
        name, rng = part.split(":")
        a, b = rng.split("-")
        out[name.strip()] = (float(a), float(b))
    for k in ("open", "fist", "wrist"):
        if k not in out:
            raise argparse.ArgumentTypeError(f"calib missing '{k}'")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh")
    ap.add_argument("wav")
    ap.add_argument("--out", default="data/gesture_shards")
    ap.add_argument("--calib", type=_parse_calib,
                    default=_parse_calib("open:4-6,fist:7-9,wrist:10-14"),
                    help="calibration segments in MOTION seconds, e.g. "
                         "'open:4-6,fist:7-9,wrist:10-14'")
    a = ap.parse_args()
    try:
        feats, targets, meta = convert(a.bvh, a.wav, calib=a.calib)
    except ValueError as e:
        print(e)
        return 1
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.bvh))[0]
    out = os.path.join(a.out, f"mocap_{stem}.npz")
    np.savez_compressed(out, feats=feats, targets=targets)
    print(f"ok {stem}: {meta['content_s']} s of content, sync offset "
          f"{meta['sync_offset_s']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
