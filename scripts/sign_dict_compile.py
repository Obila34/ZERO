#!/usr/bin/env python3
"""Compile the PerceptX ASL sign dictionary into ZERO joint trajectories.

Input: the model vault's sign_motion.npz — glosses (N,) + frames
(N, 48, 144) float16, each frame [arm 6 pts ×3 | left hand 21×3 |
right hand 21×3] MediaPipe landmarks, unmirrored camera.

Output: data/sign_dictionary.npz —
    glosses  (N,) str
    closures (N, T, 10)  [L thumb..pinky, R thumb..pinky] 0..1
    wrists   (N, T, 2)   palm-facing proxy 0..1 (0=forward, 1=in)
    arms     (N, T, 6)   degrees for [R elbow, R up_down, R in_out,
                          L elbow, L up_down, L in_out], CONSERVATIVELY
                          capped — same order of magnitude as the
                          signing stance, never the raw human excursion
    present  (N, T, 2)   per-frame hand visibility (L, R)

Closure calibration is GLOBAL, from the data itself: per finger, the
5th/95th percentile of MCP+PIP flexion across every visible frame in the
dictionary defines open/closed — one person's rig, one consistent scale.

    python scripts/sign_dict_compile.py --vault ~/perceptx-model-vault \
        --out data/sign_dictionary.npz
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

FINGER_JOINTS = {  # finger -> (MCP, PIP, TIP) landmark indices
    "thumb": (1, 2, 4), "index": (5, 6, 8), "middle": (9, 10, 12),
    "ring": (13, 14, 16), "pinky": (17, 18, 20),
}
ARM = 18
HAND = 63

# Arm caps (deg): dictionary signs stay inside the envelope the operator
# already accepted for the signing stance — the raw ASL recordings can
# reach far wider than these steppers should ever be asked to go.
ELBOW_CAP = 35.0
RAISE_CAP = 45.0
INOUT_CAP = 40.0


def _angle(a, b, c):
    ba, bc = a - b, c - b
    na = np.linalg.norm(ba, axis=-1)
    nc = np.linalg.norm(bc, axis=-1)
    d = (ba * bc).sum(-1) / np.maximum(na * nc, 1e-6)
    return np.arccos(np.clip(d, -1.0, 1.0))


def _hand_flexions(hand):        # (..., 21, 3) -> (..., 5) MCP+PIP flexion sum
    out = []
    for mcp, pip, tip in FINGER_JOINTS.values():
        f1 = np.pi - _angle(hand[..., 0, :], hand[..., mcp, :],
                            hand[..., pip, :])
        f2 = np.pi - _angle(hand[..., mcp, :], hand[..., pip, :],
                            hand[..., tip, :])
        out.append(f1 + f2)
    return np.stack(out, axis=-1)


def _palm_facing(hand):          # (..., 21, 3) -> (...,) |z| of palm normal
    n = np.cross(hand[..., 5, :] - hand[..., 0, :],
                 hand[..., 17, :] - hand[..., 0, :])
    return np.abs(n[..., 2]) / np.maximum(np.linalg.norm(n, axis=-1), 1e-6)


def compile_dictionary(vault: str):
    d = np.load(os.path.join(vault, "generation", "sign-dictionary",
                             "sign_motion.npz"))
    glosses = np.asarray(d["glosses"])
    fr = np.asarray(d["frames"], dtype=np.float32)      # (N, T, 144)
    N, T, _ = fr.shape
    arm_pts = fr[:, :, :ARM].reshape(N, T, 6, 3)
    # HANDEDNESS (verified on the data 2026-09-07): the recordings are
    # unmirrored-camera, and the signer's DOMINANT hand lands in the
    # "left" block ("hello"/"please"/"yes" are left-block-only there).
    # The robot signs right-dominant, so the data's left chain drives
    # the robot's RIGHT side and vice versa.
    rh = fr[:, :, ARM:ARM + HAND].reshape(N, T, 21, 3)      # data-left
    lh = fr[:, :, ARM + HAND:].reshape(N, T, 21, 3)         # data-right

    present = np.stack([np.abs(lh).sum((-1, -2)) > 0,
                        np.abs(rh).sum((-1, -2)) > 0], axis=-1)   # (N,T,2)

    # global open/closed flexion references, from VISIBLE frames only
    flex_l = _hand_flexions(lh)
    flex_r = _hand_flexions(rh)
    both = np.concatenate([flex_l[present[..., 0]],
                           flex_r[present[..., 1]]], axis=0)      # (M,5)
    lo = np.percentile(both, 5, axis=0)
    hi = np.percentile(both, 95, axis=0)
    span = np.maximum(hi - lo, 1e-3)

    def _closure(flex, pres):
        c = np.clip((flex - lo) / span, 0.0, 1.0)
        c[~pres] = 0.0                                   # missing hand = open
        return c

    closures = np.concatenate([_closure(flex_l, present[..., 0]),
                               _closure(flex_r, present[..., 1])],
                              axis=-1).astype(np.float16)         # (N,T,10)

    wr = np.stack([1.0 - _palm_facing(lh), 1.0 - _palm_facing(rh)],
                  axis=-1)
    wr[~present] = 0.0
    wrists = wr.astype(np.float16)                                # (N,T,2)

    # arms: shoulders(0,1) elbows(2,3) wrists(4,5); mediapipe y grows DOWN
    def _arm(sh, el, wrp, pres):
        upper = el - sh
        fore = wrp - el
        elbow = (np.pi - _angle(sh, el, wrp)) / np.pi              # 0 straight
        sh_w = np.maximum(np.linalg.norm(
            arm_pts[:, :, 0] - arm_pts[:, :, 1], axis=-1), 1e-3)
        raise_amt = np.clip((sh[..., 1] - wrp[..., 1]) / sh_w, -0.3, 1.5)
        inout_amt = np.clip(np.abs(wrp[..., 0] - sh[..., 0]) / sh_w
                            - 0.4, 0.0, 1.5)
        elbow_deg = -np.clip(elbow, 0, 1) * ELBOW_CAP              # bend fwd
        raise_deg = np.clip(raise_amt / 1.5, 0, 1) * RAISE_CAP
        inout_deg = np.clip(inout_amt / 1.5, 0, 1) * INOUT_CAP
        out = np.stack([elbow_deg, raise_deg, inout_deg], axis=-1)
        out[~pres] = 0.0
        return out

    # arm points follow the same swap: data-left chain (pose 11/13/15 =
    # indices 0/2/4) -> robot RIGHT arm
    right = _arm(arm_pts[:, :, 0], arm_pts[:, :, 2], arm_pts[:, :, 4],
                 present[..., 1])
    left = _arm(arm_pts[:, :, 1], arm_pts[:, :, 3], arm_pts[:, :, 5],
                present[..., 0])
    # sign conventions (hardware-verified via the stance): raise = +R/-L
    # up_down, outward = +R/-L in_out, elbow bend forward = negative both
    arms = np.concatenate([
        right * np.array([1.0, 1.0, 1.0]),
        left * np.array([1.0, -1.0, -1.0]),
    ], axis=-1).astype(np.float16)                                 # (N,T,6)

    return {"glosses": glosses, "closures": closures, "wrists": wrists,
            "arms": arms, "present": present}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=os.path.expanduser(
        "~/perceptx-model-vault"))
    ap.add_argument("--out", default="data/sign_dictionary.npz")
    a = ap.parse_args()
    out = compile_dictionary(a.vault)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **out)
    n = len(out["glosses"])
    mb = os.path.getsize(a.out) / 1e6
    print(f"{n} signs compiled -> {a.out} ({mb:.1f} MB)")
    print(f"arm range check: elbow [{out['arms'][..., 0].min():.1f}, "
          f"{out['arms'][..., 0].max():.1f}] raise "
          f"[{out['arms'][..., 1].min():.1f}, "
          f"{out['arms'][..., 1].max():.1f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
