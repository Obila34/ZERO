"""SMPL-X hands → ZERO's 12-DoF closure space (Phase E, N2).

BEAT2 (and every research gesture dataset) represents hands as SMPL-X /
MANO joint rotations: 15 joints per hand, 45 axis-angle DoF. ZERO's hand
has ONE flexion servo per finger and one wrist pitch. This module is the
fixed analytic bridge: per finger, closure = the summed flexion of its
three joints, normalized against a fully-curled reference; wrist pitch
from the wrist's flexion component.

DELIBERATELY approximate — this feeds the neural TEXTURE model, whose job
is rhythm and liveliness, not sign accuracy (meaning stays rule-based;
see docs/NEURAL_GESTURES_PLAN.md). The constants below say exactly what
was assumed, so they can be revisited when the first retargeted clips are
eyeballed next to the source mocap.

SMPL-X hand joint layout (per hand, after the 3 body-wrist DoF):
    index1..3, middle1..3, pinky1..3, ring1..3, thumb1..3
Flexion for the fingers is dominantly the local +x rotation in MANO's
frame; the thumb's axis is oblique, so its closure uses rotation
magnitude instead — cruder, and acceptable for texture.
"""
from __future__ import annotations

import numpy as np

# SMPL-X orders hand joints: index, middle, pinky, ring, thumb (MANO order)
_MANO_ORDER = ("index", "middle", "pinky", "ring", "thumb")
# Summed flexion (radians over 3 joints) that counts as fully curled.
FULL_CURL_RAD = 4.2
# Wrist pitch mapping: SMPL-X wrist flexion (rad) -> ZERO wrist degrees
# around its rest orientation; clamped by the hand model's envelope later.
WRIST_GAIN_DEG_PER_RAD = 40.0


def _axis_angle_flexion(aa: np.ndarray) -> np.ndarray:
    """Per-joint flexion estimate from axis-angle (…, 3): the x component
    for fingers (MANO's curl axis), full magnitude as fallback."""
    return np.asarray(aa)[..., 0]


def hand_pose_to_closure(hand_aa: np.ndarray) -> dict[str, float]:
    """One hand's 45 axis-angle values (15, 3) -> {finger: closure 0..1}."""
    aa = np.asarray(hand_aa, dtype=np.float32).reshape(15, 3)
    out: dict[str, float] = {}
    for i, finger in enumerate(_MANO_ORDER):
        joints = aa[3 * i: 3 * i + 3]
        if finger == "thumb":
            flex = float(np.linalg.norm(joints, axis=1).sum())
        else:
            flex = float(np.clip(_axis_angle_flexion(joints), 0, None).sum())
        out[finger] = float(np.clip(flex / FULL_CURL_RAD, 0.0, 1.0))
    return out


def wrist_pose_to_deg(wrist_aa: np.ndarray) -> float:
    """Body-wrist axis-angle (3,) -> pitch offset in ZERO wrist degrees."""
    flex = float(np.asarray(wrist_aa, dtype=np.float32).reshape(3)[0])
    return float(np.clip(flex * WRIST_GAIN_DEG_PER_RAD, -45.0, 45.0))


def sequence_to_targets(left_hand_aa, right_hand_aa,
                        left_wrist_aa=None, right_wrist_aa=None
                        ) -> np.ndarray:
    """Frame sequences -> (n, 12) targets in the model's output order:
    [L thumb..pinky closures (5), R thumb..pinky (5), L wrist, R wrist].
    Wrist columns are 0 when wrist rotations aren't provided."""
    n = len(left_hand_aa)
    out = np.zeros((n, 12), dtype=np.float32)
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    for t in range(n):
        lc = hand_pose_to_closure(left_hand_aa[t])
        rc = hand_pose_to_closure(right_hand_aa[t])
        for k, f in enumerate(fingers):
            out[t, k] = lc[f]
            out[t, 5 + k] = rc[f]
        if left_wrist_aa is not None:
            out[t, 10] = wrist_pose_to_deg(left_wrist_aa[t]) / 45.0
        if right_wrist_aa is not None:
            out[t, 11] = wrist_pose_to_deg(right_wrist_aa[t]) / 45.0
    return out


TARGET_DIM = 12
FINGER_SLOTS = ("thumb", "index", "middle", "ring", "pinky")
