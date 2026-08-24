"""Hand hardware truth — the 12 PCA joints (2 wrists + 10 fingers).

Ground truth is the AF-1 gateway firmware's own calibration (JOINTS_CONFIG /
FINGER_LIMITS in the gateway UI, read off the live gateway 2026-08-24), which
the Pi's first sign build got wrong in two ways this module exists to fix:

  * The hands are ASYMMETRIC. The right thumb closes at 70 deg where the left
    closes at 140; the right pinky opens at 83 vs the left's 99. Copying left
    angles onto right joints (what the old bilateral code did) drove the right
    thumb 70 deg past its physical stop on every closed-thumb sign.
  * Config envelopes of 0-180 on every finger were fiction; real travel is
    70-140 deg depending on the joint. The envelope table lives HERE, in code,
    with provenance — config may override a joint explicitly, but the default
    is the firmware's truth, so the two can no longer silently drift apart.

The portable representation is NORMALISED CLOSURE: 0.0 = fully open/extended,
1.0 = fully closed/curled, per finger. Sign handshapes are written in closure
space and expanded per side here (open + c * (close - open)) — the same
normalise-then-expand step the firmware's own fingerspell code does, and the
reason a recalibrated servo doesn't invalidate every sign.

Wrist orientation is symbolic for the same reason: the two wrists mount
mirrored, so "palm forward" is 180 deg on the left and 0 on the right.

Stdlib only.
"""
from __future__ import annotations

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("left", "right")

# Per-side, per-finger servo calibration: (open_deg, close_deg).
# From the gateway firmware's FINGER_LIMITS, except the thumbs' open end is
# held at 5.0 (the Pi's supervised calibration found 0 rests the servo on its
# hard stop; 5 is the same margin the operator's letter table uses).
FINGER_CAL: dict[str, dict[str, tuple[float, float]]] = {
    "left": {
        "thumb":  (5.0, 140.0),
        "index":  (90.0, 0.0),
        "middle": (90.0, 0.0),
        "ring":   (99.0, 0.0),
        "pinky":  (99.0, 0.0),
    },
    "right": {
        "thumb":  (5.0, 70.0),
        "index":  (93.0, 0.0),
        "middle": (100.0, 0.0),
        "ring":   (90.0, 0.0),
        "pinky":  (83.0, 0.0),
    },
}

# Wrist pitch per side, by symbolic orientation. From the firmware's joint
# labels: left 180=palm forward, 90=inward, 0=cup/upward; right mirrored
# (0=forward, 70=inward, 170=upward — the right's travel is offset, again
# per firmware).
WRIST_ORIENT: dict[str, dict[str, float]] = {
    "left":  {"forward": 180.0, "in": 90.0, "up": 0.0},
    "right": {"forward": 0.0,   "in": 70.0, "up": 170.0},
}

# Rest pose: hands hanging naturally — palms toward the body, fingers open.
REST_ORIENT = "in"
WRIST_TRAVEL = {"left": (0.0, 180.0), "right": (0.0, 180.0)}


def joint_name(side: str, finger: str) -> str:
    return f"{side}_{finger}p1_joint"


def wrist_name(side: str) -> str:
    return f"{side}_wrist_joint"


def finger_deg(side: str, finger: str, closure: float) -> float:
    """Closure fraction (0 open .. 1 closed) -> this side's servo degrees."""
    o, c = FINGER_CAL[side][finger]
    cl = min(1.0, max(0.0, float(closure)))
    return o + cl * (c - o)


def finger_closure(side: str, finger: str, deg: float) -> float:
    """Inverse of finger_deg — servo degrees back to closure fraction."""
    o, c = FINGER_CAL[side][finger]
    if c == o:
        return 0.0
    return min(1.0, max(0.0, (float(deg) - o) / (c - o)))


def wrist_deg(side: str, orient: str) -> float:
    """Symbolic orientation -> this side's wrist servo degrees. An unknown
    token raises — a sign must never be played with a guessed wrist."""
    return WRIST_ORIENT[side][orient]


def hand_pose(side: str, closure: dict[str, float],
              orient: str | None = None) -> dict[str, float]:
    """One hand's joint targets from closure fractions (+ optional wrist).
    Missing fingers default to open — a handshape states what is bent."""
    pose = {joint_name(side, f): finger_deg(side, f, closure.get(f, 0.0))
            for f in FINGERS}
    if orient is not None:
        pose[wrist_name(side)] = wrist_deg(side, orient)
    return pose


def open_hand_pose(side: str, orient: str = REST_ORIENT) -> dict[str, float]:
    return hand_pose(side, {}, orient)


def hand_joint_specs() -> dict[str, dict[str, float]]:
    """{joint: {min, max, home}} for all 12 PCA joints — the calibrated
    envelope table, generated from firmware truth rather than hand-written
    into config. min/max are the servo's real travel (order-normalised);
    home is the rest pose: open fingers, palms inward."""
    specs: dict[str, dict[str, float]] = {}
    for side in SIDES:
        for f in FINGERS:
            o, c = FINGER_CAL[side][f]
            specs[joint_name(side, f)] = {
                "min": min(o, c), "max": max(o, c), "home": o}
        lo, hi = WRIST_TRAVEL[side]
        specs[wrist_name(side)] = {
            "min": lo, "max": hi, "home": wrist_deg(side, REST_ORIENT)}
    return specs


HAND_JOINTS = frozenset(hand_joint_specs())
