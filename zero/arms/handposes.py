"""Named hand poses — the expressive finger-gesture vocabulary.

Written in CLOSURE space (0.0 open .. 1.0 closed per finger, plus a symbolic
wrist orientation) and expanded per side by zero.arms.hands at play time, so
the same definition drives both asymmetric hands correctly. The values are
the operator's supervised calibration from the Pi's first sign build,
normalised through the left hand — see zero/arms/hands.py for why the raw
degree table it replaces was unsafe on the right hand.

These play through ArmSystem's normal keyframe pipeline (min-jerk easing,
speed caps, e-stop, preemption) — never through a raw driver send.
"""
from __future__ import annotations

from zero.arms import hands

# name -> (closure, orient, spoken confirmation)
HAND_POSES: dict[str, tuple[dict[str, float], str, str]] = {
    "peace": ({"thumb": 1.0, "ring": 1.0, "pinky": 1.0},
              "forward", "Peace sign!"),
    "i_love_you": ({"middle": 1.0, "ring": 1.0},
                   "forward", "Signing I love you."),
    "thumbs_up": ({"index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
                  "in", "Thumbs up!"),
    "fist": ({"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0,
              "pinky": 1.0},
             "forward", "Making a fist."),
    "open_hand": ({}, "forward", "Opening my hand."),
    "ok_sign": ({"thumb": 0.55, "index": 0.8},
                "forward", "Okay sign."),
    "rock_on": ({"thumb": 1.0, "middle": 1.0, "ring": 1.0},
                "forward", "Rock on!"),
    "pinch": ({"thumb": 0.55, "index": 0.7, "middle": 1.0, "ring": 1.0,
               "pinky": 1.0},
              "forward", "Pinching my thumb and index finger."),
    "point_hand": ({"thumb": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
                   "forward", "Pointing."),
}

# Spoken-name aliases the voice/LLM paths may use.
ALIASES = {
    "victory": "peace", "ily": "i_love_you", "ok": "ok_sign",
    "okay": "ok_sign", "open_hands": "open_hand", "point_finger": "point_hand",
}


def resolve(name: str) -> str | None:
    n = name.lower().strip()
    n = ALIASES.get(n, n)
    return n if n in HAND_POSES else None


def _sides(side: str) -> tuple[str, ...]:
    return ("left", "right") if side in ("both", "all") else (side,)


def hand_gesture_frames(name: str, side: str = "both") -> list | None:
    """ArmSystem keyframes for a named pose: land, hold, ease back open.
    None for an unknown name — refuse, never guess."""
    key = resolve(name)
    if key is None:
        return None
    closure, orient, _spoken = HAND_POSES[key]
    pose: dict[str, float] = {}
    rest: dict[str, float] = {}
    for s in _sides(side):
        pose.update(hands.hand_pose(s, closure, orient))
        rest.update(hands.open_hand_pose(s))
    return [(pose, 0.4), (pose, 0.6), (rest, 0.5)]


def wiggle_frames(side: str = "both") -> list:
    """Fingers ripple open/closed twice — pure fun, same safe pipeline."""
    frames = []
    for _round in range(2):
        for closure in (1.0, 0.0):
            for f in hands.FINGERS:
                pose = {}
                for s in _sides(side):
                    pose[hands.joint_name(s, f)] = hands.finger_deg(
                        s, f, closure)
                frames.append((pose, 0.12))
    rest: dict[str, float] = {}
    for s in _sides(side):
        rest.update(hands.open_hand_pose(s))
    frames.append((rest, 0.4))
    return frames


def spoken(name: str) -> str:
    key = resolve(name)
    return HAND_POSES[key][2] if key else "Okay."
