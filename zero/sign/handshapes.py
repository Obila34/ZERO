"""The manual alphabet, in the robot's portable representation.

A handshape is NOT a servo pose. It is per-finger CLOSURE (0.0 extended ..
1.0 curled) plus a symbolic wrist ORIENTATION, expanded to per-side degrees
by zero.arms.hands at play time. The values here are the operator's
supervised visual calibration (Pi working tree, 2026-08-24) normalised
through the left hand's servo calibration — the flat degree table it
replaces drove the asymmetric right hand past its stops.

KSL fingerspelling uses the standard one-handed manual alphabet. Six letters
exceed what this hand can mechanically do, and the table SAYS so per letter
instead of quietly signing the wrong thing:

  * R crosses index over middle; U/V differ by finger spread; the hand has
    one flexion servo per finger and no abduction, so R and V fall back to
    the U handshape (quality "approx").
  * P and Q are K and G pointed downward; the wrist pitch cannot point the
    hand down (that needs forearm rotation — a stepper, dark until Phase 5),
    so they fall back to their upright bases (quality "approx").
  * J and Z are traced. J is achievable as a wrist sweep (encoded as motion
    keyframes); Z needs an arm path, so its static index-hand is "approx".

The engine speaks each letter as it signs ("P - E - T - E - R"), so an
approximate letter is still unambiguous to the person — and describe() lets
ZERO answer honestly when asked what it can sign.
"""
from __future__ import annotations

# closure: {finger: 0.0 open .. 1.0 closed}; missing fingers are open.
# orient: wrist token from zero.arms.hands.WRIST_ORIENT.
# quality: "exact" | "approx"; note says WHY when approx.
# motion: optional [(orient, seconds), ...] wrist keyframes after the shape
#         lands (J's sweep); static letters omit it.
HANDSHAPES: dict[str, dict] = {
    "A": {"closure": {"index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "B": {"closure": {"thumb": 1.0},
          "orient": "forward", "quality": "exact"},
    "C": {"closure": {"thumb": 0.45, "index": 0.45, "middle": 0.45,
                      "ring": 0.45, "pinky": 0.45},
          "orient": "forward", "quality": "exact"},
    "D": {"closure": {"thumb": 0.65, "middle": 0.9, "ring": 0.9, "pinky": 0.9},
          "orient": "forward", "quality": "exact"},
    "E": {"closure": {"thumb": 1.0, "index": 0.85, "middle": 0.85,
                      "ring": 0.85, "pinky": 0.85},
          "orient": "forward", "quality": "exact"},
    "F": {"closure": {"thumb": 0.55, "index": 0.85},
          "orient": "forward", "quality": "exact"},
    "G": {"closure": {"middle": 1.0, "ring": 1.0, "pinky": 1.0},
          "orient": "in", "quality": "exact"},
    "H": {"closure": {"ring": 1.0, "pinky": 1.0},
          "orient": "in", "quality": "exact"},
    "I": {"closure": {"thumb": 0.85, "index": 1.0, "middle": 1.0,
                      "ring": 1.0},
          "orient": "forward", "quality": "exact"},
    "J": {"closure": {"thumb": 0.85, "index": 1.0, "middle": 1.0,
                      "ring": 1.0},
          "orient": "forward", "quality": "exact",
          "motion": [("in", 0.4), ("forward", 0.3)]},   # the J trace
    "K": {"closure": {"thumb": 0.35, "middle": 0.4, "ring": 1.0,
                      "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "L": {"closure": {"middle": 1.0, "ring": 1.0, "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "M": {"closure": {"thumb": 1.0, "index": 0.9, "middle": 0.9,
                      "ring": 0.9, "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "N": {"closure": {"thumb": 0.85, "index": 0.85, "middle": 0.85,
                      "ring": 1.0, "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "O": {"closure": {"thumb": 0.63, "index": 0.5, "middle": 0.5,
                      "ring": 0.55, "pinky": 0.55},
          "orient": "forward", "quality": "exact"},
    "P": {"closure": {"thumb": 0.35, "middle": 0.4, "ring": 1.0,
                      "pinky": 1.0},
          "orient": "forward", "quality": "approx",
          "note": "P is K pointed down; pointing down needs forearm "
                  "rotation this build keeps dark"},
    "Q": {"closure": {"middle": 1.0, "ring": 1.0, "pinky": 1.0},
          "orient": "in", "quality": "approx",
          "note": "Q is G pointed down; pointing down needs forearm "
                  "rotation this build keeps dark"},
    "R": {"closure": {"thumb": 0.85, "ring": 1.0, "pinky": 1.0},
          "orient": "forward", "quality": "approx",
          "note": "R crosses index over middle; the fingers cannot cross"},
    "S": {"closure": {"thumb": 1.0, "index": 1.0, "middle": 1.0,
                      "ring": 1.0, "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "T": {"closure": {"thumb": 0.85, "index": 0.85, "middle": 1.0,
                      "ring": 1.0, "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "U": {"closure": {"thumb": 0.85, "ring": 1.0, "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "V": {"closure": {"thumb": 0.85, "ring": 1.0, "pinky": 1.0},
          "orient": "forward", "quality": "approx",
          "note": "V spreads index and middle; the fingers cannot spread, "
                  "so V reads as U"},
    "W": {"closure": {"thumb": 0.85, "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "X": {"closure": {"thumb": 0.85, "index": 0.5, "middle": 1.0,
                      "ring": 1.0, "pinky": 1.0},
          "orient": "forward", "quality": "exact"},
    "Y": {"closure": {"index": 1.0, "middle": 1.0, "ring": 1.0},
          "orient": "forward", "quality": "exact"},
    "Z": {"closure": {"thumb": 0.85, "middle": 1.0, "ring": 1.0,
                      "pinky": 1.0},
          "orient": "in", "quality": "approx",
          "note": "Z is traced by the index finger; the trace needs an arm "
                  "path this build keeps dark"},
}


def describe(letter: str) -> str:
    """An honest one-liner about how well this letter can be signed."""
    ent = HANDSHAPES.get(letter.upper())
    if ent is None:
        return f"I don't have a sign for {letter!r}."
    if ent["quality"] == "exact":
        return f"{letter.upper()}: signed exactly."
    return f"{letter.upper()}: approximate — {ent['note']}."


def capabilities() -> dict[str, list[str]]:
    """Letters by fidelity, for the prompt block and for honest answers."""
    out: dict[str, list[str]] = {"exact": [], "approx": []}
    for letter, ent in sorted(HANDSHAPES.items()):
        out[ent["quality"]].append(letter)
    return out
