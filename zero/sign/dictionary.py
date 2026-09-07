"""Motion sign dictionary — recorded ASL signs as bus keyframes.

data/sign_dictionary.npz (built by scripts/sign_dict_compile.py from the
PerceptX vault's sign_motion.npz) holds per-gloss trajectories already
retargeted to ZERO's spaces: finger closures 0..1, a wrist orient blend,
and CONSERVATIVELY capped arm degrees in the hardware-verified sign
conventions.

This module turns one gloss's trajectory into the exact keyframe list
the SignEngine's player consumes — so dictionary signs inherit every
playback safety the engine already has (easing, e-stop, arm lowering,
preemption, finish-open) instead of growing a second player.

Precedence lives in the ENGINE: the signer-reviewed lexicon always wins
over the dictionary; the dictionary wins over letter-by-letter
fingerspelling. Recorded motion is real signing, but it is ASL and
machine-retargeted — a signer-approved KSL entry must always shadow it.
"""
from __future__ import annotations

import os

import numpy as np

from zero.arms import hands
from zero.utils.logging import get_logger

log = get_logger("sign.dictionary")

# robot arm-joint order in the compiled `arms` array
_ARM_JOINTS = ("right_elbow_joint", "right_up_down_joint",
               "right_in_out_joint", "left_elbow_joint",
               "left_up_down_joint", "left_in_out_joint")
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class SignDictionary:
    def __init__(self, path: str, *, keyframe_s: float = 0.12,
                 arm_cap_deg: float = 45.0):
        d = np.load(path)
        self._glosses = [str(g) for g in d["glosses"]]
        self._idx = {g: i for i, g in enumerate(self._glosses)}
        self._closures = d["closures"]
        self._wrists = d["wrists"]
        self._arms = d["arms"]
        self._present = d["present"]
        self._kf_s = float(keyframe_s)
        self._cap = float(arm_cap_deg)   # defense in depth over compile caps
        log.info("sign dictionary: %d signs (%s)", len(self._glosses),
                 os.path.basename(path))

    def __contains__(self, gloss: str) -> bool:
        return self._normalize(gloss) in self._idx

    def glosses(self) -> list[str]:
        return list(self._glosses)

    @staticmethod
    def _normalize(gloss: str) -> str:
        return "".join(c for c in gloss.lower().strip() if c.isalnum())

    def frames(self, gloss: str):
        """Keyframes [(pose, move_s, hold_s), ...] for the engine's player,
        plus the arm joints used (for the engine's lower-at-end rule) and
        the sides involved. None when the gloss is unknown."""
        i = self._idx.get(self._normalize(gloss))
        if i is None:
            return None
        cl = np.asarray(self._closures[i], np.float32)     # (T,10)
        wr = np.asarray(self._wrists[i], np.float32)       # (T,2)
        ar = np.asarray(self._arms[i], np.float32)         # (T,6)
        pr = np.asarray(self._present[i], bool)            # (T,2)

        # forward-fill invisible frames from the last visible pose — a
        # tracking dropout must hold the sign, not snap the hand open
        sides_seen = {s for j, s in enumerate(("left", "right"))
                      if pr[:, j].any()}
        if not sides_seen:
            return None
        last = {"left": None, "right": None}
        keyframes = []
        arm_used: set[str] = set()
        T = len(cl)
        step = max(1, int(round(self._kf_s * 24.0)))   # source ~24 fps
        for t in range(0, T, step):
            pose: dict[str, float] = {}
            for j, side in enumerate(("left", "right")):
                if side not in sides_seen:
                    continue
                off = 0 if side == "left" else 5
                if pr[t, j]:
                    closure = {f: float(cl[t, off + k])
                               for k, f in enumerate(_FINGERS)}
                    w = float(np.clip(wr[t, j], 0.0, 1.0))
                    wrist = (hands.wrist_deg(side, "forward") * (1 - w)
                             + hands.wrist_deg(side, "in") * w)
                    arm = {}
                    for k in range(3):
                        name = _ARM_JOINTS[k if side == "right" else 3 + k]
                        deg = float(np.clip(ar[t, (0 if side == "right"
                                                   else 3) + k],
                                            -self._cap, self._cap))
                        if abs(deg) > 0.5:
                            arm[name] = deg
                            arm_used.add(name)
                    last[side] = (closure, wrist, arm)
                if last[side] is None:
                    continue
                closure, wrist, arm = last[side]
                for f, c in closure.items():
                    pose[hands.joint_name(side, f)] = hands.finger_deg(
                        side, f, max(0.0, min(1.0, c)))
                pose[hands.wrist_name(side)] = wrist
                pose.update(arm)
            if pose:
                keyframes.append((pose, self._kf_s, 0.0))
        if not keyframes:
            return None
        # settle on the final pose briefly so the sign reads before the
        # engine lowers arms and opens hands
        p, mv, _ = keyframes[-1]
        keyframes[-1] = (p, mv, 0.35)
        return keyframes, sorted(arm_used), tuple(sorted(sides_seen))


def load_dictionary(cfg):
    """The dictionary, or None when gated off (`sign.dictionary.enabled`,
    default false) or the compiled asset is absent."""
    if not cfg.get("sign.dictionary.enabled", False):
        return None
    path = str(cfg.get("sign.dictionary.path", "data/sign_dictionary.npz"))
    if not os.path.exists(path):
        log.warning("sign dictionary enabled but %s is missing — "
                    "fingerspelling only", path)
        return None
    try:
        return SignDictionary(
            path,
            keyframe_s=float(cfg.get("sign.dictionary.keyframe_s", 0.12)),
            arm_cap_deg=float(cfg.get("sign.dictionary.arm_cap_deg", 45.0)))
    except Exception as e:
        log.warning("sign dictionary failed to load (%s) — "
                    "fingerspelling only", e)
        return None
