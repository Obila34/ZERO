#!/usr/bin/env python3
"""Convert a sign_sim_replay.json into RIG-space animation for zerolabs0's
af1_urdf_rig (Blender). Gateway joint names/degrees -> the rig's anatomical
DOF, using the hardware calibration tables for finger closures and the
rig's documented sign conventions (af1_joint_map.json notes).

    python scripts/rig_export.py /tmp/sign_sim_replay.json --stream phys \
        --out /tmp/rig_anim_phys.json

Output: {"fps": 30, "duration_s": ..., "dof": {rig_dof: [[t, deg], ...]}}
Fingers become mcp1 degrees; the Blender side applies curl_coupling for
mcp2/mcp3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.arms import hands  # noqa: E402

CURL_MAX_DEG = 85.0     # full fist at mcp1; rig couples mcp2/3 from this

# gateway joint -> (rig dof, transform)
_FINGER_RIG = {"thumb": "thumb_mcp1", "index": "index_mcp1",
               "middle": "middle_mcp1", "ring": "ring_mcp1",
               "pinky": "pinky_mcp1"}


def _arm_map(name: str, deg: float):
    side = "right" if name.startswith("right") else "left"
    sign = 1.0 if side == "right" else -1.0
    if "up_down" in name:
        # ours: raise = +right/-left; rig shoulder_pitch: + raises forward
        return f"{side}_shoulder_pitch", sign * deg
    if "in_out" in name:
        # ours: outward = +right/-left; rig shoulder_roll assumed + outward
        return f"{side}_shoulder_roll", sign * deg
    if "elbow" in name:
        # ours: bend forward = NEGATIVE both sides; rig: + flexes forward
        return f"{side}_elbow", -deg
    if name == "head_tilt_joint":
        return "head_yaw", deg
    return None, 0.0


def _wrist_map(name: str, deg: float):
    side = "right" if name.startswith("right") else "left"
    fwd = hands.wrist_deg(side, "forward")
    inw = hands.wrist_deg(side, "in")
    span = (inw - fwd) or 1.0
    # forward -> 0 roll; 'in' preset -> ~70 deg of roll, linearly
    roll = (deg - fwd) / span * 70.0
    return f"{side}_wrist_roll", max(-130.0, min(130.0, roll))


def convert(replay: dict, stream: str) -> dict:
    dof: dict[str, list] = {}
    t_max = 0.0
    for gw_name, tracks in replay["joints"].items():
        t = tracks["t"]
        v = tracks[stream]
        t_max = max(t_max, t[-1] if t else 0.0)
        side = "right" if gw_name.startswith("right") else "left"
        matched = False
        for f in _FINGER_RIG:
            if f in gw_name:
                # rig hands ride CTRL_hand_{R,L} custom curl props, 0..1
                dof[f"curl:{side}:{f}"] = [
                    [ts, round(hands.finger_closure(side, f, d), 4)]
                    for ts, d in zip(t, v)]
                matched = True
                break
        if matched:
            continue
        if "wrist" in gw_name:
            pairs = [_wrist_map(gw_name, d) for d in v]
            dof[pairs[0][0]] = [[ts, p[1]] for ts, p in zip(t, pairs)]
            continue
        rig, _ = _arm_map(gw_name, 0.0)
        if rig is not None:
            dof[rig] = [[ts, _arm_map(gw_name, d)[1]]
                        for ts, d in zip(t, v)]
    return {"fps": 30, "duration_s": round(t_max + 0.5, 2),
            "stream": stream, "dof": dof}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("--stream", choices=("cmd", "phys"), default="phys")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    replay = json.load(open(a.replay))
    out = convert(replay, a.stream)
    json.dump(out, open(a.out, "w"))
    n = sum(len(v) for v in out["dof"].values())
    print(f"{a.stream}: {len(out['dof'])} rig DOFs, {n} keys, "
          f"{out['duration_s']}s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
