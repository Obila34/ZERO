"""ArmTool — let the LLM (and the fast lexical path) move ZERO's arms/hands.

Same pattern as GazeTool: registered when `arms.enabled`, reaches the live
ArmSystem late-bound through ToolContext.extras['arms']. A gesture the system
doesn't know (not calibrated yet, or a made-up name) gets an honest refusal —
never a silent success.
"""
from __future__ import annotations

from zero.arms.commands import _PART_JOINT, parse_arm_command
from zero.tools.base import Tool, ToolContext

_SPOKEN = {
    "wave_right": "Okay, waving.",
    "wave_left": "Okay, waving.",
    "rest": "Okay, arms down.",
    "raise_right": "Okay, raising my right arm.",
    "raise_left": "Okay, raising my left arm.",
    "open_right_hand": "Okay, opening my hand.",
    "open_left_hand": "Okay, opening my hand.",
    "close_right_hand": "Okay, closing my hand.",
    "close_left_hand": "Okay, closing my hand.",
    "handshake": "Okay, let's shake hands.",
}


class ArmTool(Tool):
    name = "arms"
    description = ("Move ZERO's arms. Either a named gesture ('wave', 'arms "
                   "down') or one joint via part/side/degrees ('raise your "
                   "right elbow').")
    parameters = {
        "gesture": "one of: wave_right | wave_left | raise_right | raise_left "
                   "| open_right_hand | close_right_hand | open_left_hand | "
                   "close_left_hand | handshake | rest",
        "part": "to move one joint instead: elbow | shoulder | arm | bicep",
        "side": "which arm for `part`: right | left | both (default right)",
        "degrees": "how far to move `part`; negative lowers/bends (default 15)",
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        arms = (ctx.extras or {}).get("arms")
        if arms is None:
            return "I can't move my arms right now."
        cmd = None
        if args.get("text"):
            cmd = parse_arm_command(str(args["text"]))
        elif args.get("part"):
            part = str(args["part"]).strip().lower()
            suffix = _PART_JOINT.get(part)
            if suffix is None:
                return f"I don't have a {part} I can move."
            side = str(args.get("side", "right")).strip().lower()
            sides = ("right", "left") if side == "both" else (side,)
            try:
                deg = float(args.get("degrees", 15.0))
            except (TypeError, ValueError):
                deg = 15.0
            cmd = {"kind": "joint", "part": part, "side": side,
                   "degrees": deg,
                   "joints": [f"{s}_{suffix}" for s in sides]}
        if cmd is not None and cmd.get("kind") == "joint":
            moved = arms.move_joint(cmd["joints"], cmd["degrees"])
            if not moved:
                return (f"I can't move my {cmd['part']} yet — that joint isn't "
                        "calibrated.")
            way = "up" if cmd["degrees"] >= 0 else "down"
            if cmd["side"] == "both":
                what = f"both {cmd['part']}s"       # "both elbows", not "my arm"
            else:
                what = f"{cmd['side']} {cmd['part']}"
            return f"Okay, moving my {what} {way}."
        name = (cmd["name"] if cmd else
                (str(args.get("gesture", "")).strip().lower() or None))
        if not name:
            return "I didn't catch which gesture you want."
        if not arms.play(name):
            return ("I can't do that gesture yet — that part of my arms isn't "
                    "calibrated.")
        return _SPOKEN.get(name, "Okay.")
