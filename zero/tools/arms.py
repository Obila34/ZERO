"""ArmTool — let the LLM (and the fast lexical path) move ZERO's arms/hands.

Same pattern as GazeTool: registered when `arms.enabled`, reaches the live
ArmSystem late-bound through ToolContext.extras['arms']. A gesture the system
doesn't know (not calibrated yet, or a made-up name) gets an honest refusal —
never a silent success.
"""
from __future__ import annotations

from zero.arms.commands import parse_arm_command
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
    description = ("Move ZERO's arms or hands with a named gesture. Use for "
                   "'wave', 'raise your arm', 'open/close your hand', "
                   "'arms down'.")
    parameters = {
        "gesture": "one of: wave_right | wave_left | raise_right | raise_left "
                   "| open_right_hand | close_right_hand | open_left_hand | "
                   "close_left_hand | handshake | rest",
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        arms = (ctx.extras or {}).get("arms")
        if arms is None:
            return "I can't move my arms right now."
        if args.get("text"):
            cmd = parse_arm_command(str(args["text"]))
            name = cmd["name"] if cmd else None
        else:
            name = str(args.get("gesture", "")).strip().lower() or None
        if not name:
            return "I didn't catch which gesture you want."
        if not arms.play(name):
            return ("I can't do that gesture yet — that part of my arms isn't "
                    "calibrated.")
        return _SPOKEN.get(name, "Okay.")
