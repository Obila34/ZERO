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
    description = ("Move ZERO's arms — use this for ANY request to move, "
                   "raise, lower, lift or bend an arm, shoulder, elbow or "
                   "bicep, including follow-ups like 'higher' or 'put them "
                   "down'. Normally set part/side/degrees; only use `gesture` "
                   "for a named one.")
    parameters = {
        "part": "what to move: arm | shoulder | elbow | bicep. 'arm' and "
                "'shoulder' both raise/lower the whole arm.",
        "side": "right | left | both. Use 'both' whenever the person says "
                "'arms'/'hands' plural without naming a side.",
        "degrees": "how far, POSITIVE to raise/straighten and NEGATIVE to "
                   "lower/bend. ~30 is a normal move, ~60 a big one, 999 "
                   "means as far as it goes. Omit for a normal move.",
        "gesture": "only for a named gesture: wave_right | wave_left | rest "
                   "(rest puts both arms back down).",
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
                return f"Okay, moving both {cmd['part']}s {way}."
            return f"Okay, moving my {cmd['side']} {cmd['part']} {way}."
        name = (cmd["name"] if cmd else
                (str(args.get("gesture", "")).strip().lower() or None))
        if not name:
            return "I didn't catch which gesture you want."
        if name not in arms.gesture_names() and name != "rest":
            # The model invents names ("arms_up", "raise_arms"). Read the name
            # as if it were spoken, so a near-miss still does the right thing.
            retry = parse_arm_command(name.replace("_", " "))
            if retry and retry.get("kind") == "joint":
                moved = arms.move_joint(retry["joints"], retry["degrees"])
                if moved:
                    way = "up" if retry["degrees"] >= 0 else "down"
                    return f"Okay, moving my {retry['part']} {way}."
            if retry and retry.get("kind") == "gesture":
                name = retry["name"]
        if not arms.play(name):
            return ("I can't do that one — I don't have a gesture called "
                    f"{name!r}.")
        return _SPOKEN.get(name, "Okay.")
