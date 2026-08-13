"""GazeTool — let the LLM (and the fast lexical path) move ZERO's head.

Same controller as tracking — there is no separate 'command mode'. The tool is
registered only when the head subsystem is enabled, and reaches it through
ToolContext.extras['head'] (late-bound to the live HeadSystem each call).

Two ways in:
  * FAST PATH: the router's tier-1 passes the raw utterance as {"text": "..."}.
    parse_gaze_command handles 'look left', 'face forward', 'look at Sam' with NO
    LLM round trip.
  * LLM: emits {"tool":"gaze","args":{"target":"left|right|up|down|center|<name>",
    "degrees":25, "then_answer":true}}. then_answer = turn, settle, THEN answer
    (gaze leads language) — used for 'what's on the table over there?'.
"""
from __future__ import annotations

from zero.head.commands import parse_gaze_command
from zero.tools.base import Tool, ToolContext

_DIRS = {"left": ("pan", -1.0), "right": ("pan", 1.0),
         "up": ("tilt", 1.0), "down": ("tilt", -1.0)}
_CENTER = {"center", "centre", "forward", "forwards", "ahead", "straight", "front"}


def _cmd_from_args(args: dict) -> dict | None:
    if args.get("text"):
        return parse_gaze_command(str(args["text"]))
    target = str(args.get("target", "")).strip().lower()
    if not target:
        return None
    if target in _CENTER:
        return {"kind": "center"}
    if target in _DIRS:
        axis, sign = _DIRS[target]
        return {"kind": "direction", "axis": axis, "sign": sign,
                "degrees": float(args.get("degrees", 25.0))}
    return {"kind": "person", "name": target}


class GazeTool(Tool):
    name = "gaze"
    description = ("Turn ZERO's head to look somewhere. Use for 'look left/right/"
                   "up/down', 'face forward', 'look at <name>', or to look at "
                   "something before describing it.")
    parameters = {
        "target": "where to look: left | right | up | down | center | a person's name",
        "degrees": "optional angle for a direction, 0-80 (default 25; use ~80 for "
                   "'all the way'/'completely')",
        "then_answer": "true if you must look there BEFORE answering (e.g. "
                       "'what's over there?') — ZERO turns and settles first",
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        head = (ctx.extras or {}).get("head")
        if head is None:
            return "I can't move my head right now."
        cmd = _cmd_from_args(args)
        if not cmd:
            return "I didn't catch where you want me to look."
        then = bool(args.get("then_answer") or args.get("look_then_speak"))
        if then:
            head.look_and_settle(cmd)
            return ("Turned to look. Now describe what's in view from the "
                    "current camera frame.")
        return head.apply_command(cmd)
