"""ArmTool — the LLM's (and the fast lexical path's) hands.

One tool for the whole upper body below the neck: arm joints, named arm
gestures, finger poses, individual fingers, and Kenyan Sign Language —
single letters, fingerspelt words, and lexicon signs.

Two rules the first sign build broke, restored here:
  * NOTHING calls the driver directly. Arm/hand motion goes through
    ArmSystem (min-jerk playback, speed caps, envelopes, e-stop) and sign
    motion through SignEngine (same, on the bus's sign track).
  * No fake success. A gesture that can't play, a sign the lexicon doesn't
    have, an uncalibrated joint — each comes back as an honest sentence,
    never "Command received."

The tool's return value is the exact sentence to speak (speaks_directly):
for a spelled word that sentence IS the letter readout, paced to match the
hands, so routing it through a second LLM pass would break the sync.
"""
from __future__ import annotations

from zero.arms.commands import _PART_JOINT, parse_arm_command
from zero.tools.base import Tool, ToolContext

_SPOKEN = {
    "wave_right": "Okay, waving.",
    "wave_left": "Okay, waving.",
    "rest": "Okay, arms down.",
    "handshake": "Okay, let's shake hands.",
}


class ArmTool(Tool):
    name = "arms"
    description = (
        "Move ZERO's arms, hands and fingers, and sign in Kenyan Sign "
        "Language (KSL). Use for ANY request to move, raise, lower or bend "
        "an arm, shoulder, elbow or bicep; open or close a hand; make a "
        "finger pose (peace, thumbs up, fist, I-love-you, OK, rock on, "
        "pinch, wiggle); curl or extend one finger; sign a letter; "
        "fingerspell a word; or perform a KSL sign.")
    parameters = {
        "action": "spell | letter | sign | hand_gesture | hand | finger | "
                  "joint | rest. Pick the one matching the request.",
        "word": "for spell: the word to fingerspell (e.g. PETER).",
        "letter": "for letter: one letter A-Z to sign.",
        "gloss": "for sign: the word whose KSL sign to perform.",
        "gesture": "for hand_gesture: peace | i_love_you | thumbs_up | fist "
                   "| open_hand | ok_sign | rock_on | pinch | wiggle | "
                   "point_hand. For arm gestures: wave_right | wave_left.",
        "state": "for hand: open | close.",
        "finger": "for finger: thumb | index | middle | ring | pinky.",
        "part": "for joint: arm | shoulder | elbow | bicep.",
        "side": "right | left | both. Use 'both' when no side is named.",
        "degrees": "for joint: how far, POSITIVE raises/straightens, "
                   "NEGATIVE lowers/bends; ~30 normal, 999 = as far as it "
                   "goes. For finger: servo degrees.",
    }
    speaks_directly = True

    # ── dispatch ─────────────────────────────────────────────────────────────
    def run(self, args: dict, ctx: ToolContext) -> str:
        arms = (ctx.extras or {}).get("arms")
        sign = (ctx.extras or {}).get("sign")
        if arms is None:
            return "I can't move my arms right now."

        cmd = parse_arm_command(str(args["text"])) if args.get("text") else \
            self._from_args(args)
        if cmd is None:
            part = str(args.get("part", "")).strip().lower()
            if part and part not in _PART_JOINT:
                return f"I don't have a {part} I can move."
            return "I didn't catch what you want me to do with my hands."

        kind = cmd.get("kind")
        if kind in ("spell", "spell_name", "letter", "sign"):
            return self._do_sign(cmd, sign, ctx)
        if kind == "hand_gesture":
            spoken = arms.hand_gesture(cmd["name"], cmd.get("side", "both"))
            return spoken or ("I can't do that one right now — my hands "
                              "aren't ready.")
        if kind == "hand":
            pose = "open_hand" if cmd["state"] == "open" else "fist"
            spoken = arms.hand_gesture(pose, cmd.get("side", "both"))
            if spoken is None:
                return "I can't move my hands right now."
            verb = "Opening" if cmd["state"] == "open" else "Closing"
            what = ("both hands" if cmd.get("side", "both") == "both"
                    else f"my {cmd['side']} hand")
            return f"{verb} {what}."
        if kind == "finger":
            spoken = arms.move_finger(cmd["finger"], cmd.get("side", "both"),
                                      closure=cmd.get("closure"),
                                      degrees=cmd.get("degrees"))
            return spoken or f"I can't move that {cmd['finger']} right now."
        if kind == "joint":
            moved = arms.move_joint(cmd["joints"], cmd["degrees"])
            if not moved:
                return (f"I can't move my {cmd['part']} yet — that joint "
                        "isn't calibrated.")
            way = "up" if cmd["degrees"] >= 0 else "down"
            if cmd["side"] == "both":
                return f"Okay, moving both {cmd['part']}s {way}."
            return f"Okay, moving my {cmd['side']} {cmd['part']} {way}."
        # gesture (arm) — including model-invented names, read as speech
        name = cmd.get("name", "")
        if name == "rest":
            arms.rest()
            if sign is not None:
                sign.rest()
            return _SPOKEN["rest"]
        if name not in arms.gesture_names():
            retry = parse_arm_command(name.replace("_", " "))
            if retry and retry.get("kind") != "gesture":
                return self.run({"text": name.replace("_", " ")}, ctx)
            if retry:
                name = retry["name"]
        if not arms.play(name):
            # Not an arm gesture — maybe it's a hand pose by another name.
            spoken = arms.hand_gesture(name, cmd.get("side", "both"))
            if spoken is not None:
                return spoken
            return ("I can't do that one — I don't have a gesture called "
                    f"{name!r}.")
        return _SPOKEN.get(name, "Okay.")

    # ── the sign family ──────────────────────────────────────────────────────
    def _do_sign(self, cmd: dict, sign, ctx: ToolContext) -> str:
        if sign is None:
            return ("I can't sign right now — my sign system isn't "
                    "running.")
        kind = cmd["kind"]
        if kind == "spell_name":
            name = (ctx.person_name or "").strip()
            if not name:
                return ("I don't actually know your name yet — tell me and "
                        "I'll spell it.")
            spoken = sign.spell(name)
            return spoken or "I can't spell right now."
        if kind == "spell":
            spoken = sign.spell(cmd["word"])
            return spoken or "I can't spell right now."
        if kind == "letter":
            spoken = sign.letter(cmd["letter"])
            return spoken or (f"I don't have a sign for "
                              f"{cmd['letter']!r}.")
        # kind == "sign": the lexicon, with an HONEST fingerspell fallback —
        # never an invented movement.
        gloss = cmd["gloss"]
        spoken = sign.sign(gloss)
        if spoken is not None:
            return spoken
        spelled = sign.spell(gloss)
        if spelled is None:
            return f"I don't know the sign for {gloss} yet."
        return (f"I don't know the full sign for {gloss} yet, so I'll "
                f"spell it. {spelled}")

    # ── structured args from the LLM ─────────────────────────────────────────
    def _from_args(self, args: dict) -> dict | None:
        action = str(args.get("action", "")).strip().lower()
        side = str(args.get("side", "both")).strip().lower() or "both"
        if action == "spell" or args.get("word"):
            word = str(args.get("word", "")).strip()
            if word.lower() in ("my name", "name", ""):
                return {"kind": "spell_name"}
            return {"kind": "spell", "word": word.upper()}
        if action == "letter" or args.get("letter"):
            let = str(args.get("letter", "")).strip()
            return {"kind": "letter", "letter": let.upper()} if let else None
        if action == "sign" or args.get("gloss"):
            gloss = str(args.get("gloss", "")).strip().lower()
            return {"kind": "sign", "gloss": gloss} if gloss else None
        if action == "hand":
            state = str(args.get("state", "open")).strip().lower()
            return {"kind": "hand", "state": state, "side": side}
        if action == "finger" or args.get("finger"):
            f = str(args.get("finger", "")).strip().lower()
            if not f:
                return None
            out = {"kind": "finger", "finger": f, "side": side,
                   "closure": None, "degrees": None}
            if args.get("degrees") is not None:
                try:
                    out["degrees"] = float(args["degrees"])
                except (TypeError, ValueError):
                    out["closure"] = 1.0
            else:
                out["closure"] = 1.0
            return out
        if action == "rest":
            return {"kind": "gesture", "name": "rest"}
        if action == "joint" or args.get("part"):
            part = str(args.get("part", "")).strip().lower()
            suffix = _PART_JOINT.get(part)
            if suffix is None:
                return None
            # Arm moves keep the old default of RIGHT when no side is named;
            # only the bilateral hand/sign family defaults to both.
            side = str(args.get("side", "right")).strip().lower() or "right"
            sides = ("right", "left") if side == "both" else (side,)
            try:
                deg = float(args.get("degrees", 15.0))
            except (TypeError, ValueError):
                deg = 15.0
            return {"kind": "joint", "part": part, "side": side,
                    "degrees": deg,
                    "joints": [f"{s}_{suffix}" for s in sides]}
        gesture = str(args.get("gesture", "")).strip().lower()
        if gesture:
            return {"kind": "gesture", "name": gesture, "side": side}
        return None
