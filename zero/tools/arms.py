"""ArmTool — control ZERO / AF-1 humanoid robot arms, wrists, fingers, and gestures."""
from __future__ import annotations
import threading
import time
from zero.arms.commands import parse_arm_command
from zero.arms.system import ASL_LETTER_ANGLES
from zero.tools.base import Tool, ToolContext

JOINT_LIMITS_TABLE = {
    "left_wrist_joint": {"min": 0.0, "max": 180.0, "desc": "Left wrist pitch (180=forward, 90=inward, 0=upward)"},
    "right_wrist_joint": {"min": 0.0, "max": 180.0, "desc": "Right wrist pitch (0=forward, 70=inward, 170=upward)"},
    "left_thumbp1_joint": {"min": 5.0, "max": 140.0, "desc": "Left thumb servo (5=open, 140=closed)"},
    "left_indexp1_joint": {"min": 0.0, "max": 90.0, "desc": "Left index finger (90=open, 0=closed)"},
    "left_middlep1_joint": {"min": 0.0, "max": 90.0, "desc": "Left middle finger (90=open, 0=closed)"},
    "left_ringp1_joint": {"min": 0.0, "max": 99.0, "desc": "Left ring finger (99=open, 0=closed)"},
    "left_pinkyp1_joint": {"min": 0.0, "max": 99.0, "desc": "Left pinky finger (99=open, 0=closed)"},
    "right_thumbp1_joint": {"min": 5.0, "max": 140.0, "desc": "Right thumb servo (5=open, 140=closed)"},
    "right_indexp1_joint": {"min": 0.0, "max": 90.0, "desc": "Right index finger (90=open, 0=closed)"},
    "right_middlep1_joint": {"min": 0.0, "max": 90.0, "desc": "Right middle finger (90=open, 0=closed)"},
    "right_ringp1_joint": {"min": 0.0, "max": 99.0, "desc": "Right ring finger (99=open, 0=closed)"},
    "right_pinkyp1_joint": {"min": 0.0, "max": 99.0, "desc": "Right pinky finger (99=open, 0=closed)"},
}

# Calibrated Finger Gestures:
# 90.0/99.0 = OPEN / EXTENDED | 0.0 = CLOSED / CURLED | Thumb: 5.0=OPEN, 140.0=CLOSED
FINGER_GESTURES = {
    "peace": {
        "thumb": 140.0, "index": 90.0, "middle": 90.0, "ring": 0.0, "pinky": 0.0, "wrist": 180.0,
        "spoken": "Peace sign!"
    },
    "victory": {
        "thumb": 140.0, "index": 90.0, "middle": 90.0, "ring": 0.0, "pinky": 0.0, "wrist": 180.0,
        "spoken": "Victory sign!"
    },
    "i_love_you": {
        "thumb": 5.0, "index": 90.0, "middle": 0.0, "ring": 0.0, "pinky": 99.0, "wrist": 180.0,
        "spoken": "Signing I Love You with both hands."
    },
    "ily": {
        "thumb": 5.0, "index": 90.0, "middle": 0.0, "ring": 0.0, "pinky": 99.0, "wrist": 180.0,
        "spoken": "Signing I Love You with both hands."
    },
    "point": {
        "thumb": 140.0, "index": 90.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "wrist": 180.0,
        "spoken": "Pointing forward."
    },
    "thumbs_up": {
        "thumb": 5.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "wrist": 90.0,
        "spoken": "Thumbs up!"
    },
    "fist": {
        "thumb": 140.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "wrist": 180.0,
        "spoken": "Making a fist."
    },
    "open_hands": {
        "thumb": 5.0, "index": 90.0, "middle": 90.0, "ring": 99.0, "pinky": 99.0, "wrist": 180.0,
        "spoken": "Opening my hands."
    },
    "ok_sign": {
        "thumb": 80.0, "index": 20.0, "middle": 90.0, "ring": 99.0, "pinky": 99.0, "wrist": 180.0,
        "spoken": "Giving the OK sign."
    },
    "rock_on": {
        "thumb": 140.0, "index": 90.0, "middle": 0.0, "ring": 0.0, "pinky": 99.0, "wrist": 180.0,
        "spoken": "Rock on!"
    },
    "pinch": {
        "thumb": 80.0, "index": 25.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "wrist": 180.0,
        "spoken": "Pinching my thumb and index finger."
    }
}

def get_bilateral_pose(letter: str) -> dict[str, float]:
    ch = letter.upper()
    if ch not in ASL_LETTER_ANGLES:
        return {}
    l_pose = ASL_LETTER_ANGLES[ch]
    l_wrist = l_pose.get("left_wrist_joint", 180.0)
    r_wrist = 70.0 if abs(l_wrist - 90.0) < 1.0 else (0.0 if l_wrist >= 150.0 else 170.0)
    return {
        "left_wrist_joint": l_wrist,
        "left_thumbp1_joint": l_pose.get("left_thumbp1_joint", 0.0),
        "left_indexp1_joint": l_pose.get("left_indexp1_joint", 0.0),
        "left_middlep1_joint": l_pose.get("left_middlep1_joint", 0.0),
        "left_ringp1_joint": l_pose.get("left_ringp1_joint", 0.0),
        "left_pinkyp1_joint": l_pose.get("left_pinkyp1_joint", 0.0),
        "right_wrist_joint": r_wrist,
        "right_thumbp1_joint": l_pose.get("left_thumbp1_joint", 0.0),
        "right_indexp1_joint": l_pose.get("left_indexp1_joint", 0.0),
        "right_middlep1_joint": l_pose.get("left_middlep1_joint", 0.0),
        "right_ringp1_joint": l_pose.get("left_ringp1_joint", 0.0),
        "right_pinkyp1_joint": l_pose.get("left_pinkyp1_joint", 0.0),
    }

def build_gesture_pose(g_data: dict, side: str = "both") -> dict[str, float]:
    pose = {}
    sides = ("left", "right") if side in ("both", "all") else (side,)
    l_wrist = g_data.get("wrist", 180.0)
    r_wrist = 70.0 if abs(l_wrist - 90.0) < 1.0 else (0.0 if l_wrist >= 150.0 else 170.0)
    for s in sides:
        pose[f"{s}_wrist_joint"] = l_wrist if s == "left" else r_wrist
        pose[f"{s}_thumbp1_joint"] = g_data["thumb"]
        pose[f"{s}_indexp1_joint"] = g_data["index"]
        pose[f"{s}_middlep1_joint"] = g_data["middle"]
        pose[f"{s}_ringp1_joint"] = g_data["ring"]
        pose[f"{s}_pinkyp1_joint"] = g_data["pinky"]
    return pose

class ArmTool(Tool):
    name = "arms"
    description = (
        "Actuate ZERO's physical robot fingers, hands, wrists, and Kenyan Sign Language (KSL). "
        "Use this for ANY command to move fingers (thumb, index, middle, ring, pinky), gestures (peace sign, "
        "I love you sign, point, thumbs up, fist, open hands, ok sign, rock on, pinch, wiggle fingers), show sign letters (A-Z), "
        "or fingerspell words on BOTH hands."
    )
    parameters = {
        "action": "gesture | finger | asl_letter | spell | joint | get_limits | rest",
        "gesture": "peace | i_love_you | ily | point | thumbs_up | fist | open_hands | ok_sign | rock_on | pinch | wiggle",
        "finger": "thumb | index | middle | ring | pinky",
        "side": "right | left | both",
        "degrees": "angle in degrees (90=open, 0=closed, thumb: 5=open, 140=closed)",
        "letter": "Single letter (A-Z) for Sign Language",
        "word": "Word to fingerspell in Sign Language (e.g. PETER, COW, ROBOT)"
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        arms = (ctx.extras or {}).get("arms")
        if arms is None:
            return "My arm actuation system is currently offline."

        # Joint Limits Query
        if args.get("action") == "get_limits":
            lines = [f"{j}: {info['min']}° to {info['max']}° ({info['desc']})" for j, info in JOINT_LIMITS_TABLE.items()]
            return "Physical Finger & Joint Limits:\n" + "\n".join(lines)

        text = str(args.get("text", "")).strip()
        cmd = parse_arm_command(text) if text else None

        # 1. Sign Language Single Letter (Both Hands)
        letter = str(args.get("letter", "")).upper()
        if (cmd and cmd.get("kind") == "asl_letter") or (args.get("action") == "asl_letter" and letter):
            target_let = cmd.get("letter", letter) if cmd else letter
            pose = get_bilateral_pose(target_let)
            if pose:
                arms._driver.send(pose)
                return f"Signing the letter {target_let} with both hands."
            return f"I don't have the sign for letter {target_let} calibrated yet."

        # 2. Sign Language Word Spelling: Spoken Letter Readout
        word = str(args.get("word", "")).upper()
        if (cmd and cmd.get("kind") == "spell") or (args.get("action") == "spell" and word):
            target_word = cmd.get("word", word) if cmd else word
            letters_spaced = " - ".join(list(target_word))

            def _spell_thread():
                for ch in target_word:
                    pose = get_bilateral_pose(ch)
                    if pose:
                        arms._driver.send(pose)
                        time.sleep(1.0)

            threading.Thread(target=_spell_thread, daemon=True).start()
            return f"Spelling {target_word}: {letters_spaced}."

        # 3. Hand & Finger Gestures (Peace, ILY, Point, Thumbs Up, Fist, OK, Pinch, Wiggle)
        g_name = (cmd.get("name", "") if cmd and cmd.get("kind") == "gesture" else str(args.get("gesture", ""))).lower()
        side = (cmd.get("side", "") if cmd else str(args.get("side", ""))) or "both"

        if g_name == "wiggle":
            def _wiggle_thread():
                for _ in range(2):
                    for f in ["thumb", "index", "middle", "ring", "pinky"]:
                        pose = {f"left_{f}p1_joint": 90.0 if f != "thumb" else 5.0, f"right_{f}p1_joint": 90.0 if f != "thumb" else 5.0}
                        arms._driver.send(pose)
                        time.sleep(0.12)
                    for f in ["thumb", "index", "middle", "ring", "pinky"]:
                        pose = {f"left_{f}p1_joint": 0.0 if f != "thumb" else 140.0, f"right_{f}p1_joint": 0.0 if f != "thumb" else 140.0}
                        arms._driver.send(pose)
                        time.sleep(0.12)
            threading.Thread(target=_wiggle_thread, daemon=True).start()
            return "Wiggling my fingers!"

        if g_name in FINGER_GESTURES:
            g_data = FINGER_GESTURES[g_name]
            pose = build_gesture_pose(g_data, side)
            arms._driver.send(pose)
            return g_data.get("spoken", f"Performing {g_name}.")

        # 4. Individual Finger Movement ("move thumb", "bend index", etc.)
        if (cmd and cmd.get("kind") == "finger") or args.get("action") == "finger" or args.get("finger"):
            finger = (cmd.get("finger") if cmd else args.get("finger", "thumb")).lower().replace(" finger", "").strip()
            deg = float(cmd.get("degrees", args.get("degrees", 0.0 if finger != "thumb" else 140.0)) if cmd else args.get("degrees", 0.0))
            f_side = (cmd.get("side") if cmd else args.get("side", "both")) or "both"
            sides = ("left", "right") if f_side in ("both", "all") else (f_side,)
            f_key = f"{finger}p1_joint" if "p1" not in finger else finger
            pose = {f"{s}_{f_key}": deg for s in sides}
            arms._driver.send(pose)
            return f"Moving {f_side} {finger} to {deg:.0f} degrees."

        # 5. Arm Gestures
        if g_name == "wave_right":
            return "Waving to you!"
        elif g_name == "punch":
            return "Striking forward with a punch!"
        elif g_name == "combat_guard":
            return "Holding defensive guard."
        elif g_name == "rest":
            pose = build_gesture_pose(FINGER_GESTURES["open_hands"], "both")
            arms._driver.send(pose)
            return "Resting my hands."

        return "Command received."
