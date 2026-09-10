#!/usr/bin/env python3
"""Bake a ZERO joint-stream (rig_export.py output) onto the AF1 Blender rig
and render it — the sim stage's stage 1. Runs ON zerolabs0:

    blender af1_urdf_rig.blend --background --python af1_zero_replay.py -- \
        --anim rig_anim_phys.json --out renders/phys.mp4 --label PHYSICAL

Bone/channel conventions copied from af1_animate.py (chan 0 for X hinge
else 1, hinge_sign applied); hands via CTRL_hand_{R,L} curl_<finger>
custom props (0..1). Renders workbench solid — motion truth, not beauty.
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))


def argv_after_ddash():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def main():
    args = argv_after_ddash()
    opt = {}
    k = None
    for a in args:
        if a.startswith("--"):
            k = a[2:]
            opt[k] = True
        elif k:
            opt[k] = a
            k = None
    anim = json.load(open(opt["anim"]))
    out = opt.get("out", "zero_replay.mp4")
    label = str(opt.get("label", anim.get("stream", "")))

    jmap = json.load(open(os.path.join(HERE, "af1_joint_map.json")))["joints"]
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")

    fps = int(anim.get("fps", 30))
    scene = bpy.context.scene
    scene.render.fps = fps
    dur = float(anim["duration_s"])
    scene.frame_start = 1
    scene.frame_end = max(2, int(dur * fps))

    for dof, keys in anim["dof"].items():
        if dof.startswith("curl:"):
            _, side, finger = dof.split(":")
            ctrl = rig.pose.bones[
                f"CTRL_hand_{'R' if side == 'right' else 'L'}"]
            prop = f"curl_{finger}"
            if prop not in ctrl:
                continue
            for t, v in keys:
                ctrl[prop] = float(v)
                ctrl.keyframe_insert(data_path=f'["{prop}"]',
                                     frame=1 + int(t * fps))
            continue
        meta = jmap.get(dof)
        if meta is None or dof not in rig.pose.bones:
            print(f"[zero-replay] no rig DOF {dof!r} — skipped")
            continue
        pb = rig.pose.bones[dof]
        pb.rotation_mode = "XYZ"
        chan = 0 if meta["hinge_axis"] == "X" else 1
        for t, v in keys:
            pb.rotation_euler[chan] = meta["hinge_sign"] * math.radians(v)
            pb.keyframe_insert(data_path="rotation_euler", index=chan,
                               frame=1 + int(t * fps))

    # Framing: reuse the rig's own render pipeline (CAM_sign aimed by its
    # signing-volume math) — an improvised camera framed the pedestal.
    try:
        sys.path.insert(0, HERE)
        import af1_render as R  # type: ignore
        import af1_kinematics as K  # type: ignore
        R.setup_shot("hero", (960, 720))
        # widen DOWNWARD: the rig's shots assume signing-height hands, but
        # this replay shows under-driven arms drooping low — the very
        # defect under audit must stay in frame
        from mathutils import Vector  # type: ignore
        rig_o = bpy.data.objects["AF1_RIG"]
        lo, hi = R.signing_volume(K.build_model(), rig_o)
        lo = Vector((lo.x, lo.y, lo.z - 0.45))
        R.frame_box(bpy.data.objects["CAM_sign"], lo, hi, 50.0,
                    (960, 720), yaw=-11, pitch=3)
    except Exception as e:
        print(f"[zero-replay] setup_shot failed ({e}) — improvised camera")
        bpy.ops.object.camera_add(location=(0.0, -2.2, 1.2),
                                  rotation=(math.radians(80), 0, 0))
        scene.camera = bpy.context.object
    if not any(o.type == "LIGHT" for o in bpy.data.objects):
        bpy.ops.object.light_add(type="SUN", location=(1, -2, 3))

    scene.render.engine = ("BLENDER_WORKBENCH"
                           if hasattr(bpy.types, "RenderSettings") else
                           "BLENDER_WORKBENCH")
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    # headless builds may lack FFMPEG output — render PNG frames; the
    # caller assembles the mp4 with ffmpeg
    scene.render.image_settings.file_format = "PNG"
    os.makedirs(os.path.abspath(out), exist_ok=True)
    scene.render.filepath = os.path.join(os.path.abspath(out), "f_")
    if label:
        scene.render.use_stamp = True
        scene.render.stamp_note_text = f"ZERO replay: {label}"
        scene.render.use_stamp_note = True
    print(f"[zero-replay] rendering {scene.frame_end} frames -> {out}")
    bpy.ops.render.render(animation=True)
    print("[zero-replay] done")


main()
