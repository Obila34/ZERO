"""ZERO's motion arbiter — one writer, one clock, one e-stop for every joint.

The MotionBus owns the AF-1 gateway; producers (gaze, gestures, sign) write
setpoints onto prioritised tracks and the bus resolves + posts them each
tick. See zero/motion/bus.py for the model and the safety rules.
"""
from zero.motion.bus import MotionBus, BusJoint, TRACK_PRIORITY  # noqa: F401
