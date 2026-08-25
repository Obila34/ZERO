"""Living Hands — the additive co-speech hand/wrist expression layer.

Prosody-timed beats, semantic hand shapes and a micro-motion floor, driven
from taps on the existing speech pipeline and rendered ONLY onto the
MotionBus idle track (priority 0), which every existing behavior — sign,
commands, gestures, gaze — preempts by fixed arithmetic. Architecture and
research grounding: docs/LIVING_HANDS_PLAN.md.
"""
from zero.expr.system import ExpressiveHands, build_expr  # noqa: F401
