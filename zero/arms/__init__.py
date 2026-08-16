"""ZERO's arm/hand subsystem — voice-commanded gestures over the AF1 gateway.

Safety model (mirrors the head's, stricter because arms are stronger):
  * a joint can ONLY be commanded if `arms.joints.<name>` in config carries a
    calibrated envelope (min/max/home) — an uncalibrated joint is inert;
  * stepper joints (160:1 geared, no encoders, zero = wherever the Nano
    booted) are additionally gated behind `arms.allow_steppers`;
  * every posted angle is clamped to the joint's envelope in the driver;
  * `arms.enabled: false` (default) builds nothing at all.
"""
from zero.arms.system import ArmSystem  # noqa: F401
