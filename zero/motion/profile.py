"""Minimum-jerk easing — the one motion profile every subsystem shares.

Natural reaching follows a bell-shaped velocity curve (Flash & Hogan 1985);
a constant-speed chase reads as machinery. ArmSystem discovered this the hard
way and grew its own inline copy; the sign engine and any future track need
the identical curve, so it lives here once.

Stdlib only — importing zero.motion must never pull in cv2/requests/numpy.
"""
from __future__ import annotations


def min_jerk(tau: float) -> float:
    """Position fraction at normalised time tau in [0, 1].

    10t^3 - 15t^4 + 6t^5: zero velocity AND zero acceleration at both ends,
    peak velocity 1.875 * distance / duration at the midpoint.
    """
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return 1.0
    return tau * tau * tau * (10.0 + tau * (-15.0 + 6.0 * tau))


# Peak-velocity factor of the minimum-jerk profile: v_max = PEAK * dist / dur.
# Used to stretch a segment's duration so no joint exceeds its speed cap —
# the move completes late rather than arriving half-made.
PEAK = 1.875


def stretch_for_speed(duration_s: float, distance_deg: float,
                      max_dps: float) -> float:
    """The duration this move actually needs so its min-jerk peak velocity
    stays under max_dps. Returns duration_s unchanged when already slow
    enough."""
    if max_dps <= 0.0:
        return duration_s
    return max(float(duration_s), PEAK * abs(distance_deg) / max_dps)
