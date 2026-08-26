"""ExpressiveHands — lifecycle owner of the Living Hands layer.

Attaches the HandScheduler to the SpeechTap when (and only when) the
feature flag is on, and detaches cleanly on stop. Built by build_expr(),
which enforces the plan's gating: the layer requires the arms subsystem
(the hands ARE arm joints) and its own flag, and off means OFF — the tap
stays unattached and the speech path's two tap calls are no-op attribute
checks, bit-identical behavior to the pre-layer build.
"""
from __future__ import annotations

from zero.utils.logging import get_logger

log = get_logger("expr")


class ExpressiveHands:
    def __init__(self, cfg, bus, *, arms_provider=None, room_provider=None):
        from zero.expr.neural import build_neural
        from zero.expr.schedule import HandScheduler
        from zero.expr.tap import TAP

        self._neural = build_neural(cfg)     # None unless opted in
        self._sched = HandScheduler(cfg, bus, arms_provider=arms_provider,
                                    room_provider=room_provider,
                                    neural=self._neural)
        TAP.attach(self._sched)
        self._tap = TAP

    def status(self) -> dict:
        return {"attached": self._tap.attached, **self._sched.apex_stats()}

    def stop(self) -> None:
        try:
            self._tap.detach()
        except Exception:
            pass
        self._sched.stop()
        if self._neural is not None:
            try:
                self._neural.stop()
            except Exception:
                pass


def build_expr(cfg, *, arms_provider=None, room_provider=None):
    """The Living Hands layer, or None when gated off (the default).

    Gates: expression.hands.enabled AND arms.enabled — and it shares the
    MotionBus every other producer uses, writing only the priority-0 idle
    track. Failure to build must never touch startup (same contract as
    every optional subsystem)."""
    if not cfg.get("expression.hands.enabled", False):
        return None
    if not cfg.get("arms.enabled", False):
        log.info("expression.hands enabled but arms are not — layer off")
        return None
    try:
        from zero.motion.drivers import get_bus

        eh = ExpressiveHands(cfg, get_bus(cfg), arms_provider=arms_provider,
                             room_provider=room_provider)
        log.info("living hands up: beats%s, semantic %s, floor %s",
                 "", "on" if cfg.get("expression.hands.semantic.enabled", True)
                 else "off",
                 "on" if cfg.get("expression.hands.floor.enabled", False)
                 else "off (mic-bleed gate)")
        return eh
    except Exception as e:
        log.warning("living hands build failed — running without: %s", e)
        return None
