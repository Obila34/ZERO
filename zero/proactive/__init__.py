"""Proactive — ZERO speaking without being spoken to, within guardrails.

curiosity.py  — what ZERO wants to ask (queued while alone, asked when a
                known person is around).
policy.py     — WHEN it may speak: the hard part is staying quiet.
triggers.py   — the background watcher that turns presence + timers +
                curiosity into Events on the bus.
"""
from zero.proactive.curiosity import CuriosityStore
from zero.proactive.policy import InteractionPolicy
from zero.proactive.triggers import TriggerSource

__all__ = ["CuriosityStore", "InteractionPolicy", "TriggerSource"]
