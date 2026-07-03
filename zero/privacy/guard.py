"""PrivacyGuard — bystander handling + the forget commands."""
from __future__ import annotations

import re
from dataclasses import dataclass

from zero.utils.logging import get_logger

log = get_logger("privacy")

MODES = ("open", "guarded", "strict")


@dataclass(frozen=True)
class Decision:
    respond: bool        # may ZERO answer this utterance at all?
    store_memory: bool   # may this conversation write long-term memories?


class PrivacyGuard:
    """Decides what an UNKNOWN speaker gets.

    open    — answered and remembered (single-user household default).
    guarded — answered, but nothing they say is stored long-term.
    strict  — strangers are not engaged: utterance dropped, logged once.
    Known people are always full-service.
    """

    def __init__(self, mode: str = "guarded"):
        mode = (mode or "guarded").lower()
        if mode not in MODES:
            raise ValueError(f"unknown privacy mode: {mode}")
        self.mode = mode
        self._warned = False

    def decide(self, ident) -> Decision:
        known = ident is not None and getattr(ident, "is_known", False)
        if known or self.mode == "open":
            return Decision(respond=True, store_memory=True)
        if self.mode == "guarded":
            return Decision(respond=True, store_memory=False)
        if not self._warned:
            log.info("strict mode: ignoring unknown speaker "
                     "(logged once per session)")
            self._warned = True
        return Decision(respond=False, store_memory=False)


# ── forget commands ───────────────────────────────────────────────────────────
_FORGET_ME = re.compile(
    r"\bforget\s+(?:everything\s+)?about\s+me\b|\bforget\s+me\b", re.IGNORECASE)
_FORGET_THAT = re.compile(r"\bforget\s+(?:that|it|what\s+i\s+(?:just\s+)?said)\b",
                          re.IGNORECASE)


def parse_forget_command(text: str) -> str | None:
    """'me' (erase this person), 'that' (drop the latest memory), or None."""
    t = text.strip()
    if len(t.split()) > 8:
        return None
    if _FORGET_ME.search(t):
        return "me"
    if _FORGET_THAT.search(t):
        return "that"
    return None
