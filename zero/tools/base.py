"""Tool interface — one action ZERO can take.

run() returns a short natural-language RESULT sentence; the router either
speaks it directly (tier-1) or hands it back to the LLM to phrase the final
reply (tier-2). Tools never raise out: errors come back as spoken apologies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolContext:
    """What a tool may touch. Deliberately narrow: tools get the memory store,
    the event bus and the current speaker — not the whole app."""
    memory: object | None = None       # SqliteMemory (or None)
    events: object | None = None       # EventBus (or None)
    person_id: int | None = None       # current speaker, if recognised
    person_name: str | None = None
    extras: dict = field(default_factory=dict)


class Tool(ABC):
    name: str = ""
    description: str = ""              # one line, shown to the LLM
    parameters: dict = {}              # {"arg": "what it is"} — shown to the LLM
    confirm_required: bool = False     # future: outward-facing actions
    # True = run() already returns the exact sentence to speak; the router
    # yields it verbatim instead of asking the LLM to rephrase. Set it on
    # tools whose wording is load-bearing — the arm tool's spelled-letter
    # readout is paced to the hands, and a paraphrase would break the sync.
    speaks_directly: bool = False

    @abstractmethod
    def run(self, args: dict, ctx: ToolContext) -> str:
        """Execute and return a short sentence describing the outcome."""

    def safe_run(self, args: dict, ctx: ToolContext) -> str:
        try:
            return self.run(args or {}, ctx)
        except Exception as e:  # a broken tool must never break the turn
            return f"Sorry — the {self.name} action failed ({e})."
