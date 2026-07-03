"""Tools — the hands. Lets ZERO *act* (timers, reminders, time, memory),
not just talk. Routed in three tiers: regex intents (instant), LLM JSON tool
calls (flexible), plain chat (everything else)."""
from zero.tools.base import Tool, ToolContext
from zero.tools.registry import ToolRegistry
from zero.tools.router import ToolAwareLLM

__all__ = ["Tool", "ToolContext", "ToolRegistry", "ToolAwareLLM"]
