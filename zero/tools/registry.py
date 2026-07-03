"""ToolRegistry — the allow-listed set of actions ZERO may take."""
from __future__ import annotations

from zero.tools.base import Tool
from zero.utils.logging import get_logger

log = get_logger("tools")


class ToolRegistry:
    def __init__(self, allow: list[str] | None = None):
        self._tools: dict[str, Tool] = {}
        self._allow = set(allow) if allow is not None else None

    def register(self, tool: Tool) -> bool:
        """Adds the tool unless the allow-list excludes it. True if registered."""
        if self._allow is not None and tool.name not in self._allow:
            log.info("tool %r not on the allow-list — skipped", tool.name)
            return False
        self._tools[tool.name] = tool
        return True

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def spec_block(self) -> str:
        """The tools section for the system prompt: what exists and how to call
        it. Kept terse — every token here is paid on every turn."""
        if not self._tools:
            return ""
        lines = []
        for t in sorted(self._tools.values(), key=lambda t: t.name):
            params = ", ".join(f"{k} ({v})" for k, v in t.parameters.items())
            lines.append(f"- {t.name}: {t.description}"
                         + (f" Args: {params}" if params else ""))
        return (
            "You can take real actions with tools:\n" + "\n".join(lines) + "\n"
            'To use one, reply with ONLY this JSON on a single line and nothing '
            'else: {"tool": "<name>", "args": {...}}\n'
            "Only use a tool when the user asks for something a tool does. "
            "Otherwise reply normally."
        )
