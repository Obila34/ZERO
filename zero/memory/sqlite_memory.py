"""Long-term memory — a tiny SQLite store of durable facts about the user.

Deliberately simple: one file on disk (survives reboots), no server, ships with
Python. We store FACTS (key -> value), not whole transcripts, so what gets folded
back into the prompt stays small and doesn't slow the model down.

Facts get in two ways (see main.py):
  * explicit  — the user says "remember that ..."
  * automatic — at the end of a conversation the LLM extracts durable facts.

They come out as a short block injected into the system prompt at the start of the
next conversation, so ZERO recalls things across sessions.
"""
from __future__ import annotations

import sqlite3
import time

from zero.utils.logging import get_logger

log = get_logger("memory")


class SqliteMemory:
    def __init__(self, path: str, max_facts: int = 30):
        self.max_facts = max_facts
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memory("
            "key TEXT PRIMARY KEY, value TEXT, updated_at REAL)"
        )
        self._db.commit()
        log.info("memory store ready (%s, %d facts)", path, self.count())

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM memory").fetchone()[0]

    def remember(self, key: str, value: str) -> None:
        key = key.strip().lower()[:60]
        value = value.strip()[:200]
        if not key or not value:
            return
        self._db.execute(
            "INSERT INTO memory(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, value, time.time()),
        )
        self._db.commit()
        log.info("remembered: %s -> %s", key, value)

    def facts(self) -> dict[str, str]:
        rows = self._db.execute(
            "SELECT key, value FROM memory ORDER BY updated_at DESC LIMIT ?",
            (self.max_facts,),
        ).fetchall()
        return {k: v for k, v in rows}

    def as_block(self) -> str:
        """Compact block for the system prompt; empty string if nothing known."""
        facts = self.facts()
        if not facts:
            return ""
        lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        return (
            "What you remember about the user from past chats (use it naturally, "
            "don't recite it back):\n" + lines
        )

    def forget_all(self) -> None:
        self._db.execute("DELETE FROM memory")
        self._db.commit()
        log.info("memory cleared")
