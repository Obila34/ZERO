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
    def __init__(self, path: str, max_facts: int = 30, recent_episodes: int = 3,
                 max_stored_facts: int = 200, max_stored_episodes: int = 100):
        self.max_facts = max_facts
        self.recent_episodes_n = recent_episodes
        # Hard caps on what's STORED (not just injected) — without them the tables
        # grow forever: every "remember that ..." creates a unique timestamped key
        # and every conversation adds an episode. Oldest entries are pruned first.
        self.max_stored_facts = max(max_facts, int(max_stored_facts))
        self.max_stored_episodes = max(recent_episodes, int(max_stored_episodes))
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memory("
            "key TEXT PRIMARY KEY, value TEXT, updated_at REAL)"
        )
        # Episodic memory: one short summary per past conversation, so ZERO recalls
        # the ARC of what you've talked about, not just isolated key/value facts.
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS episodes("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, summary TEXT, created_at REAL)"
        )
        self._db.commit()
        log.info("memory store ready (%s, %d facts, %d episodes)",
                 path, self.count(), self.episode_count())

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM memory").fetchone()[0]

    def episode_count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

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
        # Prune the least-recently-updated facts past the storage cap.
        self._db.execute(
            "DELETE FROM memory WHERE key NOT IN ("
            "SELECT key FROM memory ORDER BY updated_at DESC LIMIT ?)",
            (self.max_stored_facts,),
        )
        self._db.commit()
        log.info("remembered: %s -> %s", key, value)

    def facts(self) -> dict[str, str]:
        rows = self._db.execute(
            "SELECT key, value FROM memory ORDER BY updated_at DESC LIMIT ?",
            (self.max_facts,),
        ).fetchall()
        return {k: v for k, v in rows}

    def add_episode(self, summary: str) -> None:
        """Store a one-line summary of a finished conversation."""
        summary = (summary or "").strip()[:300]
        if not summary:
            return
        self._db.execute(
            "INSERT INTO episodes(summary, created_at) VALUES(?, ?)",
            (summary, time.time()),
        )
        # Prune the oldest episodes past the storage cap.
        self._db.execute(
            "DELETE FROM episodes WHERE id NOT IN ("
            "SELECT id FROM episodes ORDER BY id DESC LIMIT ?)",
            (self.max_stored_episodes,),
        )
        self._db.commit()
        log.info("episode saved: %s", summary)

    def recent_episodes(self, n: int | None = None) -> list[str]:
        """The most recent conversation summaries, oldest-first for readability."""
        n = self.recent_episodes_n if n is None else n
        if n <= 0:
            return []
        rows = self._db.execute(
            "SELECT summary FROM episodes ORDER BY created_at DESC LIMIT ?", (n,)
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def as_block(self) -> str:
        """Compact block for the system prompt; empty string if nothing known."""
        parts: list[str] = []
        facts = self.facts()
        if facts:
            lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
            parts.append(
                "What you remember about the user from past chats (use it "
                "naturally, don't recite it back):\n" + lines
            )
        episodes = self.recent_episodes()
        if episodes:
            lines = "\n".join(f"- {s}" for s in episodes)
            parts.append(
                "Recent conversations you've had with them (so you can pick up "
                "where you left off):\n" + lines
            )
        return "\n\n".join(parts)

    def forget_all(self) -> None:
        self._db.execute("DELETE FROM memory")
        self._db.execute("DELETE FROM episodes")
        self._db.commit()
        log.info("memory cleared")
