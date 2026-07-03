"""Curiosity — self-directed learning that respects presence.

When no one is conversing ZERO learns SILENTLY: it reviews what it saw but
couldn't name, and what conversations left unresolved, and queues QUESTIONS.
It never talks to an empty room. When a known person is around, the proactive
policy surfaces one queued question at a time, opportunistically:

    "Earlier I saw something on the desk I couldn't identify — what was it?"

The queue is persistent (SQLite) so overnight curiosity survives a restart,
deduplicated by a source key so seeing the same unknown mug fifty times makes
one question, not fifty.
"""
from __future__ import annotations

import sqlite3
import time

from zero.utils.logging import get_logger

log = get_logger("curiosity")


class CuriosityStore:
    def __init__(self, path: str, max_pending: int = 20,
                 min_repeat_s: float = 6 * 3600.0):
        self.max_pending = int(max_pending)
        self.min_repeat_s = float(min_repeat_s)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS questions("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "source_key TEXT UNIQUE, text TEXT, priority REAL, "
            "person_id INTEGER, created_at REAL, asked_at REAL)"
        )
        self._db.commit()
        log.info("curiosity queue ready (%s, %d pending)", path, self.pending_count())

    def pending_count(self) -> int:
        return self._db.execute(
            "SELECT COUNT(*) FROM questions WHERE asked_at IS NULL").fetchone()[0]

    def add(self, source_key: str, text: str, priority: float = 5.0,
            person_id: int | None = None) -> bool:
        """Queue a question. Deduped by source_key; re-adding an already-asked
        question only revives it after ``min_repeat_s`` (don't nag)."""
        source_key = source_key.strip().lower()[:80]
        text = text.strip()[:200]
        if not source_key or not text:
            return False
        now = time.time()
        row = self._db.execute(
            "SELECT id, asked_at FROM questions WHERE source_key=?",
            (source_key,)).fetchone()
        if row:
            _id, asked_at = row
            if asked_at is None or now - asked_at < self.min_repeat_s:
                return False
            self._db.execute(
                "UPDATE questions SET text=?, priority=?, asked_at=NULL, "
                "created_at=? WHERE id=?", (text, priority, now, _id))
            self._db.commit()
            return True
        if self.pending_count() >= self.max_pending:
            # Full: drop the lowest-priority pending question to make room.
            self._db.execute(
                "DELETE FROM questions WHERE id IN (SELECT id FROM questions "
                "WHERE asked_at IS NULL ORDER BY priority ASC, id ASC LIMIT 1)")
        self._db.execute(
            "INSERT INTO questions(source_key, text, priority, person_id, "
            "created_at, asked_at) VALUES(?,?,?,?,?,NULL)",
            (source_key, text, priority, person_id, now))
        self._db.commit()
        log.info("curious about: %s", text)
        return True

    def note_unknown_object(self, hint: str) -> bool:
        """An unfamiliar object was sighted — queue a question about it."""
        hint = (hint or "something").strip().lower()[:60]
        return self.add(
            f"unknown-object:{hint}",
            f"Earlier I noticed {hint} around here that I couldn't quite "
            "place — what is it?",
            priority=4.0)

    def next_question(self, person_id: int | None = None):
        """Highest-priority pending question suitable for this person:
        (id, text) or None. Person-specific questions only go to their person."""
        row = self._db.execute(
            "SELECT id, text FROM questions WHERE asked_at IS NULL AND "
            "(person_id IS NULL OR person_id IS ?) "
            "ORDER BY priority DESC, id ASC LIMIT 1", (person_id,)).fetchone()
        return (int(row[0]), row[1]) if row else None

    def mark_asked(self, question_id: int) -> None:
        self._db.execute("UPDATE questions SET asked_at=? WHERE id=?",
                         (time.time(), question_id))
        self._db.commit()
