"""PersonRegistry — the SQLite store of enrolled people.

One row per person; N embedding samples per (person, kind) where kind is
"face" or "voice". Matching scans all samples of a kind and returns the best
cosine score — max-over-samples is robust to pose/tone variation without any
averaging tricks. Embeddings are stored as raw float32 bytes.
"""
from __future__ import annotations

import sqlite3
import time

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("identity")

KINDS = ("face", "voice")


class PersonRegistry:
    def __init__(self, path: str, max_samples_per_kind: int = 5):
        self.max_samples = max(1, int(max_samples_per_kind))
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS people("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT UNIQUE COLLATE NOCASE, created_at REAL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, "
            "kind TEXT, dim INTEGER, vec BLOB, created_at REAL, "
            "FOREIGN KEY(person_id) REFERENCES people(id))"
        )
        self._db.commit()
        log.info("person registry ready (%s, %d people)", path, self.count())

    # ── people ───────────────────────────────────────────────────────────────
    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM people").fetchone()[0]

    def people(self) -> list[tuple[int, str]]:
        return self._db.execute(
            "SELECT id, name FROM people ORDER BY name COLLATE NOCASE"
        ).fetchall()

    def name_of(self, person_id: int) -> str | None:
        row = self._db.execute(
            "SELECT name FROM people WHERE id=?", (person_id,)
        ).fetchone()
        return row[0] if row else None

    def enroll(self, name: str) -> int:
        """Get-or-create a person by name (case-insensitive). Returns id."""
        name = " ".join(name.strip().split())[:60]
        if not name:
            raise ValueError("empty name")
        row = self._db.execute(
            "SELECT id FROM people WHERE name=? COLLATE NOCASE", (name,)
        ).fetchone()
        if row:
            return int(row[0])
        cur = self._db.execute(
            "INSERT INTO people(name, created_at) VALUES(?, ?)",
            (name, time.time()),
        )
        self._db.commit()
        log.info("enrolled new person: %s (id=%d)", name, cur.lastrowid)
        return int(cur.lastrowid)

    def forget(self, name: str) -> bool:
        """Delete a person and all their embeddings ("forget me"). True if found."""
        row = self._db.execute(
            "SELECT id FROM people WHERE name=? COLLATE NOCASE", (name.strip(),)
        ).fetchone()
        if not row:
            return False
        pid = int(row[0])
        self._db.execute("DELETE FROM embeddings WHERE person_id=?", (pid,))
        self._db.execute("DELETE FROM people WHERE id=?", (pid,))
        self._db.commit()
        log.info("forgot person: %s (id=%d)", name, pid)
        return True

    # ── embeddings ───────────────────────────────────────────────────────────
    def add_embedding(self, person_id: int, kind: str, vec: np.ndarray) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown embedding kind: {kind}")
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-6:
            log.warning("refusing degenerate %s embedding for person %d", kind, person_id)
            return
        v = v / n
        self._db.execute(
            "INSERT INTO embeddings(person_id, kind, dim, vec, created_at) "
            "VALUES(?,?,?,?,?)",
            (person_id, kind, v.size, v.tobytes(), time.time()),
        )
        # Cap samples per (person, kind): keep the newest.
        self._db.execute(
            "DELETE FROM embeddings WHERE person_id=? AND kind=? AND id NOT IN ("
            "SELECT id FROM embeddings WHERE person_id=? AND kind=? "
            "ORDER BY id DESC LIMIT ?)",
            (person_id, kind, person_id, kind, self.max_samples),
        )
        self._db.commit()

    def sample_count(self, person_id: int, kind: str) -> int:
        return self._db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE person_id=? AND kind=?",
            (person_id, kind),
        ).fetchone()[0]

    def match(self, kind: str, vec: np.ndarray) -> tuple[int, str, float] | None:
        """Best (person_id, name, cosine) for this embedding, or None if the
        registry has no samples of this kind. Max over samples per person."""
        if kind not in KINDS:
            raise ValueError(f"unknown embedding kind: {kind}")
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-6:
            return None
        v = v / n
        rows = self._db.execute(
            "SELECT e.person_id, p.name, e.dim, e.vec FROM embeddings e "
            "JOIN people p ON p.id = e.person_id WHERE e.kind=?", (kind,)
        ).fetchall()
        best: tuple[int, str, float] | None = None
        for pid, name, dim, blob in rows:
            sample = np.frombuffer(blob, dtype=np.float32)
            if sample.size != v.size:
                continue  # a different model's dim — skip, never crash
            s = float(np.dot(sample, v))
            if best is None or s > best[2]:
                best = (int(pid), name, s)
        return best
