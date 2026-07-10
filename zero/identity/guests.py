"""GuestBook — provisional identities for people ZERO hasn't been introduced to.

Enrolled people live in the PersonRegistry with POSITIVE ids. Everyone else used
to collapse into a single anonymous bucket (person_id=None) — so two different
strangers in one room, and their speech data, got lumped together. That's wrong
both for memory and for training.

The GuestBook fixes it: each unfamiliar VOICE is clustered by its embedding into
a stable provisional guest addressed by a NEGATIVE id (-1, -2, ...), so it never
collides with a real person and any caller can tell them apart at a glance
(pid > 0 = known, pid < 0 = guest, None = unusable). Guests keep their own
training data; if one later gives a name, enrolment captures a fresh voice sample
so future turns match the real (positive) person instead.
"""
from __future__ import annotations

import sqlite3
import time

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("identity.guests")


def guest_label(guest_id: int) -> str:
    """'guest-1' for -1, etc. — a human-readable tag for logs and the corpus."""
    return f"guest-{abs(int(guest_id))}"


class GuestBook:
    def __init__(self, path: str, *, match_threshold: float = 0.55,
                 max_guests: int = 50, max_samples: int = 5):
        self.match_threshold = float(match_threshold)
        self.max_guests = max(1, int(max_guests))
        self.max_samples = max(1, int(max_samples))
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS guest_samples("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, guest INTEGER, dim INTEGER, "
            "vec BLOB, created_at REAL)")
        self._db.commit()
        log.info("guest book ready (%s, %d guests)", path, self.count())

    def count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(DISTINCT guest) FROM guest_samples").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def assign(self, voice_emb) -> int | None:
        """Stable NEGATIVE guest id for this voiceprint — matching an existing
        guest when close enough, else minting a new one and remembering it.
        None when the embedding is unusable."""
        v = self._unit(voice_emb)
        if v is None:
            return None
        best_guest, best_score = None, -1.0
        for guest, dim, blob in self._db.execute(
                "SELECT guest, dim, vec FROM guest_samples").fetchall():
            sample = np.frombuffer(blob, dtype=np.float32)
            if sample.size != v.size:
                continue  # a different voice model's dim — skip, never crash
            s = float(np.dot(sample, v))
            if s > best_score:
                best_guest, best_score = int(guest), s
        if best_guest is not None and best_score >= self.match_threshold:
            guest = best_guest
        else:
            guest = self._new_guest_id()
            log.info("new guest %s", guest_label(guest))
        self._add_sample(guest, v)
        return guest

    # ── internals ─────────────────────────────────────────────────────────────
    @staticmethod
    def _unit(vec) -> np.ndarray | None:
        if vec is None:
            return None
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-6:
            return None
        return v / n

    def _new_guest_id(self) -> int:
        row = self._db.execute("SELECT MIN(guest) FROM guest_samples").fetchone()
        lowest = int(row[0]) if row and row[0] is not None else 0
        return min(-1, lowest - 1)

    def _add_sample(self, guest: int, v: np.ndarray) -> None:
        self._db.execute(
            "INSERT INTO guest_samples(guest, dim, vec, created_at) VALUES(?,?,?,?)",
            (guest, v.size, v.tobytes(), time.time()))
        # Cap samples per guest (keep the newest) and total guests (drop the
        # least-recently-heard) so the book can't grow without bound.
        self._db.execute(
            "DELETE FROM guest_samples WHERE guest=? AND id NOT IN ("
            "SELECT id FROM guest_samples WHERE guest=? ORDER BY id DESC LIMIT ?)",
            (guest, guest, self.max_samples))
        guests = [r[0] for r in self._db.execute(
            "SELECT guest FROM guest_samples GROUP BY guest "
            "ORDER BY MAX(created_at) DESC").fetchall()]
        for stale in guests[self.max_guests:]:
            self._db.execute("DELETE FROM guest_samples WHERE guest=?", (stale,))
        self._db.commit()
