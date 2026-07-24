"""The experience spine — ONE episode schema for everything ZERO lives through.

Every notable experience becomes one row: a conversation turn, a surprising
scene event, a proactive attempt, and — when ZERO grows limbs — an action.
Two scalar tags ride on every episode:

  * ``reward``   (-1..1)  how well it went (dopamine; zero/learning/reward.py)
  * ``surprise`` (0..∞)   how unexpected it was (prediction error;
                          zero/world/surprise.py)

Consumers: the nightly consolidation (scripts/consolidate.py) distills
episodes into long-term memory with reward/surprise-weighted importance and
into reward-weighted training data, then marks them consolidated.

Schema versioning — this schema must outlive today's build:
``PRAGMA user_version`` holds the schema version; ``_MIGRATIONS[i]`` upgrades
version i -> i+1 and runs exactly once, in order, inside one transaction per
step. Adding a field = append a migration; old rows keep working because
payloads are JSON and readers use ``.get()``. Rows also record the schema
version they were written under (``v``).

Payload conventions by kind (all optional, JSON):
  turn      {"user": str, "reply": str, "affect": {...}, "barged_in": bool}
  scene     {"event": str, "label": str}
  proactive {"pkind": str, "text": str}
  action    {"observation": ..., "action": ..., "result": ...}   (future)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from zero.utils.logging import get_logger

log = get_logger("learning.episodes")

SCHEMA_VERSION = 1

_MIGRATIONS = [
    # 0 -> 1: initial schema.
    [
        "CREATE TABLE IF NOT EXISTS episodes("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " v INTEGER NOT NULL,"                 # schema version at write time
        " ts REAL NOT NULL,"
        " kind TEXT NOT NULL,"                 # turn|scene|proactive|action
        " person_id INTEGER,"
        " payload TEXT NOT NULL DEFAULT '{}',"
        " reward REAL,"                        # NULL = untagged
        " surprise REAL,"
        " consolidated_at REAL)",
        "CREATE INDEX IF NOT EXISTS ep_kind_ts ON episodes(kind, ts)",
        "CREATE INDEX IF NOT EXISTS ep_unconsolidated"
        " ON episodes(consolidated_at) WHERE consolidated_at IS NULL",
    ],
]


class EpisodeStore:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")  # writers never block reads
        self._lock = threading.Lock()
        self._migrate()
        log.info("episode store ready (%s, v%d, %d rows)", path,
                 SCHEMA_VERSION, self.count())

    def _migrate(self) -> None:
        cur = self._db.execute("PRAGMA user_version").fetchone()[0]
        for version in range(cur, len(_MIGRATIONS)):
            with self._db:                     # one transaction per step
                for stmt in _MIGRATIONS[version]:
                    self._db.execute(stmt)
                self._db.execute(f"PRAGMA user_version = {version + 1}")
            log.info("episodes schema migrated v%d -> v%d", version, version + 1)

    # ── writes (hot path: single insert, must stay ~ms) ──────────────────────
    def add(self, kind: str, payload: dict | None = None, *,
            reward: float | None = None, surprise: float | None = None,
            person_id: int | None = None, ts: float | None = None) -> int:
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO episodes(v, ts, kind, person_id, payload, reward,"
                " surprise) VALUES(?,?,?,?,?,?,?)",
                (SCHEMA_VERSION, ts or time.time(), kind, person_id,
                 json.dumps(payload or {}, ensure_ascii=False),
                 reward, surprise))
            return int(cur.lastrowid)

    def tag_reward(self, episode_id: int, reward: float) -> None:
        """Retro-tag: the verdict on a turn often arrives one utterance LATER
        ("no, that's wrong") — this writes it back onto the episode it judges."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE episodes SET reward=? WHERE id=?",
                (max(-1.0, min(1.0, float(reward))), int(episode_id)))

    # ── reads (consolidation + tests) ────────────────────────────────────────
    def unconsolidated(self, kinds: tuple = ("turn", "scene", "proactive"),
                       before: float | None = None) -> list[dict]:
        q = ("SELECT id, v, ts, kind, person_id, payload, reward, surprise"
             " FROM episodes WHERE consolidated_at IS NULL"
             f" AND kind IN ({','.join('?' * len(kinds))})")
        args: list = list(kinds)
        if before is not None:
            q += " AND ts < ?"
            args.append(before)
        rows = self._db.execute(q + " ORDER BY ts", args).fetchall()
        return [self._row(r) for r in rows]

    def mark_consolidated(self, ids: list[int],
                          ts: float | None = None) -> None:
        if not ids:
            return
        with self._lock, self._db:
            self._db.executemany(
                "UPDATE episodes SET consolidated_at=? WHERE id=?",
                [(ts or time.time(), int(i)) for i in ids])

    def recent(self, kind: str | None = None, within_s: float = 86400.0,
               limit: int = 500) -> list[dict]:
        since = time.time() - within_s
        if kind:
            rows = self._db.execute(
                "SELECT id, v, ts, kind, person_id, payload, reward, surprise"
                " FROM episodes WHERE ts >= ? AND kind = ?"
                " ORDER BY ts DESC LIMIT ?", (since, kind, limit)).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, v, ts, kind, person_id, payload, reward, surprise"
                " FROM episodes WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                (since, limit)).fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    @staticmethod
    def _row(r) -> dict:
        try:
            payload = json.loads(r[5]) if r[5] else {}
        except ValueError:
            payload = {}
        return {"id": r[0], "v": r[1], "ts": r[2], "kind": r[3],
                "person_id": r[4], "payload": payload,
                "reward": r[6], "surprise": r[7]}
