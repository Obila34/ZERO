"""Long-term memory — layered, relevance-retrieved, self-consolidating.

Human-shaped rather than notepad-shaped:

  * LAYERS   — semantic (facts about people/the world), episodic (what
               happened), procedural (how to behave: preferences).
  * RECALL   — activation-scored, Generative-Agents style:
                   activation = relevance × recency_decay × importance
               with a frequency nudge (memories recalled often come back
               easier) and reconsolidation on access (recalling refreshes).
  * WRITE    — every memory gets an importance (1-10) and an embedding for
               semantic search; falls back to keyword overlap when vectors
               can't be compared.
  * SLEEP    — consolidate() prunes memories whose activation has decayed
               below threshold (gentle forgetting) and, given a reflect
               function, synthesises higher-level insights from recent
               episodes ("David asks about the weather every morning").
  * PER-PERSON — memories can be keyed to a person_id from identity; recall
               blends that person's memories with global ones.

The public API main.py already uses (remember / facts / add_episode /
recent_episodes / as_block / forget_all / count / episode_count) is preserved;
rows from the old flat schema are migrated in on first open.
"""
from __future__ import annotations

import math
import sqlite3
import time

import numpy as np

from zero.memory.embeddings import keyword_overlap
from zero.utils.logging import get_logger

log = get_logger("memory")

LAYERS = ("semantic", "episodic", "procedural")
_DAY = 86400.0

# Reserved episodic key for the durable, never-forgotten "last time we spoke"
# record — one per person, updated in place, exempt from decay/age-out/caps so a
# returning person is always greeted with what you last discussed.
LAST_CONVO_KEY = "__last_convo__"


class SqliteMemory:
    def __init__(self, path: str, max_facts: int = 30, recent_episodes: int = 3,
                 max_stored_facts: int = 200, max_stored_episodes: int = 100,
                 embedder=None, top_k: int = 6, half_life_days: float = 14.0,
                 min_activation: float = 0.05, forget_max_age_days: float = 90.0,
                 forgetting: bool = True):
        self.max_facts = max_facts
        self.recent_episodes_n = recent_episodes
        self.max_stored_facts = max(max_facts, int(max_stored_facts))
        self.max_stored_episodes = max(recent_episodes, int(max_stored_episodes))
        self._embedder = embedder
        self.top_k = int(top_k)
        self.half_life_s = float(half_life_days) * _DAY
        self.min_activation = float(min_activation)
        self.forget_max_age_s = float(forget_max_age_days) * _DAY
        self.forgetting = bool(forgetting)

        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memories("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "person_id INTEGER, layer TEXT, key TEXT, value TEXT, "
            "importance REAL DEFAULT 5.0, emb BLOB, emb_dim INTEGER, "
            "created_at REAL, last_access REAL, access_count INTEGER DEFAULT 0, "
            "protected INTEGER DEFAULT 0)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_layer ON memories(layer, person_id)"
        )
        self._db.commit()
        self._ensure_column("protected", "INTEGER DEFAULT 0")  # older DBs
        self._migrate_legacy()
        log.info("memory store ready (%s, %d facts, %d episodes, embed=%s)",
                 path, self.count(), self.episode_count(),
                 getattr(embedder, "name", None) or "off")

    def _ensure_column(self, name: str, decl: str) -> None:
        """Add a column to an already-existing DB (no-op if it's already there)."""
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(memories)")}
        if name not in cols:
            self._db.execute(f"ALTER TABLE memories ADD COLUMN {name} {decl}")
            self._db.commit()
            log.info("migrated memory schema: added column %s", name)

    # ── legacy migration ─────────────────────────────────────────────────────
    def _migrate_legacy(self) -> None:
        """Import rows from the old flat schema (memory/episodes tables) once."""
        tables = {r[0] for r in self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "memory" in tables:
            rows = self._db.execute(
                "SELECT key, value, updated_at FROM memory").fetchall()
            for key, value, ts in rows:
                self._db.execute(
                    "INSERT INTO memories(person_id, layer, key, value, importance,"
                    " created_at, last_access) VALUES(NULL,'semantic',?,?,5.0,?,?)",
                    (key, value, ts, ts))
            self._db.execute("DROP TABLE memory")
            log.info("migrated %d legacy facts", len(rows))
        if "episodes" in tables:
            rows = self._db.execute(
                "SELECT summary, created_at FROM episodes").fetchall()
            for summary, ts in rows:
                self._db.execute(
                    "INSERT INTO memories(person_id, layer, key, value, importance,"
                    " created_at, last_access) VALUES(NULL,'episodic','episode',?,"
                    "4.0,?,?)", (summary, ts, ts))
            self._db.execute("DROP TABLE episodes")
            log.info("migrated %d legacy episodes", len(rows))
        self._db.commit()

    # ── internals ────────────────────────────────────────────────────────────
    def _embed(self, text: str):
        if self._embedder is None:
            return None, None
        try:
            v = self._embedder.embed(text)
        except Exception as e:  # embedding must never block a write
            log.debug("embed failed: %s", e)
            return None, None
        if v is None:
            return None, None
        return v.astype(np.float32).tobytes(), int(v.size)

    def _prune_layer(self, layer: str, cap: int) -> None:
        # Protected rows (the durable per-person last-conversation) are never
        # pruned by the storage cap.
        self._db.execute(
            "DELETE FROM memories WHERE layer=? AND protected=0 AND id NOT IN ("
            "SELECT id FROM memories WHERE layer=? "
            "ORDER BY last_access DESC LIMIT ?)",
            (layer, layer, cap))

    # ── writing ──────────────────────────────────────────────────────────────
    def remember(self, key: str, value: str, *, person_id: int | None = None,
                 importance: float = 5.0, layer: str = "semantic") -> None:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer: {layer}")
        key = key.strip().lower()[:60]
        value = value.strip()[:300]
        if not key or not value:
            return
        importance = float(min(10.0, max(1.0, importance)))
        emb, dim = self._embed(f"{key}: {value}")
        now = time.time()
        row = self._db.execute(
            "SELECT id FROM memories WHERE layer=? AND key=? AND "
            "person_id IS ? AND layer != 'episodic'",
            (layer, key, person_id)).fetchone()
        if row:  # update-in-place: same fact told again is reinforcement
            self._db.execute(
                "UPDATE memories SET value=?, importance=?, emb=?, emb_dim=?, "
                "last_access=?, access_count=access_count+1 WHERE id=?",
                (value, importance, emb, dim, now, row[0]))
        else:
            self._db.execute(
                "INSERT INTO memories(person_id, layer, key, value, importance, "
                "emb, emb_dim, created_at, last_access) VALUES(?,?,?,?,?,?,?,?,?)",
                (person_id, layer, key, value, importance, emb, dim, now, now))
        self._prune_layer("semantic", self.max_stored_facts)
        self._db.commit()
        log.info("remembered [%s%s]: %s -> %s", layer,
                 f"/p{person_id}" if person_id else "", key, value)

    def add_episode(self, summary: str, *, person_id: int | None = None,
                    importance: float = 4.0) -> None:
        summary = (summary or "").strip()[:300]
        if not summary:
            return
        emb, dim = self._embed(summary)
        now = time.time()
        self._db.execute(
            "INSERT INTO memories(person_id, layer, key, value, importance, emb,"
            " emb_dim, created_at, last_access) VALUES(?, 'episodic', 'episode',"
            " ?, ?, ?, ?, ?, ?)",
            (person_id, summary, float(importance), emb, dim, now, now))
        self._prune_layer("episodic", self.max_stored_episodes)
        self._db.commit()
        log.info("episode saved: %s", summary)

    def set_preference(self, text: str, *, person_id: int | None = None) -> None:
        """Behavioural preference ('speak slower', 'don't call me buddy') —
        procedural layer, high importance, injected into the system prompt."""
        self.remember(f"pref: {text.strip().lower()[:50]}", text,
                      person_id=person_id, importance=8.0, layer="procedural")

    def set_last_conversation(self, person_id: int | None, summary: str) -> None:
        """Durable 'last time we spoke' record — ONE per person, updated in place.
        Marked protected, so it survives decay, age-out and the episodic cap: a
        person returning after years is still greeted with what you last
        discussed. No-op without a person_id (anonymous chats aren't attributed)."""
        if person_id is None:
            return
        summary = (summary or "").strip()[:300]
        if not summary:
            return
        emb, dim = self._embed(summary)
        now = time.time()
        row = self._db.execute(
            "SELECT id FROM memories WHERE layer='episodic' AND key=? AND "
            "person_id IS ?", (LAST_CONVO_KEY, person_id)).fetchone()
        if row:
            self._db.execute(
                "UPDATE memories SET value=?, emb=?, emb_dim=?, created_at=?, "
                "last_access=?, importance=9.0, protected=1 WHERE id=?",
                (summary, emb, dim, now, now, row[0]))
        else:
            self._db.execute(
                "INSERT INTO memories(person_id, layer, key, value, importance, "
                "emb, emb_dim, created_at, last_access, protected) VALUES(?,"
                "'episodic',?,?,9.0,?,?,?,?,1)",
                (person_id, LAST_CONVO_KEY, summary, emb, dim, now, now))
        self._db.commit()
        log.info("last-conversation saved for person %s: %s", person_id, summary)

    def last_conversation(self, person_id: int | None) -> tuple[str, float] | None:
        """(summary, when_epoch) of the last conversation with this person, or
        None. Reconsolidates on access (recalling it refreshes recency)."""
        if person_id is None:
            return None
        row = self._db.execute(
            "SELECT id, value, created_at FROM memories WHERE layer='episodic' "
            "AND key=? AND person_id IS ?", (LAST_CONVO_KEY, person_id)).fetchone()
        if row is None:
            return None
        return row[1], row[2]

    # ── recall ───────────────────────────────────────────────────────────────
    def _activation(self, query_vec, query: str, row, now: float) -> float:
        (_id, _pid, _layer, key, value, importance, emb, dim,
         created, last_access, access_count) = row
        text = f"{key}: {value}" if key != "episode" else value
        if (query_vec is not None and emb is not None
                and dim == query_vec.size):
            relevance = float(np.dot(np.frombuffer(emb, dtype=np.float32),
                                     query_vec))
            relevance = max(0.0, relevance)
        elif query:
            relevance = keyword_overlap(query, text)
        else:
            relevance = 1.0  # no query: pure recency×importance ranking
        age = max(0.0, now - (last_access or created or now))
        recency = math.exp(-age / max(1.0, self.half_life_s))
        freq = 1.0 + 0.1 * math.log1p(access_count or 0)
        return relevance * (0.2 + 0.8 * recency) * (importance / 10.0) * freq

    def search(self, query: str, *, person_id: int | None = None,
               top_k: int | None = None, layer: str | None = None) -> list[str]:
        """Top memories for this moment: the person's own + global ones,
        activation-ranked. Accessing them reconsolidates (bumps recency)."""
        top_k = top_k or self.top_k
        qv = None
        if self._embedder is not None and query:
            try:
                qv = self._embedder.embed(query)
            except Exception:
                qv = None
        where, args = "WHERE (person_id IS NULL OR person_id IS ?)", [person_id]
        if layer:
            where += " AND layer=?"
            args.append(layer)
        rows = self._db.execute(
            f"SELECT id, person_id, layer, key, value, importance, emb, emb_dim,"
            f" created_at, last_access, access_count FROM memories {where}",
            args).fetchall()
        now = time.time()
        scored = [(self._activation(qv, query, r, now), r) for r in rows]
        scored = [(s, r) for s, r in scored if s >= self.min_activation]
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = scored[:top_k]
        if hits:  # reconsolidation: recalled memories strengthen
            self._db.executemany(
                "UPDATE memories SET last_access=?, access_count=access_count+1 "
                "WHERE id=?", [(now, r[0]) for _, r in hits])
            self._db.commit()
        out = []
        for _s, r in hits:
            key, value = r[3], r[4]
            out.append(value if key == "episode" else f"{key}: {value}")
        return out

    def relevant_block(self, query: str, *, person_id: int | None = None) -> str:
        """Per-turn ephemeral recall — what a human would 'think of' hearing
        this. '' when nothing relevant surfaces."""
        hits = self.search(query, person_id=person_id, top_k=4)
        if not hits:
            return ""
        return "; ".join(hits)

    # ── legacy-compatible reads ───────────────────────────────────────────────
    def count(self) -> int:
        return self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE layer='semantic'").fetchone()[0]

    def episode_count(self) -> int:
        return self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE layer='episodic'").fetchone()[0]

    def facts(self, person_id: int | None = None) -> dict[str, str]:
        rows = self._db.execute(
            "SELECT key, value FROM memories WHERE layer='semantic' AND "
            "(person_id IS NULL OR person_id IS ?) "
            "ORDER BY last_access DESC LIMIT ?",
            (person_id, self.max_facts)).fetchall()
        return {k: v for k, v in rows}

    def recent_episodes(self, n: int | None = None,
                        person_id: int | None = None) -> list[str]:
        """The last ``n`` conversation-arc summaries. Person-scoped when a
        person_id is given (so a returning person gets THEIR history, not
        whoever spoke last), global otherwise. The durable last-conversation
        record is excluded — it's surfaced separately as a welcome-back."""
        n = self.recent_episodes_n if n is None else n
        if n <= 0:
            return []
        where = "layer='episodic' AND key != ?"
        args: list = [LAST_CONVO_KEY]
        if person_id is not None:
            where += " AND person_id IS ?"
            args.append(person_id)
        args.append(n)
        rows = self._db.execute(
            f"SELECT value FROM memories WHERE {where} "
            "ORDER BY created_at DESC LIMIT ?", args).fetchall()
        return [r[0] for r in reversed(rows)]

    def preferences(self, person_id: int | None = None) -> list[str]:
        rows = self._db.execute(
            "SELECT value FROM memories WHERE layer='procedural' AND "
            "(person_id IS NULL OR person_id IS ?) "
            "ORDER BY last_access DESC LIMIT 8", (person_id,)).fetchall()
        return [r[0] for r in rows]

    def as_block(self, person_id: int | None = None) -> str:
        """Conversation-start injection: the most ACTIVE memories (not all of
        them), recent episode arcs, and behavioural preferences."""
        parts: list[str] = []
        top = self.search("", person_id=person_id, top_k=self.max_facts,
                          layer="semantic")
        if top:
            parts.append(
                "What you remember about the user from past chats (use it "
                "naturally, don't recite it back):\n"
                + "\n".join(f"- {t}" for t in top))
        episodes = self.recent_episodes(person_id=person_id)
        if episodes:
            parts.append(
                "Recent conversations you've had with them (so you can pick up "
                "where you left off):\n" + "\n".join(f"- {s}" for s in episodes))
        prefs = self.preferences(person_id)
        if prefs:
            parts.append(
                "How they've asked you to behave (always honour these):\n"
                + "\n".join(f"- {p}" for p in prefs))
        return "\n\n".join(parts)

    # ── sleep-phase consolidation ─────────────────────────────────────────────
    def consolidate(self, reflect_fn=None) -> dict:
        """Gentle forgetting + optional reflection. Returns stats for logging.

        Forgetting: memories that are old, unimportant AND never recalled decay
        away — interference-style, capped per pass so a bug can't wipe the
        store. Reflection: ``reflect_fn(episodes_text) -> list[str]`` distils
        recent episodes into higher-level semantic insights.
        """
        stats = {"forgotten": 0, "insights": 0}
        now = time.time()
        if self.forgetting:
            cutoff = now - self.forget_max_age_s
            rows = self._db.execute(
                "SELECT id FROM memories WHERE layer != 'procedural' AND "
                "protected = 0 AND importance < 6 AND access_count <= 1 AND "
                "last_access < ? ORDER BY last_access ASC LIMIT 10",
                (cutoff,)).fetchall()
            if rows:
                self._db.executemany("DELETE FROM memories WHERE id=?", rows)
                self._db.commit()
                stats["forgotten"] = len(rows)
                log.info("forgot %d faded memories", len(rows))
        if reflect_fn is not None:
            episodes = self.recent_episodes(8)
            if len(episodes) >= 3:
                try:
                    insights = reflect_fn("\n".join(episodes)) or []
                except Exception as e:
                    log.warning("reflection failed: %s", e)
                    insights = []
                for line in insights[:3]:
                    line = str(line).strip("-• ").strip()
                    if line:
                        self.remember(f"insight: {line[:50].lower()}", line,
                                      importance=7.0)
                        stats["insights"] += 1
        return stats

    # ── forgetting on demand ─────────────────────────────────────────────────
    def forget_all(self) -> None:
        self._db.execute("DELETE FROM memories")
        self._db.commit()
        log.info("memory cleared")

    def forget_last(self, person_id: int | None = None) -> str | None:
        """'Forget that' — drop the newest memory this person could mean
        (theirs or global, any layer). Returns what was forgotten, or None."""
        row = self._db.execute(
            "SELECT id, key, value FROM memories WHERE "
            "(person_id IS NULL OR person_id IS ?) "
            "ORDER BY created_at DESC LIMIT 1", (person_id,)).fetchone()
        if row is None:
            return None
        self._db.execute("DELETE FROM memories WHERE id=?", (row[0],))
        self._db.commit()
        what = row[2] if row[1] == "episode" else f"{row[1]}: {row[2]}"
        log.info("forgot on request: %s", what)
        return what

    def forget_person(self, person_id: int) -> int:
        """'Forget me' — delete every memory keyed to this person."""
        cur = self._db.execute(
            "DELETE FROM memories WHERE person_id IS ?", (person_id,))
        self._db.commit()
        log.info("forgot %d memories for person %s", cur.rowcount, person_id)
        return cur.rowcount
