"""Human-like memory: layers, activation recall, migration, consolidation."""
from __future__ import annotations

import sqlite3
import time

import pytest

from zero.memory.embeddings import HashEmbedder, build_embedder, keyword_overlap
from zero.memory.sqlite_memory import SqliteMemory


def _mem(tmp_path, **kw):
    kw.setdefault("embedder", HashEmbedder(128))
    return SqliteMemory(str(tmp_path / "mem.sqlite"), **kw)


# ── legacy API stays intact ──────────────────────────────────────────────────
def test_legacy_api_roundtrip(tmp_path):
    m = _mem(tmp_path)
    m.remember("home", "Nairobi")
    m.add_episode("We talked about the weather.")
    assert m.count() == 1 and m.episode_count() == 1
    assert m.facts() == {"home": "Nairobi"}
    assert m.recent_episodes() == ["We talked about the weather."]
    block = m.as_block()
    assert "home: Nairobi" in block and "weather" in block
    m.forget_all()
    assert m.count() == 0 and m.as_block() == ""


def test_legacy_schema_migrates(tmp_path):
    path = str(tmp_path / "old.sqlite")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE memory(key TEXT PRIMARY KEY, value TEXT, updated_at REAL)")
    db.execute("CREATE TABLE episodes(id INTEGER PRIMARY KEY AUTOINCREMENT,"
               " summary TEXT, created_at REAL)")
    db.execute("INSERT INTO memory VALUES('name','David',?)", (time.time(),))
    db.execute("INSERT INTO episodes(summary, created_at) VALUES('First chat.',?)",
               (time.time(),))
    db.commit()
    db.close()
    m = SqliteMemory(path, embedder=HashEmbedder(128))
    assert m.facts() == {"name": "David"}
    assert m.recent_episodes() == ["First chat."]
    # Re-opening doesn't double-import.
    m2 = SqliteMemory(path, embedder=HashEmbedder(128))
    assert m2.count() == 1


# ── activation-ranked recall ─────────────────────────────────────────────────
def test_search_relevance_ranks(tmp_path):
    m = _mem(tmp_path)
    m.remember("wifi password", "the wifi password is hunter2")
    m.remember("sister birthday", "her sister's birthday is in June")
    m.remember("favorite tea", "loves strong black tea")
    hits = m.search("what's the wifi network password")
    assert hits and "hunter2" in hits[0]


def test_importance_breaks_ties(tmp_path):
    m = _mem(tmp_path, embedder=None)   # no embeddings: keyword path
    m.remember("project alpha", "working on project alpha deadline", importance=2.0)
    m.remember("project beta", "working on project beta deadline", importance=9.0)
    hits = m.search("how is the project deadline going", top_k=2)
    assert len(hits) == 2 and "beta" in hits[0]


def test_recall_reconsolidates(tmp_path):
    m = _mem(tmp_path)
    m.remember("coffee", "takes coffee black, no sugar")
    row = m._db.execute("SELECT access_count FROM memories").fetchone()
    assert row[0] == 0
    m.search("how do they take their coffee")
    row = m._db.execute("SELECT access_count FROM memories").fetchone()
    assert row[0] == 1


def test_person_scoping(tmp_path):
    m = _mem(tmp_path)
    m.remember("drink", "David drinks tea", person_id=1)
    m.remember("drink", "Jane drinks coffee", person_id=2)
    m.remember("house rule", "shoes off at the door")   # global
    david = m.search("what do they drink", person_id=1)
    assert any("tea" in h for h in david)
    assert not any("coffee" in h for h in david)        # Jane's stays hers
    assert any("shoes" in h for h in m.search("house rules", person_id=1))


def test_relevant_block_empty_safe(tmp_path):
    m = _mem(tmp_path)
    assert m.relevant_block("anything at all") == ""


def test_same_fact_reinforces_not_duplicates(tmp_path):
    m = _mem(tmp_path)
    m.remember("home", "Nairobi")
    m.remember("home", "Nairobi, Kenya")
    assert m.count() == 1
    assert m.facts()["home"] == "Nairobi, Kenya"


# ── preferences (procedural layer) ───────────────────────────────────────────
def test_preferences_layer(tmp_path):
    m = _mem(tmp_path)
    m.set_preference("speak more slowly", person_id=1)
    assert m.preferences(1) == ["speak more slowly"]
    assert "always honour" in m.as_block(1)
    assert m.count() == 0                      # not a semantic fact


# ── consolidation: forgetting + reflection ───────────────────────────────────
def test_forgetting_prunes_faded_only(tmp_path):
    m = _mem(tmp_path, forget_max_age_days=1.0)
    old = time.time() - 10 * 86400
    m.remember("trivia", "mentioned it rained once", importance=2.0)
    m.remember("name", "their name is David", importance=10.0)
    m._db.execute("UPDATE memories SET last_access=?, created_at=?", (old, old))
    m._db.commit()
    stats = m.consolidate()
    assert stats["forgotten"] == 1
    assert list(m.facts()) == ["name"]         # important memory survived


def test_reflection_stores_insights(tmp_path):
    m = _mem(tmp_path)
    for i in range(3):
        m.add_episode(f"Talked about the garden, day {i}.")
    stats = m.consolidate(
        reflect_fn=lambda text: ["The user is renovating their garden."])
    assert stats["insights"] == 1
    assert any("garden" in v.lower() for v in m.facts().values())


def test_reflection_failure_is_contained(tmp_path):
    m = _mem(tmp_path)
    for i in range(3):
        m.add_episode(f"chat {i}")

    def boom(text):
        raise RuntimeError("llm down")

    stats = m.consolidate(reflect_fn=boom)
    assert stats["insights"] == 0              # no crash


def test_forget_person(tmp_path):
    m = _mem(tmp_path)
    m.remember("drink", "tea", person_id=1)
    m.remember("drink", "coffee", person_id=2)
    assert m.forget_person(1) == 1
    assert m.search("drink", person_id=2)


# ── durable per-person last conversation (never forgotten) ───────────────────
def test_last_conversation_upserts_one_per_person(tmp_path):
    m = _mem(tmp_path)
    m.set_last_conversation(1, "We planned the garden.")
    m.set_last_conversation(1, "We planned the garden and picked tomatoes.")
    got = m.last_conversation(1)
    assert got is not None and "tomatoes" in got[0]
    n = m._db.execute(
        "SELECT COUNT(*) FROM memories WHERE key='__last_convo__' AND person_id=1"
    ).fetchone()[0]
    assert n == 1                                   # updated in place, not duplicated
    assert m.last_conversation(2) is None
    m.set_last_conversation(None, "anon")           # anonymous chats: no-op
    assert m._db.execute(
        "SELECT COUNT(*) FROM memories WHERE key='__last_convo__' "
        "AND person_id IS NULL").fetchone()[0] == 0


def test_last_conversation_survives_forgetting_and_cap(tmp_path):
    m = _mem(tmp_path, forget_max_age_days=1.0, max_stored_episodes=3)
    m.set_last_conversation(7, "The important thing we discussed.")
    # Age it far past the forgetting cutoff...
    old = time.time() - 100 * 86400
    m._db.execute("UPDATE memories SET last_access=?, created_at=?", (old, old))
    m._db.commit()
    # ...and pile on episodes to trip the episodic storage cap.
    for i in range(6):
        m.add_episode(f"filler episode {i}", person_id=7)
    m.consolidate()
    assert m.last_conversation(7) is not None        # protected: never pruned/faded
    assert "important" in m.last_conversation(7)[0]


def test_recent_episodes_person_scoped_excludes_last_convo(tmp_path):
    m = _mem(tmp_path)
    m.add_episode("David and I talked gardening.", person_id=1)
    m.add_episode("Jane and I talked cars.", person_id=2)
    m.set_last_conversation(1, "David's durable last chat.")
    davids = m.recent_episodes(person_id=1)
    assert any("gardening" in e for e in davids)
    assert not any("cars" in e for e in davids)             # Jane's stays hers
    assert not any("durable last chat" in e for e in davids)  # reserved row excluded
    # Global recall still works and also excludes the reserved row.
    all_ep = m.recent_episodes(n=10)
    assert any("gardening" in e for e in all_ep) and any("cars" in e for e in all_ep)
    assert not any("durable" in e for e in all_ep)


# ── embedders ────────────────────────────────────────────────────────────────
def test_hash_embedder_properties():
    e = HashEmbedder(128)
    a, b = e.embed("wifi password"), e.embed("wifi password")
    assert a is not None and (a == b).all()          # deterministic
    assert abs(float((a * a).sum()) - 1.0) < 1e-5    # unit norm
    assert e.embed("") is None
    sim_close = float((e.embed("the wifi password") * a).sum())
    sim_far = float((e.embed("her sister's birthday in June") * a).sum())
    assert sim_close > sim_far


def test_build_embedder_modes():
    assert build_embedder("off") is None
    assert build_embedder("hash").name == "hash"
    assert build_embedder("auto").name == "hash"                    # no host
    assert build_embedder("auto", host="http://x", model="m").name == "ollama"


def test_keyword_overlap():
    assert keyword_overlap("wifi password", "the wifi password is x") > 0
    assert keyword_overlap("", "anything") == 0.0
    assert keyword_overlap("cats", "dogs") == 0.0
