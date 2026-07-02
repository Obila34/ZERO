from zero.memory.sqlite_memory import SqliteMemory


def _mem(tmp_path, **kw):
    return SqliteMemory(str(tmp_path / "mem.sqlite"), **kw)


def test_remember_and_recall(tmp_path):
    mem = _mem(tmp_path)
    mem.remember("name", "Greg")
    assert mem.facts() == {"name": "Greg"}
    assert "Greg" in mem.as_block()


def test_upsert_same_key(tmp_path):
    mem = _mem(tmp_path)
    mem.remember("city", "Nairobi")
    mem.remember("city", "Mombasa")
    assert mem.count() == 1
    assert mem.facts()["city"] == "Mombasa"


def test_stored_facts_are_pruned(tmp_path):
    mem = _mem(tmp_path, max_facts=3, max_stored_facts=5)
    for i in range(10):
        mem.remember(f"key{i}", f"value {i}")
    assert mem.count() == 5  # storage cap
    assert len(mem.facts()) == 3  # injection cap
    # The newest facts survive.
    assert "key9" in mem.facts()


def test_stored_episodes_are_pruned(tmp_path):
    mem = _mem(tmp_path, recent_episodes=2, max_stored_episodes=4)
    for i in range(10):
        mem.add_episode(f"talked about topic {i}")
    assert mem.episode_count() == 4
    # recent_episodes returns the newest, oldest-first.
    assert mem.recent_episodes() == ["talked about topic 8", "talked about topic 9"]


def test_empty_key_or_value_ignored(tmp_path):
    mem = _mem(tmp_path)
    mem.remember("", "value")
    mem.remember("key", "  ")
    assert mem.count() == 0


def test_forget_all(tmp_path):
    mem = _mem(tmp_path)
    mem.remember("a", "b")
    mem.add_episode("chat")
    mem.forget_all()
    assert mem.count() == 0
    assert mem.episode_count() == 0
    assert mem.as_block() == ""
