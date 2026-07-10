"""End-of-session memory is split PER SPEAKER (voice-owned sessions).

Drives the real Zero._save_memories with a fake LLM + a real SQLite store, so a
multi-person conversation can never write one speaker's facts/summary into
another's memory (the contamination bug this fixes).
"""
from __future__ import annotations

import types

from zero.main import Zero
from zero.memory.embeddings import HashEmbedder
from zero.memory.sqlite_memory import SqliteMemory


class _FakeLLM:
    """Summaries echo the transcript (so we can assert isolation); facts +
    reflection are stubbed to NONE to keep the test about the split, not parsing."""

    def stream(self, messages):
        sysmsg = messages[0]["content"]
        body = messages[-1]["content"]
        if "Summarize this conversation" in sysmsg:
            yield "We discussed " + body.replace("\n", "; ")
        else:
            yield "NONE"


class _Cfg:
    def __init__(self, d):
        self.d = d

    def get(self, k, default=None):
        return self.d.get(k, default)


def _zero_stub(tmp_path):
    obj = types.SimpleNamespace()
    obj.cfg = _Cfg({"memory.last_conversation.enabled": True})
    obj.llm = _FakeLLM()
    obj.memory = SqliteMemory(str(tmp_path / "m.sqlite"), embedder=HashEmbedder(64))
    for name in ("_save_memories", "_extract_facts", "_summarize_session", "_reflect"):
        setattr(obj, name, types.MethodType(getattr(Zero, name), obj))
    return obj


def test_save_memories_is_split_per_speaker(tmp_path):
    z = _zero_stub(tmp_path)
    z._save_memories([
        (1, "user", "I love planets and my NASA space trip"),
        (1, "assistant", "Tell me more about the trip"),
        (2, "user", "I love blue flowers and anime worlds"),
        (2, "assistant", "Nice, which anime?"),
        (None, "user", "hi"),                       # too short -> skipped entirely
    ])
    one = z.memory.last_conversation(1)
    two = z.memory.last_conversation(2)
    assert one and "planets" in one[0] and "flowers" not in one[0]
    assert two and "flowers" in two[0] and "planets" not in two[0]


def test_anonymous_turns_get_a_global_episode_not_a_person_record(tmp_path):
    z = _zero_stub(tmp_path)
    z._save_memories([
        (None, "user", "I was chatting about gardening today"),
        (None, "assistant", "nice"),
    ])
    assert z.memory.last_conversation(1) is None       # no person credited
    assert any("gardening" in e for e in z.memory.recent_episodes())  # global arc kept


def test_low_content_bucket_is_ignored(tmp_path):
    z = _zero_stub(tmp_path)
    z._save_memories([(3, "user", "hey"), (3, "assistant", "hi there")])
    assert z.memory.last_conversation(3) is None       # a bare greeting isn't a chat
