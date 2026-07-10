"""Interaction corpus: per-speaker JSONL training records."""
from __future__ import annotations

import json

from zero.learning.corpus import Corpus


def _records(tmp_path):
    lines = (tmp_path / "corpus" / "interactions.jsonl").read_text(
        encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def test_writes_one_record_per_speaker(tmp_path):
    c = Corpus(str(tmp_path / "corpus"))
    n = c.add_session({
        1: [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hey"}],
        -2: [{"role": "user", "text": "who are you"}],
    })
    assert n == 2
    recs = _records(tmp_path)
    kinds = {r["speaker"]: r["speaker_kind"] for r in recs}
    assert kinds == {1: "known", -2: "guest"}


def test_anonymous_and_empty_handling(tmp_path):
    c = Corpus(str(tmp_path / "corpus"))
    n = c.add_session({
        None: [{"role": "user", "text": "just chatting"}],
        3: [{"role": "user", "text": "   "}],   # all-blank -> dropped
    })
    assert n == 1
    recs = _records(tmp_path)
    assert len(recs) == 1 and recs[0]["speaker_kind"] == "anonymous"


def test_appends_across_sessions(tmp_path):
    c = Corpus(str(tmp_path / "corpus"))
    c.add_session({1: [{"role": "user", "text": "first"}]})
    c.add_session({1: [{"role": "user", "text": "second"}]})
    recs = _records(tmp_path)
    assert len(recs) == 2
    assert [r["turns"][0]["text"] for r in recs] == ["first", "second"]


def test_meta_is_recorded(tmp_path):
    c = Corpus(str(tmp_path / "corpus"))
    c.add_session({1: [{"role": "user", "text": "hi"}]}, meta={"source": "voice"})
    assert _records(tmp_path)[0]["meta"] == {"source": "voice"}
