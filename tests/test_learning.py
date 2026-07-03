"""Learning: object teaching (parse + store + match), curiosity queue, preferences."""
from __future__ import annotations

import numpy as np
import pytest

from zero.identity.service import parse_enrollment
from zero.memory.preferences import apply_rate_delta, parse_preference
from zero.proactive.curiosity import CuriosityStore
from zero.vision.learned import LearnedObjects, parse_object_teach


# ── teach-phrase parsing (and non-collision with person enrolment) ──────────
@pytest.mark.parametrize("text,name", [
    ("This is a french press.", "french press"),
    ("that's my espresso cup", "espresso cup"),
    ("Remember, this is the blender", "blender"),
    ("it's a soldering iron", "soldering iron"),
])
def test_parse_object_teach_accepts(text, name):
    assert parse_object_teach(text) == name


@pytest.mark.parametrize("text", [
    "This is Peter.",            # person, not object (no article)
    "this is great",             # too short a "name"
    "what is this",
    "this is a x",               # too short
])
def test_parse_object_teach_rejects(text):
    assert parse_object_teach(text) is None


def test_person_vs_object_routing():
    """The article decides: 'this is Peter' -> person, 'this is a peter
    grinder' -> object. Both parsers must agree on who owns what."""
    assert parse_enrollment("This is Peter.") == "Peter"
    assert parse_object_teach("This is Peter.") is None
    assert parse_object_teach("this is a pepper grinder") == "pepper grinder"
    assert parse_enrollment("this is a pepper grinder") is None


# ── learned-object store (fake embedder: no cv2 needed) ─────────────────────
class CropColorEmbedder:
    """Embeds a crop as its mean RGB, unit-normalized — enough to prove the
    store's teach/match/threshold logic without cv2."""
    name = "fake"

    def embed(self, crop):
        if crop is None or crop.size == 0:
            return None
        v = np.asarray(crop, dtype=np.float32).reshape(-1, 3).mean(axis=0)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None


def _crop(r, g, b):
    return np.full((8, 8, 3), (r, g, b), dtype=np.uint8)


@pytest.fixture()
def store(tmp_path):
    return LearnedObjects(str(tmp_path / "obj.sqlite"), CropColorEmbedder(),
                          match_threshold=0.98, max_samples_per_object=2)


def test_teach_and_match(store):
    assert store.teach("red mug", _crop(200, 10, 10)) is True
    assert store.teach("blue bottle", _crop(10, 10, 200)) is True
    hit = store.match(_crop(190, 20, 20))
    assert hit is not None and hit[0] == "red mug"
    assert store.match(_crop(10, 200, 10)) is None      # green: nothing learned
    assert sorted(store.names()) == ["blue bottle", "red mug"]


def test_teach_sample_cap(store):
    for i in range(4):
        store.teach("mug", _crop(200, 10 + i, 10))
    n = store._db.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
    assert n == 2


def test_forget_object(store):
    store.teach("mug", _crop(200, 10, 10))
    assert store.forget("MUG") is True
    assert store.match(_crop(200, 10, 10)) is None
    assert store.forget("ghost") is False


def test_teach_rejects_empty(store):
    assert store.teach("", _crop(1, 2, 3)) is False
    assert store.teach("mug", np.zeros((0, 0, 3), dtype=np.uint8)) is False


class _Det:
    """Minimal Detection stand-in with pydantic's model_copy surface."""
    def __init__(self, label, bbox, confidence):
        self.label, self.bbox, self.confidence = label, bbox, confidence

    def model_copy(self, update=None):
        d = _Det(self.label, self.bbox, self.confidence)
        for k, v in (update or {}).items():
            setattr(d, k, v)
        return d


def test_annotate_overrides_labels(store):
    store.teach("red mug", _crop(200, 10, 10))
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[2:10, 2:10] = (200, 10, 10)          # the red mug region
    dets = [_Det("cup", [2, 2, 8, 8], 0.9), _Det("person", [12, 12, 6, 6], 0.9)]
    out = store.annotate(frame, dets)
    assert out[0].label == "red mug"           # learned name wins
    assert out[1].label == "person"            # untouched


# ── curiosity queue ──────────────────────────────────────────────────────────
@pytest.fixture()
def curiosity(tmp_path):
    return CuriosityStore(str(tmp_path / "cur.sqlite"), max_pending=3)


def test_curiosity_add_and_ask(curiosity):
    assert curiosity.add("k1", "What's the red thing on the desk?", 5.0)
    assert curiosity.pending_count() == 1
    q = curiosity.next_question()
    assert q is not None and "red thing" in q[1]
    curiosity.mark_asked(q[0])
    assert curiosity.next_question() is None    # asked once, not repeated


def test_curiosity_dedup(curiosity):
    assert curiosity.add("k1", "q?") is True
    assert curiosity.add("k1", "q again?") is False
    assert curiosity.pending_count() == 1


def test_curiosity_priority_and_cap(curiosity):
    curiosity.add("a", "low?", 1.0)
    curiosity.add("b", "mid?", 5.0)
    curiosity.add("c", "high?", 9.0)
    curiosity.add("d", "newer?", 6.0)          # cap 3: evicts lowest ('a')
    assert curiosity.pending_count() == 3
    assert curiosity.next_question()[1] == "high?"


def test_curiosity_person_scoping(curiosity):
    curiosity.add("p1", "for david?", 5.0, person_id=1)
    assert curiosity.next_question(person_id=2) is None
    assert curiosity.next_question(person_id=1)[1] == "for david?"


def test_note_unknown_object(curiosity):
    assert curiosity.note_unknown_object("vase") is True
    assert "vase" in curiosity.next_question()[1]


# ── preferences ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,pref,delta_sign", [
    ("Please speak slower", "speak more slowly", +1),
    ("slow down", "speak more slowly", +1),
    ("talk faster please", "speak faster", -1),
    ("keep it short", "keep replies short and concise", 0),
    ("stop calling me buddy", "never call the user 'buddy'", 0),
])
def test_parse_preference_accepts(text, pref, delta_sign):
    got = parse_preference(text)
    assert got is not None
    got_pref, delta = got
    assert got_pref == pref
    if delta_sign == 0:
        assert delta is None
    else:
        assert delta is not None and (delta > 0) == (delta_sign > 0)


@pytest.mark.parametrize("text", [
    "the train was slower than usual today",
    "my brother told me to keep it short but I ignored him completely, classic",
    "what time is it",
])
def test_parse_preference_rejects(text):
    assert parse_preference(text) is None


def test_apply_rate_delta_finds_piper_knob():
    class Piper:
        length_scale = 1.0

    class Orchestrator:
        def __init__(self):
            self.tts = Piper()

    voice = Orchestrator()
    assert apply_rate_delta(voice, +0.15) is True
    assert voice.tts.length_scale == pytest.approx(1.15)
    # Clamped: can't run away.
    for _ in range(10):
        apply_rate_delta(voice, +0.15)
    assert voice.tts.length_scale <= 1.6


def test_apply_rate_delta_no_knob_is_false():
    class Remote:
        pass

    assert apply_rate_delta(Remote(), 0.15) is False
