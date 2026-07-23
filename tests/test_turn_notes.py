"""Identity notes attach on recognition CHANGE, not every turn.

Repeating "(You can see Greg)" each turn fed the LLM greeting-fodder — it kept
re-greeting mid-conversation instead of staying on topic. These drive the real
Zero._attach_vision with a minimal stub (no camera, no memory).
"""
from __future__ import annotations

import types

from zero.main import Zero


class _FakeCfg:
    def __init__(self, d=None):
        self.d = d or {}

    def get(self, k, default=None):
        return self.d.get(k, default)


def _stub(person=None, face_name=None, memory=None, cfg=None):
    obj = types.SimpleNamespace()
    obj.eyes = None
    obj.memory = memory
    obj.cfg = cfg or _FakeCfg()
    obj._person = person
    obj._face_name = face_name
    obj._last_id_key = None
    obj._turn_notes = []
    for n in ("_attach_vision", "_look", "_is_visual", "_filler_category"):
        setattr(obj, n, types.MethodType(getattr(Zero, n), obj))
    return obj


def _msgs():
    return [{"role": "system", "content": "s"},
            {"role": "user", "content": "tell me more about planets"}]


def _last_content(out):
    return out[-1]["content"]


def test_id_note_attaches_once_then_dedups():
    greg = types.SimpleNamespace(is_known=True, name="Greg", person_id=1)
    z = _stub(person=greg, face_name="Greg")
    first = z._attach_vision(_msgs(), "tell me more about planets")
    assert "You can see Greg" in _last_content(first)          # first sighting
    z._turn_notes = []
    second = z._attach_vision(_msgs(), "tell me more about planets")
    assert "You can see Greg" not in _last_content(second)     # same person: quiet


def test_id_note_reattaches_on_person_change():
    greg = types.SimpleNamespace(is_known=True, name="Greg", person_id=1)
    z = _stub(person=greg, face_name="Greg")
    z._attach_vision(_msgs(), "hello there my friend")
    z._person = types.SimpleNamespace(is_known=True, name="Sam", person_id=2)
    z._face_name = "Sam"
    z._turn_notes = []
    out = z._attach_vision(_msgs(), "hello again my friend")
    assert "You can see Sam" in _last_content(out)             # new person: announce


def test_id_note_reattaches_on_visual_question():
    greg = types.SimpleNamespace(is_known=True, name="Greg", person_id=1)
    z = _stub(person=greg, face_name="Greg")
    z._attach_vision(_msgs(), "tell me more about planets")
    z._turn_notes = []
    out = z._attach_vision(_msgs(), "what do you see right now")
    assert "You can see Greg" in _last_content(out)            # asked: answer it


def test_recall_over_budget_is_skipped_not_waited_for():
    # A slow memory/embedder must NEVER sit in the reply path — over budget,
    # the turn simply goes without the recall note.
    import time as _time

    class SlowMemory:
        def relevant_block(self, text, person_id=None, exclude=""):
            _time.sleep(2.0)                       # way over the 300ms budget
            return "something ancient"

    z = _stub(memory=SlowMemory())
    t0 = _time.monotonic()
    out = z._attach_vision(_msgs(), "tell me more about planets")
    elapsed = _time.monotonic() - t0
    assert elapsed < 1.0                           # did not wait the 2s
    assert "reminds you" not in _last_content(out)


def test_recall_within_budget_attaches_note():
    class FastMemory:
        def relevant_block(self, text, person_id=None, exclude=""):
            return "they love planets"

    z = _stub(memory=FastMemory())
    out = z._attach_vision(_msgs(), "tell me more about planets")
    assert "they love planets" in _last_content(out)


def test_id_note_reattaches_after_absence():
    greg = types.SimpleNamespace(is_known=True, name="Greg", person_id=1)
    z = _stub(person=greg, face_name="Greg")
    z._attach_vision(_msgs(), "tell me more about planets")
    z._person, z._face_name = None, None                       # nobody recognised
    z._turn_notes = []
    z._attach_vision(_msgs(), "tell me more about planets")
    z._person, z._face_name = greg, "Greg"                     # Greg returns
    z._turn_notes = []
    out = z._attach_vision(_msgs(), "tell me more about planets")
    assert "You can see Greg" in _last_content(out)
