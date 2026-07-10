"""Identity notes attach on recognition CHANGE, not every turn.

Repeating "(You can see Greg)" each turn fed the LLM greeting-fodder — it kept
re-greeting mid-conversation instead of staying on topic. These drive the real
Zero._attach_vision with a minimal stub (no camera, no memory).
"""
from __future__ import annotations

import types

from zero.main import Zero


def _stub(person=None, face_name=None):
    obj = types.SimpleNamespace()
    obj.eyes = None
    obj.memory = None
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
