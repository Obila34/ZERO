"""Open-vocab detection server: /perceive/vocab endpoint + learned.py coexistence.

The real YOLO-World model needs a GPU and a weights download, so these tests
drive the endpoints with a fake model that mimics the slice of the ultralytics
API perception.py touches (predict / set_classes / names). What IS real here:
the vocab parsing, dedup, persistence, locking paths, response shapes, and the
label flow from model names -> /detect response -> LearnedObjects override.
"""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
cv2 = pytest.importorskip("cv2")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.vision import perception


# ── fake ultralytics model ────────────────────────────────────────────────────
class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [np.asarray(xyxy, dtype=np.float32)]
        self.cls = [cls]
        self.conf = [conf]


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class FakeWorldModel:
    """set_classes works (open-vocab); predict returns one box of class 0."""

    def __init__(self):
        self.classes: list[str] = []
        self.names = {0: "unset"}
        self.set_classes_calls = 0

    def set_classes(self, words):
        self.set_classes_calls += 1
        self.classes = list(words)
        self.names = dict(enumerate(words))

    def predict(self, frame, conf=0.25, verbose=False):
        return [FakeResult(self.names, [FakeBox([2, 3, 12, 23], 0, 0.9)])]


class FakeClosedModel(FakeWorldModel):
    """Plain YOLO: no set_classes."""

    def set_classes(self, words):  # ultralytics raises on non-world models
        raise AttributeError("this model does not support set_classes")


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture()
def client(monkeypatch, tmp_path):
    """TestClient over the perceive router with a fake OPEN-vocab model and a
    tmp vocab file seeded with three words."""
    seed = tmp_path / "seed.txt"
    seed.write_text("mug\nlaptop  # comment survives parsing\n\nscrewdriver\n")
    model = FakeWorldModel()
    monkeypatch.setattr(perception, "VOCAB_PATH", tmp_path / "vocab.runtime.txt")
    monkeypatch.setattr(perception, "VOCAB_SEED", seed)
    monkeypatch.setattr(perception, "_YOLO", model)
    monkeypatch.setattr(perception, "_IS_WORLD", True)
    model.set_classes(perception._load_vocab())

    app = FastAPI()
    app.include_router(perception.router)
    return TestClient(app), model


def _jpeg_b64() -> str:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf).decode("ascii")


# ── /vocab ───────────────────────────────────────────────────────────────────
def test_vocab_get_reports_seeded_classes(client):
    c, _model = client
    body = c.get("/perceive/vocab").json()
    assert body["classes"] == ["mug", "laptop", "screwdriver"]
    assert body["open_vocab"] is True


def test_vocab_add_updates_live_model_and_persists(client):
    c, model = client
    before = model.set_classes_calls
    body = c.post("/perceive/vocab",
                  json={"add": ["fire extinguisher", "  mug "]}).json()
    # "mug" already present (whitespace/case-insensitive dedup) -> one new word.
    assert body["classes"] == ["mug", "laptop", "screwdriver", "fire extinguisher"]
    assert model.set_classes_calls == before + 1
    assert model.classes[-1] == "fire extinguisher"          # live model updated
    assert "set_classes_ms" in body
    persisted = perception.VOCAB_PATH.read_text().split()
    assert "extinguisher" in " ".join(persisted)             # survives restart


def test_vocab_remove_is_case_insensitive(client):
    c, model = client
    body = c.post("/perceive/vocab", json={"remove": ["LAPTOP"]}).json()
    assert body["classes"] == ["mug", "screwdriver"]
    assert model.classes == ["mug", "screwdriver"]


def test_vocab_refuses_to_empty_the_vocabulary(client):
    c, _model = client
    r = c.post("/perceive/vocab",
               json={"remove": ["mug", "laptop", "screwdriver"]})
    assert r.status_code == 400


def test_vocab_on_closed_set_model_is_409(client, monkeypatch):
    c, _model = client
    monkeypatch.setattr(perception, "_YOLO", FakeClosedModel())
    monkeypatch.setattr(perception, "_IS_WORLD", False)
    r = c.post("/perceive/vocab", json={"add": ["anything"]})
    assert r.status_code == 409


# ── /detect labels come from the live vocabulary ─────────────────────────────
def test_detect_returns_open_vocab_label(client):
    c, _model = client
    body = c.post("/perceive/detect", json={"image_jpeg_b64": _jpeg_b64()}).json()
    assert body["detections"][0]["label"] == "mug"           # vocab word, id 0
    assert body["detections"][0]["bbox"] == [2.0, 3.0, 10.0, 20.0]  # xywh


def test_detect_label_tracks_vocab_change(client):
    c, _model = client
    c.post("/perceive/vocab", json={"remove": ["mug"]})
    body = c.post("/perceive/detect", json={"image_jpeg_b64": _jpeg_b64()}).json()
    assert body["detections"][0]["label"] == "laptop"        # new id 0


# ── learned.py coexists unmodified with open-vocab labels ────────────────────
def test_learned_override_applies_to_open_vocab_label(tmp_path):
    """"This is David's mug" must override the open-vocab label 'mug' exactly as
    it overrode COCO labels — learned.py is label-source agnostic."""
    from zero.vision.learned import HistEmbedder, LearnedObjects
    from zero.vision.schemas import Detection

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[8:56, 8:56] = (200, 40, 40)                        # a red "mug"
    store = LearnedObjects(str(tmp_path / "obj.sqlite"), HistEmbedder(),
                           match_threshold=0.8)
    assert store.teach("David's mug", frame[8:56, 8:56])

    det = Detection(label="mug", bbox=[8, 8, 48, 48], confidence=0.9, color=None)
    out = store.annotate(frame, [det])
    assert out[0].label == "David's mug"
    assert det.label == "mug"                                # original untouched
