"""Remote perception: client wire format, same-surface engines, fallbacks."""
from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np
import pytest

pytest.importorskip("cv2")

from zero.perception.remote import (
    FallbackDetector, FallbackFace, FallbackObjectEmbedder, FallbackSpeaker,
    PerceptionClient, RemoteDetector, RemoteFaceRecognizer,
    RemoteObjectEmbedder, RemoteSpeakerEmbedder,
)

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)
FRAME[10:30, 10:30] = (200, 30, 30)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


class FakePost:
    """Records calls; replies from a path->payload map."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def __call__(self, url, json=None, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "data": data,
                           "headers": headers, "timeout": timeout})
        for path, payload in self.replies.items():
            if url.endswith(path):
                if isinstance(payload, Exception):
                    raise payload
                return FakeResponse(payload)
        return FakeResponse({}, status=404)


def _client(replies):
    post = FakePost(replies)
    return PerceptionClient("http://gpu:8000", post=post), post


# ── wire format ───────────────────────────────────────────────────────────────
def test_detect_sends_jpeg_and_parses_detections():
    client, post = _client({"/perceive/detect": {"detections": [
        {"label": "cup", "bbox": [1, 2, 3, 4], "confidence": 0.9}]}})
    dets = RemoteDetector(client).detect(FRAME)
    # Request carried a decodable JPEG.
    b64 = post.calls[0]["json"]["image_jpeg_b64"]
    assert base64.b64decode(b64)[:2] == b"\xff\xd8"          # JPEG magic
    # Response became real pydantic Detections.
    assert len(dets) == 1
    assert dets[0].label == "cup" and dets[0].confidence == 0.9
    assert dets[0].bbox == [1, 2, 3, 4]


def test_face_embeds_parsed_and_normalized_shape():
    emb = [0.6, 0.8, 0.0]
    client, _ = _client({"/perceive/face": {"faces": [
        {"embedding": emb, "bbox": [5, 6, 7, 8]}]}})
    faces = RemoteFaceRecognizer(client).embed_faces(FRAME)
    assert len(faces) == 1
    vec, box = faces[0]
    assert vec.dtype == np.float32 and box == (5, 6, 7, 8)
    # .detect surface returns just the boxes (for the FER crop path).
    assert RemoteFaceRecognizer(client).detect(FRAME) == [(5, 6, 7, 8)]


def test_speaker_posts_valid_wav():
    client, post = _client({"/perceive/speaker": {"embedding": [1.0, 0.0]}})
    audio = (np.sin(np.linspace(0, 100, 16000)) * 20000).astype(np.int16)
    emb = RemoteSpeakerEmbedder(client).embed(audio)
    assert emb is not None and emb.size == 2
    body = post.calls[0]["data"]
    with wave.open(io.BytesIO(body)) as w:                   # server-parseable
        assert w.getframerate() == 16000 and w.getnchannels() == 1
    assert post.calls[0]["headers"]["Content-Type"] == "audio/wav"


def test_speaker_none_embedding_passthrough():
    client, _ = _client({"/perceive/speaker": {"embedding": None}})
    assert RemoteSpeakerEmbedder(client).embed(
        np.ones(1600, dtype=np.int16)) is None


def test_object_embed_roundtrip():
    client, _ = _client({"/perceive/embed_object":
                         {"embedding": [0.0, 1.0], "dim": 2}})
    emb = RemoteObjectEmbedder(client).embed(FRAME)
    assert emb is not None and emb.size == 2
    assert RemoteObjectEmbedder(client).embed(
        np.zeros((0, 0, 3), dtype=np.uint8)) is None         # empty crop guard


def test_client_failure_raises_runtimeerror():
    client, _ = _client({"/perceive/detect": ConnectionError("tunnel down")})
    with pytest.raises(RuntimeError, match="tunnel"):
        client.detect(FRAME)


# ── fallback wrappers ─────────────────────────────────────────────────────────
class DeadRemote:
    def detect(self, frame):
        raise RuntimeError("tunnel down")

    def embed_faces(self, frame, max_faces=3):
        raise RuntimeError("tunnel down")

    def embed(self, x):
        raise RuntimeError("tunnel down")


class LocalDetector:
    def detect(self, frame):
        return ["local-detection"]


def test_fallback_detector_degrades_and_recovers():
    calls = {"n": 0}

    class FlakyRemote:
        def detect(self, frame):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("tunnel down")
            return ["remote-detection"]

    # retry_backoff_s=0 keeps the original per-call probing for this test; the
    # backoff only changes HOW OFTEN a down remote is probed, not whether
    # recovery works. See test_dead_remote_is_not_probed_every_call.
    d = FallbackDetector(FlakyRemote(), lambda: LocalDetector(),
                         retry_backoff_s=0)
    assert d.detect(FRAME) == ["local-detection"]            # outage -> local
    assert d.degraded is True
    assert d.detect(FRAME) == ["remote-detection"]           # recovery per-call
    assert d.degraded is False


def test_dead_remote_is_not_probed_every_call():
    """A down tunnel must not cost a connection timeout on every frame. A live
    log showed the remote detector failing several times per second, each
    attempt paying a full timeout on an already-congested link."""
    calls = {"n": 0}

    class DeadCounting:
        def detect(self, frame):
            calls["n"] += 1
            raise RuntimeError("tunnel down")

    d = FallbackDetector(DeadCounting(), lambda: LocalDetector())
    for _ in range(50):
        assert d.detect(FRAME) == ["local-detection"]   # always served
    assert calls["n"] == 1, f"probed the dead remote {calls['n']} times, want 1"
    assert d.degraded is True


def test_fallback_face_and_speaker_empty_safe_without_local():
    f = FallbackFace(DeadRemote(), None)                     # no local model
    assert f.embed_faces(FRAME) == []
    assert f.detect(FRAME) == []
    s = FallbackSpeaker(DeadRemote(), None)
    assert s.embed(np.ones(1600, dtype=np.int16)) is None
    assert f.degraded and s.degraded


def test_fallback_local_builder_failure_is_contained():
    def broken_builder():
        raise ImportError("no model on disk")

    d = FallbackDetector(DeadRemote(), broken_builder)
    assert d.detect(FRAME) == []                             # never raises
    assert d.detect(FRAME) == []                             # builder not retried


def test_fallback_object_embedder_name_and_dims():
    class LocalHist:
        name = "hist"

        def embed(self, crop):
            v = np.ones(8, dtype=np.float32)
            return v / np.linalg.norm(v)

    fb = FallbackObjectEmbedder(DeadRemote(), lambda: LocalHist())
    assert fb.embed(FRAME) is not None and fb.embed(FRAME).size == 8
    # Mixed-dim safety lives in LearnedObjects: rows of another dim are skipped.
    from zero.vision.learned import LearnedObjects
    import tempfile, os

    path = os.path.join(tempfile.mkdtemp(), "obj.sqlite")
    store = LearnedObjects(path, fb, match_threshold=0.9)
    assert store.teach("mug", FRAME) is True                 # 8-d local vector

    class Clip512:
        name = "clip-remote"

        def embed(self, crop):
            v = np.ones(512, dtype=np.float32)
            return v / np.linalg.norm(v)

    store._embedder = Clip512()                              # tunnel came back
    assert store.match(FRAME) is None                        # skipped, no crash
