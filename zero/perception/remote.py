"""Remote perception — the Pi's heavy senses served by the GPU.

The Pi stays a frontend: camera frames and utterance audio go up the existing
SSH tunnel (:8000, the vision server) and detections/embeddings come back.
Every client below has the SAME surface as the local engine it replaces, and
each is wrapped in a Fallback* that lazily builds the local engine on the
first remote failure — the exact pattern RemoteSTT/FallbackSTT proved:

    RemoteDetector        <-> zero.vision.detector.Detector      (.detect)
    RemoteFaceRecognizer  <-> zero.identity.face.FaceRecognizer  (.embed_faces/.detect)
    RemoteSpeakerEmbedder <-> zero.voiceid.speaker.SpeakerVerifier (.embed)
    RemoteObjectEmbedder  <-> zero.vision.learned.HistEmbedder   (.embed)

A dropped tunnel degrades ZERO to its local (smaller) models per-call and
recovers the moment the tunnel returns. `degraded` on each wrapper feeds the
self-state narration.
"""
from __future__ import annotations

import base64
import io
import wave
import time
from typing import Callable, Optional

import numpy as np
import requests

from zero.utils.logging import get_logger

log = get_logger("perception.remote")


def _encode_jpeg_b64(frame_rgb: np.ndarray, quality: int = 80) -> str:
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not JPEG-encode frame")
    return base64.b64encode(buf).decode("ascii")


class PerceptionClient:
    """Thin HTTP client for the GPU /perceive/* endpoints. Raises RuntimeError
    on any failure so fallback wrappers can tell 'tunnel down' from 'empty'."""

    def __init__(self, base_url: str, timeout: float = 5.0,
                 detect_timeout: float = 10.0, jpeg_quality: int = 80,
                 post: Callable = requests.post):
        self.base = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.detect_timeout = float(detect_timeout)
        self.jpeg_quality = int(jpeg_quality)
        self._post = post

    def _json(self, path: str, payload: dict, timeout: float) -> dict:
        try:
            r = self._post(f"{self.base}{path}", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise RuntimeError(f"remote perception {path} failed "
                               f"(server/tunnel up?): {e}") from e

    def detect(self, frame_rgb: np.ndarray) -> list[dict]:
        payload = {"image_jpeg_b64": _encode_jpeg_b64(frame_rgb, self.jpeg_quality)}
        return self._json("/perceive/detect", payload,
                          self.detect_timeout).get("detections", [])

    def face_embeds(self, frame_rgb: np.ndarray,
                    max_faces: int = 3) -> list[tuple[np.ndarray, tuple]]:
        payload = {"image_jpeg_b64": _encode_jpeg_b64(frame_rgb, self.jpeg_quality),
                   "max_faces": int(max_faces)}
        out = []
        for f in self._json("/perceive/face", payload, self.timeout).get("faces", []):
            emb = np.asarray(f.get("embedding", []), dtype=np.float32)
            if emb.size:
                out.append((emb, tuple(f.get("bbox", (0, 0, 0, 0)))))
        return out

    def speaker_embed(self, audio: np.ndarray,
                      sample_rate: int = 16000) -> Optional[np.ndarray]:
        pcm = (audio if audio.dtype == np.int16
               else np.clip(audio * 32768.0, -32768, 32767).astype(np.int16))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        try:
            r = self._post(f"{self.base}/perceive/speaker", data=buf.getvalue(),
                           headers={"Content-Type": "audio/wav"},
                           timeout=self.timeout)
            r.raise_for_status()
            emb = r.json().get("embedding")
        except Exception as e:
            raise RuntimeError(f"remote perception /perceive/speaker failed "
                               f"(server/tunnel up?): {e}") from e
        return np.asarray(emb, dtype=np.float32) if emb else None

    def object_embed(self, crop_rgb: np.ndarray) -> Optional[np.ndarray]:
        payload = {"image_jpeg_b64": _encode_jpeg_b64(crop_rgb, self.jpeg_quality)}
        emb = self._json("/perceive/embed_object", payload,
                         self.timeout).get("embedding")
        return np.asarray(emb, dtype=np.float32) if emb else None


# ── same-surface remote engines ───────────────────────────────────────────────
class RemoteDetector:
    """Detector surface (.detect) served by GPU YOLO11x."""

    def __init__(self, client: PerceptionClient):
        self._client = client
        log.info("remote detector -> %s (YOLO on the GPU)", client.base)

    def detect(self, frame_rgb: np.ndarray) -> list:
        from zero.vision.schemas import Detection

        return [Detection(label=d["label"], bbox=list(d["bbox"]),
                          confidence=float(d["confidence"]))
                for d in self._client.detect(frame_rgb)]


class RemoteFaceRecognizer:
    """FaceRecognizer surface (.embed_faces / .detect) served by the GPU."""

    def __init__(self, client: PerceptionClient):
        self._client = client
        log.info("remote face recognizer -> %s", client.base)

    def embed_faces(self, frame_rgb: np.ndarray,
                    max_faces: int = 3) -> list[tuple[np.ndarray, tuple]]:
        return self._client.face_embeds(frame_rgb, max_faces=max_faces)

    def detect(self, frame_rgb: np.ndarray) -> list[tuple]:
        return [box for _emb, box in self._client.face_embeds(frame_rgb)]


class RemoteSpeakerEmbedder:
    """SpeakerVerifier's .embed surface served by the GPU."""

    def __init__(self, client: PerceptionClient):
        self._client = client
        log.info("remote speaker embedder -> %s", client.base)

    def embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        return self._client.speaker_embed(audio)


class RemoteObjectEmbedder:
    """Learned-objects embedder surface (.embed) served by GPU CLIP."""

    name = "clip-remote"

    def __init__(self, client: PerceptionClient):
        self._client = client
        log.info("remote object embedder -> %s (CLIP on the GPU)", client.base)

    def embed(self, crop_rgb: np.ndarray) -> Optional[np.ndarray]:
        if crop_rgb is None or crop_rgb.size == 0:
            return None
        return self._client.object_embed(crop_rgb)


# ── fallback wrappers (remote first, lazily-built local on failure) ──────────
class _FallbackBase:
    def __init__(self, primary, builder: Optional[Callable] = None,
                 what: str = "perception"):
        self._primary = primary
        self._builder = builder
        self._local = None
        self._local_broken = False
        self._what = what
        self.degraded = False
        self._retry_after = 0.0    # skip the dead remote until this time
        self._fail_streak = 0

    def _fallback(self):
        if self._local is None and self._builder is not None and not self._local_broken:
            try:
                log.info("building local fallback %s (first remote failure)...",
                         self._what)
                self._local = self._builder()
            except Exception as e:
                self._local_broken = True
                log.error("could not build local %s fallback: %s", self._what, e)
        return self._local

    def _call(self, method: str, *args, empty=None, **kwargs):
        # Back off from a dead remote instead of retrying it on EVERY call.
        # The retry-primary-first design recovers instantly when a tunnel comes
        # back, but while it is down each call pays a full connection timeout —
        # on a congested network that is seconds of latency added to every
        # frame, and the log showed it firing several times per second. Retry
        # occasionally so recovery still happens, just not on the hot path.
        now = time.monotonic()
        if now < self._retry_after:
            local = self._fallback()
            if local is not None:
                try:
                    return getattr(local, method)(*args, **kwargs)
                except Exception as e:
                    log.debug("local %s fallback failed: %s", self._what, e)
            return empty
        try:
            out = getattr(self._primary, method)(*args, **kwargs)
            if self.degraded:
                log.info("remote %s recovered", self._what)
            self.degraded = False
            self._fail_streak = 0
            return out
        except Exception as e:
            self._fail_streak += 1
            # Exponential-ish backoff, capped: 1s, 2s, 4s ... 30s.
            self._retry_after = now + min(30.0, 2.0 ** min(self._fail_streak, 5))
            if self._fail_streak <= 2 or self._fail_streak % 20 == 0:
                log.warning("remote %s failed (%d in a row, retrying in %.0fs): %s",
                            self._what, self._fail_streak,
                            self._retry_after - now, e)
        self.degraded = True
        local = self._fallback()
        if local is None:
            return empty
        try:
            return getattr(local, method)(*args, **kwargs)
        except Exception as e:
            log.error("local %s fallback failed too: %s", self._what, e)
            return empty


class FallbackDetector(_FallbackBase):
    def __init__(self, primary, builder=None):
        super().__init__(primary, builder, what="detector")

    def detect(self, frame_rgb):
        return self._call("detect", frame_rgb, empty=[])


class FallbackFace(_FallbackBase):
    def __init__(self, primary, builder=None):
        super().__init__(primary, builder, what="face recognizer")

    def embed_faces(self, frame_rgb, max_faces=3):
        return self._call("embed_faces", frame_rgb, max_faces=max_faces, empty=[])

    def detect(self, frame_rgb):
        return self._call("detect", frame_rgb, empty=[])


class FallbackSpeaker(_FallbackBase):
    def __init__(self, primary, builder=None):
        super().__init__(primary, builder, what="speaker embedder")

    def embed(self, audio):
        return self._call("embed", audio, empty=None)


class FallbackObjectEmbedder(_FallbackBase):
    """Note: remote CLIP (512-d) and local histogram vectors live in different
    dims — LearnedObjects stores dim per row and only compares like-with-like,
    so mixed teaching during an outage degrades matching, never corrupts it."""

    def __init__(self, primary, builder=None):
        super().__init__(primary, builder, what="object embedder")

    @property
    def name(self):
        return getattr(self._primary, "name", "remote")

    def embed(self, crop_rgb):
        return self._call("embed", crop_rgb, empty=None)
