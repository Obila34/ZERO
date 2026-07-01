"""HTTP client from the Pi to the GPU vision server, over the SSH tunnel.

* ``check_health()`` — cheap liveness probe used to decide fallback.
* ``get_facts(frame_rgb, detections)`` — send a keyframe + local detections, get
  back grounded ``SceneFact``s (``distance_m`` + ``bearing`` per object).

All failures raise a clear ``RuntimeError`` so the caller can fall back to a
local-only (detections + color) answer instead of hanging.
"""
from __future__ import annotations

import base64

import requests

from zero.utils.logging import get_logger
from zero.vision.schemas import (AnalyzeRequest, AnalyzeResponse, Detection,
                                 IngestRequest, SceneResponse, SceneFact)

log = get_logger("vision.client")


class VisionClient:
    def __init__(self, url: str = "http://127.0.0.1:8000",
                 health_path: str = "/health", facts_path: str = "/facts",
                 ingest_path: str = "/ingest",
                 health_timeout_s: float = 5.0, facts_timeout_s: float = 15.0,
                 ingest_timeout_s: float = 10.0, jpeg_quality: int = 80):
        self.base = url.rstrip("/")
        self.health_path = health_path
        self.facts_path = facts_path
        self.ingest_path = ingest_path
        self.health_timeout = float(health_timeout_s)
        self.facts_timeout = float(facts_timeout_s)
        self.ingest_timeout = float(ingest_timeout_s)
        self.jpeg_quality = int(jpeg_quality)

    def check_health(self) -> dict:
        url = self.base + self.health_path
        try:
            resp = requests.get(url, timeout=self.health_timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach GPU vision server at {url}. Is the ssh -L "
                f"tunnel up and vision_server running? ({exc})"
            ) from exc
        return resp.json()

    def _encode_frame_rgb(self, frame_rgb) -> str:
        import cv2

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode the frame.")
        return base64.b64encode(buf).decode("ascii")

    def get_facts(self, frame_rgb, detections: list[Detection],
                  question: str = "") -> list[SceneFact]:
        """POST keyframe + detections to ``/facts``; return grounded facts."""
        url = self.base + self.facts_path
        request = AnalyzeRequest(
            image_jpeg_b64=self._encode_frame_rgb(frame_rgb),
            detections=list(detections),
            question=question,
            history=[],
        )
        try:
            resp = requests.post(url, json=request.model_dump(),
                                 timeout=self.facts_timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"GPU /facts request to {url} failed ({exc})."
            ) from exc
        try:
            parsed = AnalyzeResponse.model_validate(resp.json())
        except ValueError as exc:
            raise RuntimeError(f"GPU /facts returned an unparseable body: {exc}") from exc
        return parsed.facts

    def ingest(self, frame_rgb, want_facts: bool = False) -> SceneResponse:
        """Stream one frame to ``/ingest``; get back the current live scene
        (tracked detections + colors, and distances if ``want_facts``)."""
        url = self.base + self.ingest_path
        request = IngestRequest(
            image_jpeg_b64=self._encode_frame_rgb(frame_rgb), want_facts=want_facts,
        )
        try:
            resp = requests.post(url, json=request.model_dump(),
                                 timeout=self.ingest_timeout)
            resp.raise_for_status()
            return SceneResponse.model_validate(resp.json())
        except requests.RequestException as exc:
            raise RuntimeError(f"GPU /ingest request to {url} failed ({exc}).") from exc
        except ValueError as exc:
            raise RuntimeError(f"GPU /ingest returned an unparseable body: {exc}") from exc
