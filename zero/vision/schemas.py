"""Vision wire contract — shared by the Pi (edge) and the GPU vision server.

These are the request/response models exchanged with the GPU vision service over
the SSH tunnel (``/facts`` for grounded depth + bearing). The same definitions
live on the server side under ``server/vision/shared/schemas.py``; keep them in
sync. Pydantic v2.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """One object found by the Pi's local detector (YOLO11n)."""

    label: str = Field(..., description="Class name, e.g. 'cup', 'person'.")
    bbox: list[float] = Field(
        ...,
        description="Bounding box as [x, y, w, h] in pixels of the sent frame.",
        min_length=4,
        max_length=4,
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    color: Optional[str] = Field(
        None, description="Dominant color word from HSV naming, if known."
    )
    track_id: Optional[int] = Field(
        None, description="Persistent tracker id across frames (object permanence)."
    )


class AnalyzeRequest(BaseModel):
    """Pi -> GPU: a keyframe plus local detections and the user's question."""

    image_jpeg_b64: str = Field(
        ..., description="JPEG-encoded frame, base64 (str)."
    )
    detections: list[Detection] = Field(default_factory=list)
    question: str = Field("", description="The user's spoken question (optional).")
    history: list[dict] = Field(
        default_factory=list,
        description="Short rolling conversation history [{role, content}, ...].",
    )


class SceneFact(BaseModel):
    """GPU -> grounded fact about one object (depth + bearing)."""

    label: str
    color: Optional[str] = None
    distance_m: Optional[float] = Field(None, ge=0.0)
    bearing: Optional[str] = Field(
        None, description="Coarse direction, e.g. 'left' | 'center' | 'right'."
    )


class AnalyzeResponse(BaseModel):
    """GPU -> Pi: structured facts plus an optional natural-language reply.

    For ``/facts`` the ``reply`` is empty — ZERO's own LLM writes the spoken
    answer. ``reply`` is only populated by the optional ``/analyze`` endpoint.
    """

    facts: list[SceneFact] = Field(default_factory=list)
    reply: str = Field("", description="Optional reply (only from /analyze).")


class IngestRequest(BaseModel):
    """Pi -> GPU: one streamed video frame for continuous perception."""

    image_jpeg_b64: str = Field(..., description="JPEG-encoded frame, base64.")
    want_facts: bool = Field(
        False, description="Also run depth on this frame and return distances."
    )


class SceneResponse(BaseModel):
    """GPU -> Pi: the current live scene from the streaming detector/tracker."""

    detections: list[Detection] = Field(default_factory=list)
    facts: list[SceneFact] = Field(default_factory=list)
    frame_index: int = 0
    age_s: float = 0.0
