"""Zero Vision wire contract — shared by the Pi (edge) and the GPU server.

This is the single source of truth for the frame-analysis request/response
models. Because the two nodes do NOT share a filesystem, identical copies of
this file are mirrored into ``pi/shared/schemas.py`` and ``server/shared/schemas.py``
so each deployed node directory is self-contained. Keep all copies in sync;
edit this canonical file first, then run ``python shared/sync_schemas.py``.

Fields are defined now (CP0) even though detection/depth/VLM are not yet
implemented — later checkpoints fill in the producers/consumers.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """One object found by the Pi's local detector (YOLO11n, CP1)."""

    label: str = Field(..., description="Class name, e.g. 'cup', 'person'.")
    bbox: list[float] = Field(
        ...,
        description="Bounding box as [x, y, w, h] in pixels of the sent frame.",
        min_length=4,
        max_length=4,
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    color: Optional[str] = Field(
        None, description="Dominant color word from HSV naming (CP1), if known."
    )


class AnalyzeRequest(BaseModel):
    """Pi -> GPU: a keyframe plus local context and the user's question."""

    image_jpeg_b64: str = Field(
        ..., description="JPEG-encoded frame, base64 (str). See architecture.md."
    )
    detections: list[Detection] = Field(default_factory=list)
    question: str = Field(..., description="Raw text the user typed in the CLI.")
    history: list[dict] = Field(
        default_factory=list,
        description="Short rolling conversation history [{role, content}, ...].",
    )


class SceneFact(BaseModel):
    """GPU -> grounded fact about one object (depth + bearing, CP4)."""

    label: str
    color: Optional[str] = None
    distance_m: Optional[float] = Field(None, ge=0.0)
    bearing: Optional[str] = Field(
        None, description="Coarse direction, e.g. 'left' | 'center' | 'right'."
    )


class AnalyzeResponse(BaseModel):
    """GPU -> Pi: structured facts plus the natural-language reply to speak."""

    facts: list[SceneFact] = Field(default_factory=list)
    reply: str = Field(..., description="Sentence the Pi speaks via Piper TTS.")


if __name__ == "__main__":
    # Smoke test: construct one of everything and round-trip through JSON.
    req = AnalyzeRequest(
        image_jpeg_b64="",
        detections=[Detection(label="cup", bbox=[10, 20, 30, 40], confidence=0.9, color="red")],
        question="What do you see?",
        history=[{"role": "user", "content": "hi"}],
    )
    resp = AnalyzeResponse(
        facts=[SceneFact(label="cup", color="red", distance_m=1.2, bearing="left")],
        reply="That's a red cup, about 1.2 meters to your left.",
    )
    print(req.model_dump_json(indent=2))
    print(resp.model_dump_json(indent=2))
    print("schemas OK")
