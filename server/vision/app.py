"""Zero Vision GPU service — FastAPI entrypoint.

* ``GET /health``   (CP0) — proves the tunnel is alive, reports CUDA/VRAM.
* ``POST /facts``   (CP4) — frame + detections -> per-object ``distance_m`` and
  ``bearing`` (Depth Anything V2 + intrinsics). ``reply`` comes back empty; kept
  standalone for fallback/debug.
* ``POST /analyze`` (CP5) — runs the CP4 facts, then a local VLM grounded in
  those facts, returning ``{facts, reply}`` ready for the Pi to speak. This is
  the endpoint CP6 wires the Pi CLI to.

Run (from the ``server/`` directory)::

    uvicorn app:app --host 0.0.0.0 --port 8000

or simply::

    python app.py
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import numpy as np

# Robust imports: works whether launched as `app:app` (cwd=server/) or
# `server.app:app` (cwd=repo root).
try:
    from config import load_config
    from depth.depth_anything import DepthEstimator
    from facts.intrinsics import Intrinsics, load_intrinsics
    from facts.spatial import to_scene_facts
    from shared.schemas import (AnalyzeRequest, AnalyzeResponse, IngestRequest,
                                SceneResponse)
    from stream import LiveDetector, LiveScene
    from vlm.model import VLM
except ImportError:  # pragma: no cover - import path fallback
    from server.vision.config import load_config
    from server.vision.depth.depth_anything import DepthEstimator
    from server.vision.facts.intrinsics import Intrinsics, load_intrinsics
    from server.vision.facts.spatial import to_scene_facts
    from server.vision.shared.schemas import (AnalyzeRequest, AnalyzeResponse,
                                              IngestRequest, SceneResponse)
    from server.vision.stream import LiveDetector, LiveScene
    from server.vision.vlm.model import VLM

import cv2
from fastapi import FastAPI, HTTPException

CONFIG = load_config()
SERVER_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Zero Vision GPU service", version="0.5-cp5")

# Lazily constructed so /health stays cheap and importing the app never pulls in
# torch. The depth model loads on the first /facts (or /analyze) call; the VLM on
# the first /analyze call. Both are cached for the process lifetime.
_DEPTH: Optional[DepthEstimator] = None
_VLM: Optional[VLM] = None
_LIVE_DETECTOR: Optional[LiveDetector] = None
_LIVE_SCENE = LiveScene()  # always-fresh scene from the streamed video


def _depth() -> DepthEstimator:
    global _DEPTH
    if _DEPTH is None:
        _DEPTH = DepthEstimator(CONFIG)
    return _DEPTH


def _live_detector() -> LiveDetector:
    global _LIVE_DETECTOR
    if _LIVE_DETECTOR is None:
        _LIVE_DETECTOR = LiveDetector(CONFIG)
    return _LIVE_DETECTOR


def _vlm() -> VLM:
    global _VLM
    if _VLM is None:
        _VLM = VLM(CONFIG)
    return _VLM


def gpu_status() -> tuple[bool, float]:
    """Return (cuda_available, total_vram_gb). Degrades to (False, 0.0)."""
    try:
        import torch
    except ImportError:
        return False, 0.0

    if not torch.cuda.is_available():
        return False, 0.0

    props = torch.cuda.get_device_properties(0)
    vram_gb = round(props.total_memory / (1024**3), 2)
    return True, vram_gb


def _decode_frame(image_jpeg_b64: str) -> np.ndarray:
    """base64 JPEG (per architecture.md) -> BGR ndarray; raise on bad input."""
    if not image_jpeg_b64:
        raise HTTPException(status_code=422, detail="image_jpeg_b64 is empty.")
    try:
        jpg = base64.b64decode(image_jpeg_b64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"image_jpeg_b64 not valid base64: {exc}")
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode JPEG frame.")
    return frame


def _intrinsics_for(width: int, height: int) -> Intrinsics:
    cam = CONFIG.get("camera", {})
    raw = cam.get("intrinsics_path")
    path = None
    if raw:
        p = Path(raw)
        path = p if p.is_absolute() else (SERVER_DIR / p).resolve()
    intr = load_intrinsics(
        path,
        target_width=width,
        target_height=height,
        fallback_hfov_deg=float(cam.get("fallback_hfov_deg", 70.0)),
    )
    if intr.approximate:
        print(
            f"[facts] using APPROXIMATE intrinsics (FOV={cam.get('fallback_hfov_deg', 70.0)}°); "
            f"calibrate and copy intrinsics.json for accurate distances."
        )
    return intr


def _save_depth_debug(depth_map: np.ndarray) -> None:
    dbg = CONFIG.get("debug", {})
    if not dbg.get("enabled", False):
        return
    out_dir = SERVER_DIR / str(dbg.get("dir", "debug"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "depth_latest.png"
    cv2.imwrite(str(path), DepthEstimator.colorize(depth_map))
    print(f"[facts] wrote depth debug image -> {path}")


@app.get("/health")
def health() -> dict:
    gpu, vram_gb = gpu_status()
    return {"status": "ok", "gpu": gpu, "vram_gb": vram_gb}


@app.post("/ingest", response_model=SceneResponse)
def ingest(req: IngestRequest) -> SceneResponse:
    """Continuous streaming perception: one video frame in -> current scene out.

    Runs YOLO + tracker on the GPU (fast) and updates the live scene, returning
    the tracked detections so the Pi always holds a fresh view. Depth is only run
    when ``want_facts`` is set (it's the expensive step), so the stream stays fast.
    """
    frame = _decode_frame(req.image_jpeg_b64)
    detections = _live_detector().track(frame)
    _LIVE_SCENE.update(detections, frame_bgr=frame)

    facts = []
    if req.want_facts and detections:
        height, width = frame.shape[:2]
        depth_map = _depth().estimate(frame)
        intrinsics = _intrinsics_for(width, height)
        facts = to_scene_facts(frame, detections, depth_map, intrinsics,
                               CONFIG.get("facts", {}))
    _dets, _frame, idx, age = _LIVE_SCENE.snapshot()
    return SceneResponse(detections=detections, facts=facts, frame_index=idx, age_s=age)


@app.get("/scene", response_model=SceneResponse)
def scene() -> SceneResponse:
    """The current live scene (no new frame) — what the GPU is seeing right now."""
    dets, _frame, idx, age = _LIVE_SCENE.snapshot()
    return SceneResponse(detections=dets, facts=[], frame_index=idx, age_s=age)


def _compute_facts(req: AnalyzeRequest):
    """Shared CP4 pipeline: decode frame -> depth -> grounded scene facts.

    Returns ``(frame_bgr, scene_facts)`` so callers that also need the image
    (e.g. the VLM) don't decode it twice.
    """
    frame = _decode_frame(req.image_jpeg_b64)
    height, width = frame.shape[:2]

    depth_map = _depth().estimate(frame)
    intrinsics = _intrinsics_for(width, height)
    scene_facts = to_scene_facts(
        frame, req.detections, depth_map, intrinsics, CONFIG.get("facts", {})
    )
    _save_depth_debug(depth_map)
    return frame, scene_facts


@app.post("/facts", response_model=AnalyzeResponse)
def facts(req: AnalyzeRequest) -> AnalyzeResponse:
    """Frame + detections -> grounded scene facts (CP4). ``reply`` is empty."""
    _frame, scene_facts = _compute_facts(req)
    return AnalyzeResponse(facts=scene_facts, reply="")


def run_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """CP5 pipeline: CP4 facts -> grounded VLM reply -> ``{facts, reply}``.

    Factored out of the endpoint so the eval script (``vlm/test_analyze.py``)
    exercises the exact same path in-process, without HTTP.
    """
    frame, scene_facts = _compute_facts(req)
    reply = _vlm().generate(
        frame, facts=scene_facts, question=req.question, history=req.history
    )
    return AnalyzeResponse(facts=scene_facts, reply=reply)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Frame + detections + question + history -> grounded spoken reply (CP5)."""
    return run_analyze(req)


def main() -> None:
    import uvicorn

    srv = CONFIG["server"]
    uvicorn.run(
        app,
        host=srv["host"],
        port=int(srv["port"]),
        log_level=srv.get("log_level", "info"),
    )


if __name__ == "__main__":
    main()
