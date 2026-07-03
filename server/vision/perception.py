"""GPU perception endpoints — the Pi's heavy senses, moved server-side.

Mounted into the existing vision app (:8000, same SSH tunnel), so the Pi needs
no new ports. Every model loads lazily on first use and stays cached:

  POST /perceive/detect        JPEG -> object detections (YOLO11x, CUDA)
  POST /perceive/face          JPEG -> face embeddings + boxes
                               (InsightFace SCRFD+ArcFace if installed,
                                else Haar + ArcFace ONNX)
  POST /perceive/speaker       WAV  -> ECAPA voice embedding (reuses the
                               repo's own zero/voiceid implementation)
  POST /perceive/embed_object  JPEG crop -> CLIP image embedding
                               (transformers, already present for the VLM)

Each endpoint returns 503 with a clear reason when its model isn't available,
so the Pi's fallback wrappers degrade to their local engines instead of
guessing.
"""
from __future__ import annotations

import base64
import io
import threading
import wave
from pathlib import Path
from typing import Optional

# One lock per lazy model. FastAPI runs sync endpoints in a large threadpool,
# so without this the first burst of concurrent requests each start their OWN
# model build (and, worse, their own weight download of the SAME file, which
# corrupts and restarts endlessly). Double-checked locking loads exactly once.
_LOAD_LOCK = threading.Lock()

import numpy as np

import cv2
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/perceive")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── config knobs (overridable via the vision server's config.yaml) ───────────
YOLO_MODEL = "yolo11x.pt"          # auto-downloaded by ultralytics on first use
YOLO_CONF = 0.35
CLIP_MODEL = "openai/clip-vit-base-patch32"
SPEAKER_ONNX = REPO_ROOT / "models" / "voiceid" / "voxceleb_ECAPA512_LM.onnx"
ARCFACE_ONNX = REPO_ROOT / "models" / "identity" / "arcface.onnx"


def _decode_jpeg_b64(image_jpeg_b64: str) -> np.ndarray:
    """base64 JPEG -> RGB ndarray (all zero/ code is RGB-native)."""
    if not image_jpeg_b64:
        raise HTTPException(422, "image_jpeg_b64 is empty.")
    try:
        raw = base64.b64decode(image_jpeg_b64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, f"not valid base64: {exc}")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(422, "could not decode JPEG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FrameReq(BaseModel):
    image_jpeg_b64: str
    max_faces: int = 3


# ── YOLO11x detection ─────────────────────────────────────────────────────────
_YOLO = None


def _yolo():
    global _YOLO
    if _YOLO is None:
        with _LOAD_LOCK:
            if _YOLO is None:  # re-check inside the lock (download happens once)
                from ultralytics import YOLO  # pip install ultralytics

                _YOLO = YOLO(YOLO_MODEL)
                print(f"[perceive] YOLO loaded: {YOLO_MODEL}", flush=True)
    return _YOLO


@router.post("/detect")
def detect(req: FrameReq) -> dict:
    try:
        model = _yolo()
    except Exception as exc:
        raise HTTPException(503, f"YOLO unavailable (pip install ultralytics): {exc}")
    frame = _decode_jpeg_b64(req.image_jpeg_b64)
    results = model.predict(frame, conf=YOLO_CONF, verbose=False)
    dets = []
    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            dets.append({
                "label": names[int(box.cls[0])],
                "bbox": [x1, y1, x2 - x1, y2 - y1],   # xywh, matching the Pi
                "confidence": float(box.conf[0]),
            })
    return {"detections": dets}


# ── face embeddings (SCRFD+ArcFace preferred, Haar+ONNX fallback) ────────────
_FACE_APP = None          # insightface FaceAnalysis
_FACE_LOCAL = None        # (haar_cascade, onnx_session) fallback


def _face_engine():
    """Returns ("insight", app) or ("local", (cascade, session)). Loads once
    even under concurrent requests (double-checked lock)."""
    global _FACE_APP, _FACE_LOCAL
    if _FACE_APP is not None:
        return "insight", _FACE_APP
    if _FACE_LOCAL is not None:
        return "local", _FACE_LOCAL
    with _LOAD_LOCK:
        if _FACE_APP is not None:
            return "insight", _FACE_APP
        if _FACE_LOCAL is not None:
            return "local", _FACE_LOCAL
        try:
            from insightface.app import FaceAnalysis  # pip install insightface

            app = FaceAnalysis(name="buffalo_l",
                               allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            _FACE_APP = app
            print("[perceive] InsightFace SCRFD+ArcFace ready", flush=True)
            return "insight", app
        except Exception as exc:
            print(f"[perceive] insightface unavailable ({exc}); trying local ONNX",
                  flush=True)
        if not ARCFACE_ONNX.exists():
            raise HTTPException(
                503, "no face engine: install insightface or place "
                     f"{ARCFACE_ONNX.name} under models/identity/")
        import onnxruntime as ort

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        sess = ort.InferenceSession(str(ARCFACE_ONNX))
        _FACE_LOCAL = (cascade, sess)
        return "local", _FACE_LOCAL


@router.post("/face")
def face(req: FrameReq) -> dict:
    kind, engine = _face_engine()
    frame = _decode_jpeg_b64(req.image_jpeg_b64)
    faces = []
    if kind == "insight":
        for f in engine.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            faces.append({"embedding": emb.tolist(),
                          "bbox": [x1, y1, x2 - x1, y2 - y1]})
    else:
        cascade, sess = engine
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        boxes = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        boxes = sorted((tuple(int(v) for v in b) for b in boxes),
                       key=lambda b: b[2] * b[3], reverse=True)
        input_name = sess.get_inputs()[0].name
        for (x, y, w, h) in boxes:
            m = int(round(max(w, h) * 0.15))
            H, W = frame.shape[:2]
            crop = frame[max(0, y - m):min(H, y + h + m),
                         max(0, x - m):min(W, x + w + m)]
            if crop.size == 0:
                continue
            img = cv2.resize(crop, (112, 112))
            xin = ((img.astype(np.float32) - 127.5) / 128.0)
            xin = np.transpose(xin, (2, 0, 1))[None]
            emb = np.asarray(sess.run(None, {input_name: xin})[0][0],
                             dtype=np.float32)
            n = float(np.linalg.norm(emb))
            if n > 1e-6:
                faces.append({"embedding": (emb / n).tolist(),
                              "bbox": [float(x), float(y), float(w), float(h)]})
    # Largest faces first, capped.
    faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
    return {"faces": faces[: max(1, req.max_faces)]}


# ── speaker embedding (the repo's own ECAPA implementation) ──────────────────
_SPEAKER = None


def _speaker():
    global _SPEAKER
    if _SPEAKER is None:
        with _LOAD_LOCK:
            if _SPEAKER is None:
                if not SPEAKER_ONNX.exists():
                    raise HTTPException(
                        503, f"speaker model missing: {SPEAKER_ONNX} "
                             "(download the WeSpeaker ECAPA ONNX)")
                import sys

                sys.path.insert(0, str(REPO_ROOT))
                from zero.voiceid.speaker import SpeakerVerifier

                _SPEAKER = SpeakerVerifier(str(SPEAKER_ONNX))
                print("[perceive] speaker embedder ready", flush=True)
    return _SPEAKER


@router.post("/speaker")
async def speaker(request: Request) -> dict:
    verifier = _speaker()
    data = await request.body()
    try:
        with wave.open(io.BytesIO(data)) as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    except Exception as exc:
        raise HTTPException(422, f"body must be a 16-bit mono WAV: {exc}")
    if sr != 16000:
        raise HTTPException(422, f"expected 16 kHz audio, got {sr}")
    emb = verifier.embed(pcm)
    if emb is None:
        return {"embedding": None}
    return {"embedding": emb.tolist()}


# ── CLIP object embedding ─────────────────────────────────────────────────────
_CLIP = None


def _clip():
    global _CLIP
    if _CLIP is None:
        with _LOAD_LOCK:
            if _CLIP is None:
                try:
                    import torch
                    from transformers import (
                        CLIPImageProcessor, CLIPVisionModelWithProjection,
                    )
                except Exception as exc:
                    raise HTTPException(
                        503, f"CLIP unavailable (needs transformers+torch): {exc}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = CLIPVisionModelWithProjection.from_pretrained(
                    CLIP_MODEL).to(device).eval()
                proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
                _CLIP = (model, proc, device, torch)
                print(f"[perceive] CLIP ready on {device}: {CLIP_MODEL}", flush=True)
    return _CLIP


@router.post("/embed_object")
def embed_object(req: FrameReq) -> dict:
    model, proc, device, torch = _clip()
    crop = _decode_jpeg_b64(req.image_jpeg_b64)
    inputs = proc(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model(**inputs).image_embeds[0].float().cpu().numpy()
    n = float(np.linalg.norm(emb))
    if n < 1e-6:
        return {"embedding": None, "dim": 0}
    v = (emb / n).astype(np.float32)
    return {"embedding": v.tolist(), "dim": int(v.size)}
