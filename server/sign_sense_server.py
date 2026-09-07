#!/usr/bin/env python3
"""Sign-sense sidecar — fingerspell recognition for ZERO (PerceptX lumen-codex).

The first time sign goes INTO ZERO instead of only out: a camera frame (or
pre-extracted hand landmarks) comes in, a letter comes back — or an honest
"that is not a letter". Serves the PerceptX model vault's lumen-codex ONNX
transformer with the EXACT feature pipeline it was trained with:

    image -> MediaPipe HandLandmarker (the vault's own .task extractor)
          -> 126-vec [left 21x3 | right 21x3] by handedness
          -> hand_features.normalize126 (wrist-center + span-scale, the vault
             reader's mandatory step; feature_meta.json normalize:true)
          -> StandardScaler (mean/scale extracted to scaler_params.npz —
             no sklearn at runtime)
          -> ONNX -> 26 letter logits + 128-d embedding
          -> prototype gate: cosine vs class centroids; below threshold the
             answer is known:false, NEVER a confident wrong letter.

Endpoints:
    GET  /health              -> {ok, model, letters, providers}
    POST /spell {"landmarks": [126 floats]}            -> result
    POST /spell {"image": "<base64 jpeg/png>"}         -> result
    result: {letter, confidence, known, mode, top:[...]} or {no_hand: true}

Env: SIGN_VAULT (default ~/perceptx-model-vault), SIGN_PORT (8210),
SIGN_PROTO_THRESHOLD (0.55, the vault reader's default).

Stateless per request — letter-stream debouncing/segmentation is the
caller's job (ZERO's perception loop), same division of labor as the
gesture sidecar.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

VAULT = os.path.expanduser(os.environ.get("SIGN_VAULT",
                                          "~/perceptx-model-vault"))
sys.path.insert(0, os.path.join(VAULT, "code", "backend", "ml", "training"))

import hand_features  # noqa: E402  (vault module: normalize126)

_BASE = os.path.join(VAULT, "recognition", "lumen-codex")
_THRESHOLD = float(os.environ.get("SIGN_PROTO_THRESHOLD", "0.55"))

_sess = None
_scaler = None
_labels: list[str] = []
_protos = None          # (labels list, centroids (N,128))
_landmarker = None


def _load():
    global _sess, _scaler, _labels, _protos
    import onnxruntime as ort

    _sess = ort.InferenceSession(
        os.path.join(_BASE, "asl_transformer.onnx"),
        providers=["CPUExecutionProvider"])
    p = np.load(os.path.join(_BASE, "scaler_params.npz"))
    _scaler = (p["mean"].astype(np.float32), p["scale"].astype(np.float32))
    _labels = open(os.path.join(_BASE, "labels.txt")).read().split()
    d = np.load(os.path.join(_BASE, "prototypes.npz"), allow_pickle=True)
    _protos = ([str(x) for x in d["labels"].tolist()],
               d["centroids"].astype(np.float32))


def _get_landmarker():
    """MediaPipe HandLandmarker from the vault's own .task — using any other
    extractor version silently shifts the feature distribution (CAVEATS.md)."""
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=os.path.join(
                VAULT, "extractors", "hand_landmarker.task")),
            num_hands=2)
        _landmarker = vision.HandLandmarker.create_from_options(opts)
    return _landmarker


def _image_to_126(img_bytes: bytes) -> np.ndarray | None:
    import cv2
    import mediapipe as mp

    arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("undecodable image")
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    res = _get_landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    vec = np.zeros(126, dtype=np.float32)
    for lms, handed in zip(res.hand_landmarks, res.handedness):
        side = handed[0].category_name.lower()   # 'left' / 'right'
        off = 0 if side == "left" else 63
        for i, lm in enumerate(lms[:21]):
            vec[off + 3 * i:off + 3 * i + 3] = (lm.x, lm.y, lm.z)
    return vec


def _predict(vec126: np.ndarray) -> dict:
    x = hand_features.normalize126(vec126)
    mean, scale = _scaler
    x = ((x - mean) / scale).astype(np.float32)[None]
    logits, emb = _sess.run(None, {"landmarks": x})
    probs = np.exp(logits[0] - logits[0].max())
    probs /= probs.sum()
    # prototype gate — the vault reader's predict_nn, verbatim in spirit:
    # cosine similarity to class centroids; below threshold -> known:false
    e = emb[0]
    n = float(np.linalg.norm(e))
    if n > 1e-8:
        e = e / n
    labels_p, cents = _protos
    sims = cents @ e
    order = np.argsort(sims)[::-1][:5]
    best = float(sims[order[0]])
    return {
        "letter": labels_p[int(order[0])],
        "confidence": max(0.0, best),
        "known": best >= _THRESHOLD,
        "mode": "embedding",
        "softmax_letter": _labels[int(probs.argmax())],
        "softmax_confidence": float(probs.max()),
        "top": [{"letter": labels_p[int(i)],
                 "confidence": max(0.0, float(sims[i]))} for i in order],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):       # quiet
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "model": "lumen-codex",
                             "letters": len(_labels),
                             "threshold": _THRESHOLD})
        else:
            self._json(404, {"ok": False})

    def do_POST(self):
        if self.path != "/spell":
            self._json(404, {"ok": False})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode())
            if "landmarks" in req:
                vec = np.asarray(req["landmarks"],
                                 dtype=np.float32).reshape(-1)
                if vec.shape[0] != 126:
                    raise ValueError(f"landmarks must be 126 floats, "
                                     f"got {vec.shape[0]}")
            elif "image" in req:
                vec = _image_to_126(base64.b64decode(req["image"]))
                if vec is None:
                    self._json(200, {"no_hand": True})
                    return
            else:
                raise ValueError("need 'landmarks' or 'image'")
            self._json(200, _predict(vec))
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})


def main() -> int:
    port = int(os.environ.get("SIGN_PORT", "8210"))
    _load()
    print(f"sign-sense sidecar: lumen-codex, {len(_labels)} letters, "
          f"gate threshold {_THRESHOLD}, port {port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
