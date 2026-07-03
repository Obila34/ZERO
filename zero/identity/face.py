"""FaceRecognizer — face detection (OpenCV Haar) + ArcFace ONNX embedding.

The same embed→cosine shape as the voice path: a frame goes in, a 512-d unit
embedding per detected face comes out. Haar ships inside opencv-python (no
model download); ArcFace is a single ONNX file selected in config. Alignment
is a margin-crop + resize — not landmark-perfect, but consistent between
enrolment and matching, which is what cosine similarity actually needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("identity.face")

_ARC_SIZE = 112  # ArcFace canonical input


class FaceRecognizer:
    def __init__(self, model_path: str, min_size: int = 60, margin: float = 0.15):
        import cv2  # lazy — vision stack is optional
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"face model not found: {model_path}")
        self._cv2 = cv2
        self.min_size = int(min_size)
        self.margin = float(margin)
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self._cascade.empty():
            raise RuntimeError("could not load OpenCV Haar frontal-face cascade")
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name
        log.info("face recognizer loaded (%s)", model_path)

    # ── detection ────────────────────────────────────────────────────────────
    def detect(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Face boxes (x, y, w, h) in the frame, largest first."""
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_size, self.min_size),
        )
        boxes = [tuple(int(v) for v in f) for f in faces]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    # ── embedding ────────────────────────────────────────────────────────────
    def _crop(self, frame_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        m = int(round(max(w, h) * self.margin))
        H, W = frame_rgb.shape[:2]
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(W, x + w + m), min(H, y + h + m)
        return frame_rgb[y0:y1, x0:x1]

    def embed_crop(self, face_rgb: np.ndarray) -> np.ndarray | None:
        cv2 = self._cv2
        if face_rgb.size == 0:
            return None
        img = cv2.resize(face_rgb, (_ARC_SIZE, _ARC_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) - 127.5) / 128.0        # ArcFace norm
        x = np.transpose(x, (2, 0, 1))[None, :, :, :]        # NCHW
        out = self._sess.run(None, {self._input: x})[0][0]
        emb = np.asarray(out, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(emb))
        if not np.isfinite(n) or n < 1e-6:
            return None
        return emb / n

    def embed_faces(self, frame_rgb: np.ndarray,
                    max_faces: int = 3) -> list[tuple[np.ndarray, tuple]]:
        """[(embedding, bbox)] for up to ``max_faces`` faces, largest first.
        Never raises — a bad frame just returns []."""
        try:
            boxes = self.detect(frame_rgb)
        except Exception as e:
            log.debug("face detect failed: %s", e)
            return []
        out: list[tuple[np.ndarray, tuple]] = []
        for box in boxes[:max_faces]:
            try:
                emb = self.embed_crop(self._crop(frame_rgb, box))
            except Exception as e:
                log.debug("face embed failed: %s", e)
                emb = None
            if emb is not None:
                out.append((emb, box))
        return out
