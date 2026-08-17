"""FaceRecognizer — face detection + landmark alignment + ArcFace embedding.

The same embed→cosine shape as the voice path: a frame goes in, a 512-d unit
embedding per detected face comes out. Detection and alignment are tiered,
best-available picked at construction and demoted per-call on failure:

* **YuNet** (``detector_path``, runs on cv2's built-in FaceDetectorYN) — the
  room-scale detector: small faces across a room, angled faces, dim light.
  Returns the five canonical ArcFace points (eyes, nose, mouth corners)
  itself, so every detection can be properly aligned even with nothing else
  installed.
* **MediaPipe FaceLandmarker** (``landmarker_path`` + the ``mediapipe``
  wheel) — the alignment refiner. Its internal detector is short-range
  BlazeFace (selfie distance — it misses room-distance faces on a full
  frame), so it runs on each YuNet face *crop*, where it excels, and its
  iris-accurate landmarks replace YuNet's. Without YuNet it doubles as a
  close-range full-frame detector.
* **Haar cascade** — legacy last resort, frontal-only, margin-crop alignment.
  Note: opencv-python wheels >= 5.0 ship no cascade data.

Aligned faces are warped to the 112x112 ArcFace template with a similarity
transform, so enrolment and matching see the same geometry regardless of head
pose. A per-call backend failure demotes that tier for the session rather than
raising: this class IS the degraded path when the GPU tunnel is down, so it
must never be the thing that fails.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("identity.face")

_ARC_SIZE = 112  # ArcFace canonical input

# Canonical ArcFace 112x112 landmark template (the InsightFace standard):
# left eye, right eye, nose tip, left mouth corner, right mouth corner —
# "left/right" in image coordinates (left = smaller x).
_ARC_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# FaceLandmarker (478-point mesh) indices for those five points. Iris centers
# need the refined-landmark model, which face_landmarker.task includes.
_LM_IRIS_A, _LM_IRIS_B = 468, 473
_LM_NOSE = 1
_LM_MOUTH_A, _LM_MOUTH_B = 61, 291

_REFINE_MARGIN = 0.4  # bbox expansion for the per-face landmarker crop


class YuNetDetector:
    """Room-scale face detection via cv2's built-in FaceDetectorYN.

    Detect-only and cheap (~2-5 ms per 640x480 frame on a Pi), so it can run
    inside the Eyes loop as well as under FaceRecognizer. Each call returns
    ``[(bbox, pts)]`` — (x, y, w, h) ints plus the five ArcFace landmark
    points (5x2 float32, image coords, eyes/mouth ordered left-first) —
    largest face first. Thread-safe via an internal lock (FaceDetectorYN is
    stateful: input size is set per frame).
    """

    def __init__(self, model_path: str, min_size: int = 60,
                 score_threshold: float = 0.8, detect_scale: float = 1.0):
        import cv2

        if not Path(model_path).exists():
            raise FileNotFoundError(f"yunet model not found: {model_path}")
        self._cv2 = cv2
        self.min_size = int(min_size)
        # Detection cost scales with PIXELS, and setInputSize() below hands the
        # net the frame's full resolution — 640x480 measured ~97 ms on a Pi 4,
        # which capped face tracking at ~10 Hz. detect_scale <1 runs the net on
        # a downscaled copy and scales the boxes back: 0.5 is ~4x cheaper for
        # the same faces at conversational distance. 1.0 = unchanged (identity
        # and enrolment keep full resolution, where accuracy matters most).
        self._scale = min(1.0, max(0.1, float(detect_scale)))
        # 0.8: real faces score ~0.9 even small/rotated; the sub-0.7 tail is
        # false positives — and a false face could get ENROLLED or framed.
        self._det = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320),
            score_threshold=float(score_threshold),
            nms_threshold=0.3, top_k=50,
        )
        self._lock = threading.Lock()

    def detect_faces(self, frame_rgb: np.ndarray) -> list[tuple[tuple, np.ndarray]]:
        cv2 = self._cv2
        H, W = frame_rgb.shape[:2]
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        s = self._scale
        if s < 1.0:
            bgr = cv2.resize(bgr, (max(1, int(W * s)), max(1, int(H * s))),
                             interpolation=cv2.INTER_LINEAR)
        dh, dw = bgr.shape[:2]
        with self._lock:
            self._det.setInputSize((dw, dh))
            _, faces = self._det.detect(bgr)
        inv = 1.0 / s if s < 1.0 else 1.0
        out = []
        for row in (faces if faces is not None else []):
            x, y, w, h = (v * inv for v in row[:4])   # back to frame coords
            if min(w, h) < self.min_size:
                continue
            bbox = (int(x), int(y), int(w), int(h))
            pts = _order_5pt(
                np.array(row[4:14], dtype=np.float32).reshape(5, 2) * inv)
            out.append((bbox, pts))
        return sorted(out, key=lambda f: f[0][2] * f[0][3], reverse=True)


def _order_5pt(pts: np.ndarray) -> np.ndarray:
    """Template rows are image-left-first; detector left/right labels flip
    with head pose, so order eyes and mouth corners by x instead."""
    if pts[0, 0] > pts[1, 0]:
        pts[[0, 1]] = pts[[1, 0]]
    if pts[3, 0] > pts[4, 0]:
        pts[[3, 4]] = pts[[4, 3]]
    return pts


def align_5pt(frame_rgb: np.ndarray, pts: np.ndarray,
              size: int = _ARC_SIZE) -> np.ndarray | None:
    """Warp ``frame_rgb`` so ``pts`` (5x2, image coords) land on the ArcFace
    template. Similarity transform only (rotation+scale+translation) — a full
    affine would shear the face and shift the embedding."""
    import cv2

    M, _ = cv2.estimateAffinePartial2D(
        np.asarray(pts, dtype=np.float32), _ARC_TEMPLATE, method=cv2.LMEDS
    )
    if M is None:
        return None
    return cv2.warpAffine(frame_rgb, M, (size, size), flags=cv2.INTER_LINEAR)


class FaceRecognizer:
    def __init__(self, model_path: str, min_size: int = 60, margin: float = 0.15,
                 detector_path: str | None = None,
                 landmarker_path: str | None = None, max_faces: int = 4):
        import cv2  # lazy — vision stack is optional
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"face model not found: {model_path}")
        self._cv2 = cv2
        self.min_size = int(min_size)
        self.margin = float(margin)
        self.max_faces = int(max_faces)
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name

        self._yunet = None
        self._landmarker = None
        self._mp = None
        self._cascade = None
        self._det_lock = threading.Lock()  # YuNet/FaceLandmarker aren't thread-safe
        if detector_path:
            try:
                self._load_yunet(detector_path)
            except Exception as e:
                log.warning("YuNet detector unavailable (%s)", e)
        if landmarker_path:
            try:
                self._load_landmarker(landmarker_path)
            except Exception as e:
                log.warning("mediapipe landmarker unavailable (%s)", e)
        if self._yunet is None and self._landmarker is None:
            self._load_cascade()  # raises if cv2 ships no cascades — then
            # there is no detector at all and the recognizer is useless
        log.info("face recognizer loaded (%s, backend=%s)",
                 model_path, self.backend)

    @property
    def backend(self) -> str:
        """'<detector>+mediapipe' when the refiner is active, e.g.
        'yunet+mediapipe', 'yunet', 'mediapipe', 'haar', 'none'."""
        det = ("yunet" if self._yunet is not None else
               "mediapipe" if self._landmarker is not None else
               "haar" if self._cascade is not None else "none")
        if det == "yunet" and self._landmarker is not None:
            det += "+mediapipe"
        return det

    # ── backend loading / demotion ───────────────────────────────────────────
    def _load_yunet(self, path: str) -> None:
        self._yunet = YuNetDetector(path, min_size=self.min_size)

    def _load_landmarker(self, path: str) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        if not Path(path).exists():
            raise FileNotFoundError(f"landmarker model not found: {path}")
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=self.max_faces,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp

    def _load_cascade(self) -> None:
        # opencv-python wheels >= 5.0 ship NO cascade data — this can
        # legitimately fail even though cv2 imported fine.
        cv2 = self._cv2
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            raise RuntimeError("could not load OpenCV Haar frontal-face cascade")
        self._cascade = cascade

    def _demote(self, tier: str, err: Exception) -> None:
        """Disable a failing tier for the session; keep something alive."""
        log.warning("%s failed (%s) — demoting for the rest of this session",
                    tier, err)
        if tier == "yunet":
            self._yunet = None
        else:
            self._landmarker = None
        if self._yunet is None and self._landmarker is None \
                and self._cascade is None:
            try:
                self._load_cascade()
            except Exception as e:
                # No detector left: stay alive and blind (empty results)
                # rather than take the whole identity channel down.
                log.error("Haar fallback also unavailable (%s) — face "
                          "detection disabled until restart", e)

    # ── detection ────────────────────────────────────────────────────────────
    def _yunet_faces(self, frame_rgb: np.ndarray) -> list[tuple[tuple, np.ndarray]]:
        """[(bbox, 5x2 points)] from YuNet, largest first."""
        return self._yunet.detect_faces(frame_rgb)

    def _mp_faces(self, frame_rgb: np.ndarray,
                  min_size: int = 0) -> list[tuple[tuple, np.ndarray]]:
        """[(bbox, 5x2 points)] from a full-frame FaceLandmarker pass. Only
        finds close-range faces — used as the detector when YuNet is absent,
        and on face crops by ``_refine``."""
        mp = self._mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(frame_rgb, dtype=np.uint8))
        with self._det_lock:
            result = self._landmarker.detect(img)
        H, W = frame_rgb.shape[:2]
        out = []
        for lms in result.face_landmarks:
            xs = np.clip([p.x for p in lms], 0.0, 1.0) * W
            ys = np.clip([p.y for p in lms], 0.0, 1.0) * H
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            w, h = x1 - x0, y1 - y0
            if min(w, h) < min_size:
                continue
            pts = _order_5pt(np.array(
                [[lms[i].x * W, lms[i].y * H]
                 for i in (_LM_IRIS_A, _LM_IRIS_B, _LM_NOSE,
                           _LM_MOUTH_A, _LM_MOUTH_B)],
                dtype=np.float32,
            ))
            bbox = (int(x0), int(y0), int(round(w)), int(round(h)))
            out.append((bbox, pts))
        return sorted(out, key=lambda f: f[0][2] * f[0][3], reverse=True)

    def _haar_boxes(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._cascade is None:
            return []
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_size, self.min_size),
        )
        boxes = [tuple(int(v) for v in f) for f in faces]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def _faces(self, frame_rgb: np.ndarray) -> list[tuple[tuple, np.ndarray | None]]:
        """[(bbox, 5x2 points or None)] via the best live tier. Never raises."""
        if self._yunet is not None:
            try:
                return self._yunet_faces(frame_rgb)
            except Exception as e:
                self._demote("yunet", e)
        if self._landmarker is not None:
            try:
                return self._mp_faces(frame_rgb, min_size=self.min_size)
            except Exception as e:
                self._demote("mediapipe landmarker", e)
        try:
            return [(box, None) for box in self._haar_boxes(frame_rgb)]
        except Exception as e:
            log.debug("face detect failed: %s", e)
            return []

    def _refine(self, frame_rgb: np.ndarray, bbox: tuple,
                pts: np.ndarray | None) -> np.ndarray | None:
        """Swap detector landmarks for MediaPipe's iris-accurate ones, computed
        on an expanded face crop (short-range BlazeFace territory). Falls back
        to the detector's own points on any failure."""
        if self._landmarker is None or self._yunet is None:
            return pts  # no refiner, or landmarker already was the detector
        x, y, w, h = bbox
        m = int(round(max(w, h) * _REFINE_MARGIN))
        H, W = frame_rgb.shape[:2]
        x0, y0 = max(0, x - m), max(0, y - m)
        crop = frame_rgb[y0:min(H, y + h + m), x0:min(W, x + w + m)]
        if crop.size == 0:
            return pts
        try:
            refined = self._mp_faces(crop)
        except Exception as e:
            self._demote("mediapipe landmarker", e)
            return pts
        if not refined:
            return pts
        return refined[0][1] + np.float32([x0, y0])

    def detect(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Face boxes (x, y, w, h) in the frame, largest first."""
        return [bbox for bbox, _ in self._faces(frame_rgb)]

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
        out: list[tuple[np.ndarray, tuple]] = []
        for bbox, pts in self._faces(frame_rgb)[:max_faces]:
            pts = self._refine(frame_rgb, bbox, pts)
            face = align_5pt(frame_rgb, pts) if pts is not None else None
            if face is None:  # Haar tier, or degenerate landmarks
                face = self._crop(frame_rgb, bbox)
            try:
                emb = self.embed_crop(face)
            except Exception as e:
                log.debug("face embed failed: %s", e)
                emb = None
            if emb is not None:
                out.append((emb, bbox))
        return out
