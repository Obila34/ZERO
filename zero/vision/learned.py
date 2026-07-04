"""Few-shot learned objects — teach ZERO new names with no retraining.

"This is a French press" → the current object crop is embedded and bound to
the name; next time a similar crop is seen, the learned name overrides the
detector's generic COCO label ("cup" → "French press"). The embed→cosine→
threshold pattern, a third time.

Embedders:
  * ClipImageEmbedder — a CLIP-style image encoder ONNX (config path). Strong
    open-world features; use it if you export one.
  * HistEmbedder — offline fallback: HSV color histogram + coarse gradient-
    orientation histogram over a grid. Weak by research standards but honest
    and dependency-free: it reliably separates visually distinct household
    objects, which is the actual job.

Both produce unit vectors; LearnedObjects stores per-name samples (capped) and
matches by max cosine, exactly like the person registry.
"""
from __future__ import annotations

import sqlite3
import time

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("vision.learned")


class HistEmbedder:
    """HSV histogram (color) + gradient-orientation histogram (shape), gridded.
    Deterministic, no deps beyond numpy/cv2."""

    name = "hist"

    def __init__(self, grid: int = 3, hue_bins: int = 12, ori_bins: int = 8):
        self.grid = int(grid)
        self.hue_bins = int(hue_bins)
        self.ori_bins = int(ori_bins)

    def embed(self, crop_rgb: np.ndarray) -> np.ndarray | None:
        import cv2

        if crop_rgb is None or crop_rgb.size == 0:
            return None
        img = cv2.resize(crop_rgb, (96, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        ori = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # 0..1

        cell = 96 // self.grid
        feats: list[np.ndarray] = []
        for r in range(self.grid):
            for c in range(self.grid):
                sl = (slice(r * cell, (r + 1) * cell),
                      slice(c * cell, (c + 1) * cell))
                h = np.histogram(hsv[..., 0][sl], bins=self.hue_bins,
                                 range=(0, 180), weights=hsv[..., 1][sl] / 255.0)[0]
                o = np.histogram(ori[sl], bins=self.ori_bins, range=(0, 1),
                                 weights=mag[sl])[0]
                feats.append(h.astype(np.float32))
                feats.append(o.astype(np.float32))
        v = np.concatenate(feats)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None


class ClipImageEmbedder:
    """A CLIP-style image encoder exported to ONNX (224x224 RGB in, vector out)."""

    name = "clip"

    def __init__(self, model_path: str):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(model_path,
                                          providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        log.info("CLIP image embedder loaded (%s)", model_path)

    def embed(self, crop_rgb: np.ndarray) -> np.ndarray | None:
        import cv2

        if crop_rgb is None or crop_rgb.size == 0:
            return None
        img = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None]
        out = self._sess.run(None, {self._input: x})[0][0]
        emb = np.asarray(out, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(emb))
        return emb / n if n > 0 else None


class LearnedObjects:
    """SQLite store + matcher for user-taught object names."""

    def __init__(self, path: str, embedder, match_threshold: float = 0.80,
                 max_samples_per_object: int = 5):
        self._embedder = embedder
        self.threshold = float(match_threshold)
        self.max_samples = max(1, int(max_samples_per_object))
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS objects("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT COLLATE NOCASE, "
            "person_id INTEGER, dim INTEGER, vec BLOB, created_at REAL)"
        )
        self._db.commit()
        log.info("learned objects ready (%s, %d names, embed=%s)",
                 path, self.name_count(), embedder.name)

    def name_count(self) -> int:
        return self._db.execute(
            "SELECT COUNT(DISTINCT name) FROM objects").fetchone()[0]

    def names(self) -> list[str]:
        return [r[0] for r in self._db.execute(
            "SELECT DISTINCT name FROM objects ORDER BY name")]

    def teach(self, name: str, crop_rgb: np.ndarray,
              person_id: int | None = None) -> bool:
        """Bind this crop's embedding to ``name``. False if no usable embedding."""
        name = " ".join(name.strip().split())[:60]
        if not name:
            return False
        v = self._embedder.embed(crop_rgb)
        if v is None:
            return False
        self._db.execute(
            "INSERT INTO objects(name, person_id, dim, vec, created_at) "
            "VALUES(?,?,?,?,?)",
            (name, person_id, v.size, v.astype(np.float32).tobytes(), time.time()))
        self._db.execute(
            "DELETE FROM objects WHERE name=? COLLATE NOCASE AND id NOT IN ("
            "SELECT id FROM objects WHERE name=? COLLATE NOCASE "
            "ORDER BY id DESC LIMIT ?)", (name, name, self.max_samples))
        self._db.commit()
        log.info("learned object: %r (%d samples)", name,
                 self._db.execute(
                     "SELECT COUNT(*) FROM objects WHERE name=? COLLATE NOCASE",
                     (name,)).fetchone()[0])
        return True

    def match(self, crop_rgb: np.ndarray) -> tuple[str, float] | None:
        """(name, score) of the best learned match above threshold, or None."""
        # Nothing taught yet -> never embed. Embedding is expensive (a remote
        # CLIP round trip per crop); doing it with an empty store just to find
        # zero rows to compare against was stalling every conversation turn.
        if self.name_count() == 0:
            return None
        v = self._embedder.embed(crop_rgb)
        if v is None:
            return None
        best: tuple[str, float] | None = None
        for name, dim, blob in self._db.execute(
                "SELECT name, dim, vec FROM objects"):
            if dim != v.size:
                continue
            s = float(np.dot(np.frombuffer(blob, dtype=np.float32), v))
            if best is None or s > best[1]:
                best = (name, s)
        if best is not None and best[1] >= self.threshold:
            return best
        return None

    def forget(self, name: str) -> bool:
        cur = self._db.execute(
            "DELETE FROM objects WHERE name=? COLLATE NOCASE", (name.strip(),))
        self._db.commit()
        return cur.rowcount > 0

    # ── per-turn annotation ──────────────────────────────────────────────────
    def annotate(self, frame_rgb: np.ndarray, detections: list) -> list:
        """Return detections with labels overridden by learned names where a
        crop matches. Pure function of the inputs — originals untouched."""
        if frame_rgb is None or not detections:
            return list(detections)
        if self.name_count() == 0:   # nothing taught -> skip the embed loop entirely
            return list(detections)
        H, W = frame_rgb.shape[:2]
        out = []
        for det in detections:
            try:
                x, y, w, h = (int(v) for v in det.bbox)
                x0, y0 = max(0, x), max(0, y)
                crop = frame_rgb[y0:min(H, y + h), x0:min(W, x + w)]
                hit = self.match(crop) if crop.size else None
            except Exception as e:  # annotation must never break perception
                log.debug("annotate failed: %s", e)
                hit = None
            if hit is not None:
                out.append(det.model_copy(update={"label": hit[0]}))
            else:
                out.append(det)
        return out


# ── teach-phrase parsing (main.py) ────────────────────────────────────────────
import re as _re

# "this is a french press", "that's my espresso cup", "remember this is the
# blender". The article/possessive is what separates OBJECT teaching from
# PERSON enrolment ("this is Peter" has no article).
_TEACH_RE = _re.compile(
    r"\b(?:remember[,;]?\s+)?(?:this|that|it)(?:'s| is)\s+"
    r"(?:called\s+)?(?:a|an|my|the|our)\s+([a-z][a-z0-9' \-]{2,40})$",
    _re.IGNORECASE)


def parse_object_teach(text: str) -> str | None:
    """The object name being taught, or None."""
    m = _TEACH_RE.search(text.strip().rstrip(".!?"))
    if not m:
        return None
    name = " ".join(m.group(1).lower().split())
    return name if 3 <= len(name) <= 40 else None


def build_object_embedder(clip_model_path: str | None):
    """CLIP if a model is configured and loads; histogram fallback otherwise."""
    if clip_model_path:
        try:
            return ClipImageEmbedder(clip_model_path)
        except Exception as e:
            log.warning("CLIP embedder unavailable (%s) — using histogram", e)
    return HistEmbedder()
