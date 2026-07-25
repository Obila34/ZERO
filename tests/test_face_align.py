"""Face path: 5-point ArcFace alignment + detector backend selection/fallback.

The MediaPipe end-to-end (real face in, embedding out) is exercised by
scripts/smoke_face.py against a real photo; here we pin down the geometry and
the degradation contract, which must hold on any box regardless of which
optional wheels are installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from zero.identity.face import _ARC_SIZE, _ARC_TEMPLATE, align_5pt

REPO = Path(__file__).resolve().parents[1]
ARCFACE = REPO / "models" / "identity" / "arcface.onnx"
LANDMARKER = REPO / "models" / "face" / "face_landmarker.task"
YUNET = REPO / "models" / "face" / "yunet_2023mar.onnx"


def _similarity(pts: np.ndarray, angle_deg: float, scale: float,
                tx: float, ty: float) -> np.ndarray:
    """Apply rotation+scale+translation to Nx2 points."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (pts @ R.T) * scale + [tx, ty]


# ── alignment geometry ────────────────────────────────────────────────────────
@pytest.mark.parametrize("angle,scale,tx,ty", [
    (0, 1.0, 0, 0),          # identity placement
    (25, 1.8, 140, 90),      # tilted, larger, off-center
    (-40, 0.9, 60, 200),     # tilted the other way, smaller
])
def test_align_maps_points_back_to_template(angle, scale, tx, ty):
    src = _similarity(_ARC_TEMPLATE, angle, scale, tx, ty).astype(np.float32)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for (x, y), c in zip(src, colors):
        cv2.circle(canvas, (int(round(x)), int(round(y))),
                   max(3, int(4 * scale)), c, -1)
    aligned = align_5pt(canvas, src)
    assert aligned is not None and aligned.shape == (_ARC_SIZE, _ARC_SIZE, 3)
    # Each colored dot must land on its template coordinate after the warp.
    for (tx_, ty_), c in zip(_ARC_TEMPLATE, colors):
        px = aligned[int(round(ty_)), int(round(tx_))]
        assert tuple(px) == c, f"point expected {c} at ({tx_:.0f},{ty_:.0f}), got {tuple(px)}"


def test_align_degenerate_points_returns_none():
    # All five points identical — no transform can be estimated.
    pts = np.tile(np.float32([50, 50]), (5, 1))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert align_5pt(frame, pts) is None


# ── backend selection / degradation ──────────────────────────────────────────
needs_arcface = pytest.mark.skipif(not ARCFACE.exists(),
                                   reason="arcface.onnx not downloaded")

# opencv-python >= 5.0 wheels ship no cascade data, so Haar-backend
# expectations are environment-conditional.
HAAR_OK = Path(cv2.data.haarcascades,
               "haarcascade_frontalface_default.xml").exists()


@needs_arcface
def test_no_landmarker_path_uses_haar_or_raises():
    from zero.identity.face import FaceRecognizer

    if HAAR_OK:
        fr = FaceRecognizer(str(ARCFACE), landmarker_path=None)
        assert fr.backend == "haar"
    else:
        # No detector at all must raise cleanly at construction — the
        # Fallback* wrapper logs it and marks the local channel broken.
        with pytest.raises(RuntimeError):
            FaceRecognizer(str(ARCFACE), landmarker_path=None)


@needs_arcface
@pytest.mark.skipif(not HAAR_OK, reason="cv2 wheel ships no Haar cascades")
def test_missing_landmarker_file_degrades_to_haar_not_error():
    from zero.identity.face import FaceRecognizer

    fr = FaceRecognizer(str(ARCFACE), landmarker_path="/nonexistent/lm.task")
    assert fr.backend == "haar"
    # And the degraded instance still honours the never-raises contract.
    assert fr.embed_faces(np.zeros((240, 320, 3), dtype=np.uint8)) == []


@needs_arcface
@pytest.mark.skipif(not YUNET.exists(), reason="yunet model not downloaded")
def test_yunet_tiers_and_demotion():
    from zero.identity.face import FaceRecognizer

    has_mp = LANDMARKER.exists()
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        has_mp = False
    fr = FaceRecognizer(str(ARCFACE), detector_path=str(YUNET),
                        landmarker_path=str(LANDMARKER) if has_mp else None)
    assert fr.backend == ("yunet+mediapipe" if has_mp else "yunet")
    # Faceless frames: empty result, no exception, no demotion.
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert fr.embed_faces(noise) == []
    assert fr.backend.startswith("yunet")

    class Broken:
        def detect_faces(self, *_a):
            raise RuntimeError("wedged")

    fr._yunet = Broken()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert fr.embed_faces(frame) == []          # absorbed, not raised
    assert not fr.backend.startswith("yunet")   # tier demoted for the session
    assert fr.embed_faces(frame) == []          # next tier still answers


@needs_arcface
@pytest.mark.skipif(not LANDMARKER.exists(),
                    reason="face_landmarker.task not downloaded")
def test_mediapipe_backend_loads_and_never_raises():
    pytest.importorskip("mediapipe")
    from zero.identity.face import FaceRecognizer

    fr = FaceRecognizer(str(ARCFACE), landmarker_path=str(LANDMARKER))
    assert fr.backend == "mediapipe"
    # Faceless frames: noise and flat — empty result, no exception, no demotion.
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert fr.embed_faces(noise) == []
    assert fr.embed_faces(np.zeros((240, 320, 3), dtype=np.uint8)) == []
    assert fr.backend == "mediapipe"


@needs_arcface
@pytest.mark.skipif(not LANDMARKER.exists(),
                    reason="face_landmarker.task not downloaded")
def test_runtime_mediapipe_failure_demotes_without_raising():
    pytest.importorskip("mediapipe")
    from zero.identity.face import FaceRecognizer

    fr = FaceRecognizer(str(ARCFACE), landmarker_path=str(LANDMARKER))

    class Broken:
        def detect(self, *_a, **_k):
            raise RuntimeError("wedged")

    fr._landmarker = Broken()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert fr.embed_faces(frame) == []   # absorbed, not raised
    # Demoted for the rest of the session: Haar where available, else blind —
    # but never back on the broken landmarker, and never an exception.
    assert fr.backend == ("haar" if HAAR_OK else "none")
    assert fr.embed_faces(frame) == []
    assert fr.detect(frame) == []
