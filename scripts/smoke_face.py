"""Smoke-test the local face path on a real photo.

    python scripts/smoke_face.py <image.jpg>

Loads FaceRecognizer (full YuNet+MediaPipe stack), detects + embeds every
face, then re-runs on a 20-degree-rotated copy and checks that each face's
embedding stays stable across the rotation (faces paired by mapping bbox
centers through the known rotation) while distinct faces stay separated.
Pass/fail is printed per check; exit code 1 if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero.identity.face import FaceRecognizer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def rotate(frame: np.ndarray, deg: float) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(frame, M, (w, h))


def main() -> int:
    import cv2

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not img_path or not Path(img_path).exists():
        print("usage: python scripts/smoke_face.py <image-with-faces.jpg>")
        return 1
    frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    fr = FaceRecognizer(
        str(REPO / "models/identity/arcface.onnx"),
        detector_path=str(REPO / "models/face/yunet_2023mar.onnx"),
        landmarker_path=str(REPO / "models/face/face_landmarker.task"),
    )
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok &= cond

    print(f"backend: {fr.backend}")
    check(fr.backend == "yunet+mediapipe", "full tier stack active")

    faces = fr.embed_faces(frame, max_faces=4)
    check(len(faces) >= 1, f"faces found: {len(faces)}")
    for emb, bbox in faces:
        check(abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3,
              f"unit-norm 512-d embedding at bbox {bbox} (dim={emb.size})")

    import cv2

    deg = 20.0
    rot = rotate(frame, deg)
    faces_rot = fr.embed_faces(rot, max_faces=4)
    check(len(faces_rot) >= 1, f"faces found in {deg:.0f}deg-rotated copy: {len(faces_rot)}")

    # Pair faces across the two views by mapping bbox centers through the
    # known rotation — "largest first" order is not stable across views.
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    for emb, (x, y, bw, bh) in faces:
        cx, cy = M @ [x + bw / 2, y + bh / 2, 1]
        best = min(
            faces_rot,
            key=lambda f: (f[1][0] + f[1][2] / 2 - cx) ** 2
                          + (f[1][1] + f[1][3] / 2 - cy) ** 2,
        )
        dist = np.hypot(best[1][0] + best[1][2] / 2 - cx,
                        best[1][1] + best[1][3] / 2 - cy)
        if dist > max(bw, bh) / 2:
            continue  # this face left the frame or went undetected when rotated
        cos = float(np.dot(emb, best[0]))
        check(cos > 0.60,
              f"same face at ({x},{y}) stable across rotation (cosine={cos:.3f})")

    # Different people must stay far apart (identity face_threshold is 0.42).
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            cos = float(np.dot(faces[i][0], faces[j][0]))
            check(cos < 0.42,
                  f"distinct faces {i} vs {j} separated (cosine={cos:.3f})")

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
