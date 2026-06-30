"""Turn detections + a metric depth map into grounded scene facts (CP4).

``to_scene_facts`` is the core of the GPU spatial step. For each 2D detection it:

1. Reads the depth over the detection's box (central crop, to drop the
   background ring around the object), takes the **median** valid depth — robust
   to the depth model's edge bleed and to a few bad pixels.
2. Back-projects the box-center pixel + that depth through the camera intrinsics
   to a metric ``(X, Y, Z)`` point in the camera frame.
3. Derives ``distance_m`` (Euclidean range) and a coarse ``bearing``
   ("left" / "center" / "right", optionally with "high"/"low") from the bearing
   angle ``atan2(X, Z)`` — i.e. from the intrinsics, not a pixel heuristic.

Coordinate conventions (OpenCV camera frame): ``+X`` right, ``+Y`` down,
``+Z`` forward (away from the camera). Pixel ``u`` grows rightward, ``v``
downward. So a positive bearing angle means the object is to the viewer's right.

All inputs must share one pixel space: the ``depth_map`` is expected at the same
resolution as the frame the ``detections`` boxes were measured in, and
``intrinsics`` scaled to match (see ``DepthEstimator.estimate`` and
``load_intrinsics``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    from facts.intrinsics import Intrinsics
    from shared.schemas import Detection, SceneFact
except ImportError:  # pragma: no cover - import path fallback (cwd=repo root)
    from server.facts.intrinsics import Intrinsics
    from server.shared.schemas import Detection, SceneFact


# Tunable bearing/region defaults; ``to_scene_facts`` callers override from config.
DEFAULTS = {
    "bbox_center_crop": 0.6,   # use central 60% of the box for the depth median
    "center_bearing_deg": 12.0,  # |horizontal angle| below this -> "center"
    "vertical_bearing": False,   # also emit "high"/"low"
    "vertical_deg": 18.0,        # |vertical angle| above this -> high/low
    "min_valid_pixels": 10,      # need at least this many good depth samples
}


def _median_region_depth(
    depth_map: np.ndarray, bbox: list[float], center_crop: float, min_valid: int
) -> Optional[float]:
    """Median of finite, positive depths in the central crop of ``bbox``.

    Returns None when the box is off-frame or has too few valid depth samples.
    """
    h, w = depth_map.shape[:2]
    x, y, bw, bh = bbox

    # Shrink the box toward its center to drop the background ring.
    crop = float(np.clip(center_crop, 0.05, 1.0))
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    half_w = (bw * crop) / 2.0
    half_h = (bh * crop) / 2.0

    x0 = int(max(0, math.floor(cx - half_w)))
    y0 = int(max(0, math.floor(cy - half_h)))
    x1 = int(min(w, math.ceil(cx + half_w)))
    y1 = int(min(h, math.ceil(cy + half_h)))
    if x1 <= x0 or y1 <= y0:
        return None

    region = depth_map[y0:y1, x0:x1]
    valid = region[np.isfinite(region) & (region > 0)]
    if valid.size < min_valid:
        return None
    return float(np.median(valid))


def _bearing(
    x_m: float,
    y_m: float,
    z_m: float,
    center_deg: float,
    vertical: bool,
    vertical_deg: float,
) -> str:
    """Coarse direction word(s) from a metric point in the camera frame."""
    # Horizontal angle: +ve = object to the viewer's right.
    h_angle = math.degrees(math.atan2(x_m, max(z_m, 1e-6)))
    if h_angle < -center_deg:
        horizontal = "left"
    elif h_angle > center_deg:
        horizontal = "right"
    else:
        horizontal = "center"

    if not vertical:
        return horizontal

    # Vertical angle: +Y is down, so flip the sign for "high"/"low".
    v_angle = math.degrees(math.atan2(-y_m, max(z_m, 1e-6)))
    if v_angle > vertical_deg:
        return f"high {horizontal}"
    if v_angle < -vertical_deg:
        return f"low {horizontal}"
    return horizontal


def to_scene_facts(
    frame: np.ndarray,
    detections: list[Detection],
    depth_map: np.ndarray,
    intrinsics: Intrinsics,
    config: Optional[dict] = None,
) -> list[SceneFact]:
    """Build one ``SceneFact`` per detection (skipping any with no valid depth).

    ``frame`` is accepted for shape reference / future use (e.g. masking); the
    depth/geometry comes from ``depth_map`` + ``intrinsics``, which must be in
    the same pixel space as the detection boxes.
    """
    cfg = {**DEFAULTS, **(config or {})}
    center_crop = float(cfg["bbox_center_crop"])
    center_deg = float(cfg["center_bearing_deg"])
    vertical = bool(cfg["vertical_bearing"])
    vertical_deg = float(cfg["vertical_deg"])
    min_valid = int(cfg["min_valid_pixels"])

    facts: list[SceneFact] = []
    for det in detections:
        depth = _median_region_depth(depth_map, det.bbox, center_crop, min_valid)
        if depth is None:
            # No usable depth (off-frame box, or model gave no signal there):
            # still report the object, just without spatial grounding.
            facts.append(SceneFact(label=det.label, color=det.color))
            continue

        x, y, bw, bh = det.bbox
        u = x + bw / 2.0
        v = y + bh / 2.0

        # Pixel + depth -> metric point via pinhole back-projection.
        z_m = depth
        x_m = (u - intrinsics.cx) * z_m / intrinsics.fx
        y_m = (v - intrinsics.cy) * z_m / intrinsics.fy
        distance_m = math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)

        bearing = _bearing(x_m, y_m, z_m, center_deg, vertical, vertical_deg)

        facts.append(
            SceneFact(
                label=det.label,
                color=det.color,
                distance_m=round(distance_m, 2),
                bearing=bearing,
            )
        )
    return facts
