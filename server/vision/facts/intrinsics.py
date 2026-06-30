"""Camera intrinsics for the GPU depth->metric step (CP4).

The intrinsics describe the *camera that took the frame* (focal lengths and
principal point in pixels). They are produced by ``calibration/calibrate.py``
on the node that owns the camera, then **copied to the GPU** and pointed at by
``server/config.yaml -> camera.intrinsics_path`` — the two machines do not share
a filesystem.

If no real calibration is available (file missing, or still holding the CP0
``null`` placeholder), we fall back to a coarse estimate derived from the
horizontal field of view and the frame resolution. That keeps ``/facts`` working
out of the box, but distances/bearings are approximate and the fallback is
clearly logged.

Intrinsics are always returned **scaled to the resolution of the frame the
detections were measured in**, so depth map, bounding boxes and intrinsics share
one pixel coordinate space.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels, valid for a (width, height) image."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    approximate: bool = False  # True when derived from FOV, not calibration

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Return a copy rescaled to a different image resolution.

        Focal length and principal point scale linearly with resolution.
        """
        if width == self.width and height == self.height:
            return self
        sx = width / float(self.width)
        sy = height / float(self.height)
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
            approximate=self.approximate,
        )


def default_intrinsics(width: int, height: int, hfov_deg: float) -> Intrinsics:
    """Coarse intrinsics from horizontal FOV + resolution (no calibration).

    ``fx = (width / 2) / tan(hfov / 2)``; assumes square pixels (``fy == fx``)
    and a centered principal point. Distortion is ignored.
    """
    hfov = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    return Intrinsics(
        fx=fx,
        fy=fx,
        cx=width / 2.0,
        cy=height / 2.0,
        width=int(width),
        height=int(height),
        approximate=True,
    )


def _from_calibration_file(path: Path) -> Optional[Intrinsics]:
    """Parse a calibrate.py ``intrinsics.json``; None if missing/placeholder."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    K = data.get("camera_matrix")
    w = data.get("image_width")
    h = data.get("image_height")
    if not K or not w or not h:
        return None  # CP0 placeholder (null fields) or incomplete

    try:
        fx = float(K[0][0])
        fy = float(K[1][1])
        cx = float(K[0][2])
        cy = float(K[1][2])
    except (TypeError, IndexError, ValueError):
        return None

    # A file written by the FOV fallback carries approximate=true; preserve it
    # so the server still warns that distances are rough.
    approximate = bool(data.get("approximate", False))
    return Intrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, width=int(w), height=int(h), approximate=approximate
    )


def load_intrinsics(
    path: str | Path | None,
    *,
    target_width: int,
    target_height: int,
    fallback_hfov_deg: float,
) -> Intrinsics:
    """Load calibrated intrinsics (scaled to the target frame) or fall back.

    ``path`` is the configured intrinsics file. When it is missing or still a
    placeholder, a FOV-based estimate at the target resolution is returned with
    ``approximate=True`` (callers should log that).
    """
    calibrated = _from_calibration_file(Path(path)) if path else None
    if calibrated is not None:
        return calibrated.scaled_to(target_width, target_height)
    return default_intrinsics(target_width, target_height, fallback_hfov_deg)
