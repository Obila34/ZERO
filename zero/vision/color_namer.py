"""Per-object dominant-color naming (Pi-side, local).

Given an RGB frame and a bounding box, crop the object, work in HSV, drop the
background-ish edge pixels, and reduce the rest to a single human color word
(``red``, ``blue``, ...). Achromatic crops resolve to ``black`` / ``gray`` /
``white``. Returns ``None`` when the crop is too small or washed-out to call.

A hue histogram (not K-Means): dependency-free beyond OpenCV/numpy, fast enough
to run every frame on the Pi, robust to stray pixels.
"""
from __future__ import annotations

from typing import Optional, Sequence

# Hue ranges in OpenCV's 0-179 scale -> color word. Red wraps around both ends.
_HUE_BANDS: list[tuple[int, int, str]] = [
    (0, 9, "red"), (10, 20, "orange"), (21, 33, "yellow"), (34, 85, "green"),
    (86, 95, "cyan"), (96, 130, "blue"), (131, 145, "purple"), (146, 159, "pink"),
    (160, 179, "red"),
]


def _hue_to_word(hue: int, sat: float, val: float) -> str:
    for low, high, name in _HUE_BANDS:
        if low <= hue <= high:
            if name in ("orange", "yellow", "red") and val < 160 and sat < 200:
                if val < 130:
                    return "brown"
            return name
    return "red"


class ColorNamer:
    def __init__(self, center_crop: float = 0.6, min_saturation: int = 50,
                 min_value: int = 40, white_value: int = 200,
                 min_colorful_ratio: float = 0.10, min_pixels: int = 50):
        self._center_crop = float(center_crop)
        self._min_sat = int(min_saturation)
        self._min_val = int(min_value)
        self._white_val = int(white_value)
        self._min_colorful_ratio = float(min_colorful_ratio)
        self._min_pixels = int(min_pixels)

    def name(self, frame_rgb, bbox: Sequence[float]) -> Optional[str]:
        """Return a color word for the object in ``bbox`` ([x, y, w, h]), or None."""
        import cv2
        import numpy as np

        crop = self._center_region(frame_rgb, bbox)
        if crop is None or crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].reshape(-1)
        s = hsv[:, :, 1].reshape(-1)
        v = hsv[:, :, 2].reshape(-1)
        total = h.size
        if total < self._min_pixels:
            return None

        chromatic = (s >= self._min_sat) & (v >= self._min_val)
        colorful_ratio = float(np.count_nonzero(chromatic)) / float(total)
        if colorful_ratio >= self._min_colorful_ratio:
            hues = h[chromatic]
            weights = (s[chromatic].astype(np.float32) / 255.0) * (
                v[chromatic].astype(np.float32) / 255.0)
            hist = np.bincount(hues, weights=weights, minlength=180)
            dominant_hue = int(np.argmax(hist))
            mean_sat = float(np.mean(s[chromatic]))
            mean_val = float(np.mean(v[chromatic]))
            return _hue_to_word(dominant_hue, mean_sat, mean_val)

        return self._achromatic(v)

    def _achromatic(self, v) -> str:
        import numpy as np

        mean_val = float(np.mean(v))
        if mean_val < self._min_val:
            return "black"
        if mean_val >= self._white_val:
            return "white"
        return "gray"

    def _center_region(self, frame_rgb, bbox: Sequence[float]):
        H, W = frame_rgb.shape[:2]
        x, y, w, h = (float(v) for v in bbox)
        keep = max(0.1, min(1.0, self._center_crop))
        cx, cy = x + w / 2.0, y + h / 2.0
        nw, nh = w * keep, h * keep
        x0, y0 = int(round(cx - nw / 2.0)), int(round(cy - nh / 2.0))
        x1, y1 = int(round(cx + nw / 2.0)), int(round(cy + nh / 2.0))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return frame_rgb[y0:y1, x0:x1]
