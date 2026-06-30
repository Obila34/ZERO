"""Monocular depth with Depth Anything V2 (CP4, GPU side).

``DepthEstimator`` wraps a Depth Anything V2 checkpoint (via 🤗 Transformers) and
turns one frame into a depth map **aligned to the input frame's resolution**, so
the spatial step can index it with detection boxes that live in that same pixel
space (see ``facts/spatial.py``).

Metric vs relative
------------------
Depth Anything V2 ships two families:

* **Metric** checkpoints (e.g. ``Depth-Anything-V2-Metric-Indoor-Small-hf``) —
  the default here. ``predicted_depth`` is already in **meters** (indoor models
  are trained to ~20 m), so a median over a box is directly a distance in
  meters. This is what makes "about 1.2 m to your left" truthful.
* **Relative** checkpoints (e.g. ``Depth-Anything-V2-Small-hf``) — output is
  affine-invariant *inverse* depth (unitless, larger = nearer). With
  ``metric: false`` we expose it multiplied by ``relative_scale`` as a coarse
  pseudo-metric value and log loudly that distances are approximate. Prefer a
  metric checkpoint for real distance words.

VRAM: the Small variant is a few hundred MB and sits comfortably alongside the
CP5 VLM in 12–16 GB. Pick Base/Large via ``depth.model`` in ``server/config.yaml``.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np


class DepthEstimator:
    """Loads Depth Anything V2 and produces frame-resolution depth maps."""

    def __init__(self, config: Optional[dict] = None) -> None:
        try:
            from config import load_config
        except ImportError:  # pragma: no cover - import path fallback
            from server.config import load_config

        cfg = config or load_config()
        depth = cfg.get("depth", {})
        runtime = cfg.get("runtime", {})

        self._model_id = str(depth.get("model", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"))
        self._metric = bool(depth.get("metric", True))
        self._relative_scale = float(depth.get("relative_scale", 1.0))
        self._max_depth_m = float(depth.get("max_depth_m", 20.0))
        self._cache_dir = runtime.get("cache_dir") or None
        self._device_pref = str(runtime.get("device", "auto"))

        self._model = None
        self._processor = None
        self._device = None  # resolved on first load

    # ── loading ──────────────────────────────────────────────────────────────
    def _resolve_device(self) -> str:
        import torch

        if self._device_pref == "cpu":
            return "cpu"
        if self._device_pref == "cuda":
            return "cuda"
        # 'auto'
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "torch + transformers are required for depth. Install them from "
                "server/requirements.txt (a CUDA torch build for the GPU)."
            ) from exc

        self._device = self._resolve_device()
        print(
            f"[depth] loading {self._model_id} on {self._device} "
            f"(metric={self._metric})",
            file=sys.stderr,
        )
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        self._model = model.to(self._device).eval()

    # ── inference ────────────────────────────────────────────────────────────
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:  # noqa: D401
        """Return a ``float32`` depth map the same H×W as ``frame_bgr``.

        Units are meters for a metric checkpoint; ``relative_scale``-scaled
        pseudo-meters for a relative one. The result is resized back to the
        input resolution so it co-registers with the detection boxes.
        """
        import cv2
        import torch

        self._ensure_model()
        assert self._model is not None and self._processor is not None

        h, w = frame_bgr.shape[:2]
        # Transformers image processors expect RGB and a contiguous array
        # (cvtColor returns a fresh contiguous buffer — a `[..., ::-1]` view has
        # negative strides that torch.from_numpy rejects).
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        inputs = self._processor(images=frame_rgb, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Prefer the processor's post-processing (handles resize + any model
        # specific scaling); fall back to manual interpolation if unavailable.
        depth: Optional[np.ndarray] = None
        post = getattr(self._processor, "post_process_depth_estimation", None)
        if post is not None:
            try:
                result = post(outputs, target_sizes=[(h, w)])
                depth = result[0]["predicted_depth"].detach().cpu().numpy()
            except Exception:  # pragma: no cover - older/newer API shapes
                depth = None

        if depth is None:
            pred = outputs.predicted_depth  # (B, h', w')
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze()
            depth = pred.detach().cpu().numpy()

        depth = np.asarray(depth, dtype=np.float32)

        if not self._metric:
            # Relative inverse-depth -> coarse pseudo-metric. Approximate.
            depth = depth * self._relative_scale

        # Clamp absurd values (model noise / sky) to keep medians sane.
        np.clip(depth, 0.0, self._max_depth_m, out=depth)
        return depth

    # ── debug ────────────────────────────────────────────────────────────────
    @staticmethod
    def colorize(depth_map: np.ndarray) -> np.ndarray:
        """Colorized BGR image of a depth map for the debug dump (near = warm)."""
        import cv2

        d = np.asarray(depth_map, dtype=np.float32)
        finite = d[np.isfinite(d)]
        if finite.size == 0:
            return np.zeros((*d.shape, 3), dtype=np.uint8)
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-6:
            norm = np.zeros_like(d)
        else:
            norm = (d - lo) / (hi - lo)
        # Invert so nearer (smaller depth) is brighter/warmer.
        u8 = (255.0 * (1.0 - np.clip(norm, 0.0, 1.0))).astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def main(argv: Optional[list[str]] = None) -> int:
    """Smoke test: ``python -m server.depth.depth_anything <image.jpg>``."""
    import cv2

    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m server.depth.depth_anything <image_path>", file=sys.stderr)
        return 2
    frame = cv2.imread(argv[0])
    if frame is None:
        print(f"[depth] could not read image: {argv[0]}", file=sys.stderr)
        return 1

    est = DepthEstimator()
    depth = est.estimate(frame)
    print(
        f"[depth] map {depth.shape} dtype={depth.dtype} "
        f"min={depth.min():.2f} median={np.median(depth):.2f} max={depth.max():.2f}"
    )
    out = "depth_debug.png"
    cv2.imwrite(out, est.colorize(depth))
    print(f"[depth] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
