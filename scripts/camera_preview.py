"""Live camera preview — see exactly what ZERO's eyes see, without the voice stack.

Runs ONLY the camera + detector + color loop from ``config.yaml`` and opens a
window with detection boxes drawn on the feed. Handy to check the camera, aim it,
and compare detectors (yolo11s vs the YOLO-World open-vocab export) before wiring
up the full assistant. No mic, wake word, Ollama or GPU tunnel required.

    python scripts/camera_preview.py

Press 'q' in the window to close the preview, or Ctrl-C in the terminal to quit.
"""
from __future__ import annotations

import time

from zero.config import load_config
from zero.factory import build_vision


def main() -> int:
    cfg = load_config()
    # Force vision + preview on for this tool, whatever config.yaml says, and skip
    # the GPU depth server (not needed just to look at the camera).
    vis = cfg.raw.setdefault("vision", {})
    vis["enabled"] = True
    vis["preview"] = True
    vis.setdefault("gpu", {})["enabled"] = False

    eyes = build_vision(cfg)
    if eyes is None:
        print("Vision could not start — is a camera connected and "
              "requirements-vision.txt installed?")
        return 1

    model = cfg.get("vision.detect.model_path", "yolo11n.onnx")
    print(f"Camera preview using {model}. "
          f"Press 'q' in the window or Ctrl-C here to quit.")
    eyes.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        eyes.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
