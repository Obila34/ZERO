"""CP5 eval: run a few saved frames + questions through the full /analyze path.

Exercises the same in-process pipeline the HTTP endpoint uses
(``app.run_analyze`` -> CP4 facts -> VLM), printing the structured facts and the
spoken reply for each (frame, question) pair. Use it to spot-check that the
reply stays grounded (distances/colors match the facts) and that follow-up
questions using ``history`` answer sensibly.

Detections
----------
Facts need 2D detections (normally produced by the Pi's YOLO in CP1). Since the
GPU box may not run the detector, this script reads them from an optional sidecar
JSON next to each frame: for ``frame.jpg`` it looks for ``frame.json`` holding a
list of ``Detection`` dicts ``[{"label","bbox":[x,y,w,h],"confidence","color"}]``.
With no sidecar it sends an empty detection list — the VLM still answers from the
image, just without grounded distances.

Usage::

    python -m server.vlm.test_analyze                 # default sample frames
    python -m server.vlm.test_analyze img1.jpg img2.jpg
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from server.app import run_analyze
    from server.shared.schemas import AnalyzeRequest, Detection
except ImportError:  # pragma: no cover - import path fallback (cwd=server/)
    from app import run_analyze
    from shared.schemas import AnalyzeRequest, Detection

SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SERVER_DIR.parent

# Where to look for sample frames when none are passed on the CLI.
_DEFAULT_FRAME_DIRS = [
    SERVER_DIR / "test_frames",
    REPO_DIR / "pi" / "captures",
]

# Each frame is asked the opener, then a follow-up that relies on history.
_QUESTIONS = ["What are you looking at?", "Is it close to me?"]


def _load_detections(frame_path: Path) -> list[Detection]:
    sidecar = frame_path.with_suffix(".json")
    if not sidecar.exists():
        return []
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        return [Detection(**d) for d in raw]
    except Exception as exc:  # malformed sidecar shouldn't abort the eval
        print(f"[test_analyze] ignoring bad sidecar {sidecar.name}: {exc}", file=sys.stderr)
        return []


def _b64(frame_path: Path) -> str:
    return base64.b64encode(frame_path.read_bytes()).decode("ascii")


def _discover_frames() -> list[Path]:
    frames: list[Path] = []
    for d in _DEFAULT_FRAME_DIRS:
        if d.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                frames.extend(sorted(d.glob(ext)))
    return frames


def run(frame_paths: list[Path]) -> int:
    if not frame_paths:
        searched = ", ".join(str(d) for d in _DEFAULT_FRAME_DIRS)
        print(
            "[test_analyze] no frames given and none found in: "
            f"{searched}\nPass image paths, or drop a few JPEGs in server/test_frames/.",
            file=sys.stderr,
        )
        return 1

    for frame_path in frame_paths:
        if not frame_path.exists():
            print(f"[test_analyze] skipping missing frame: {frame_path}", file=sys.stderr)
            continue

        b64 = _b64(frame_path)
        detections = _load_detections(frame_path)
        print(f"\n=== {frame_path.name}  ({len(detections)} detections) ===")

        history: list[dict] = []
        for question in _QUESTIONS:
            req = AnalyzeRequest(
                image_jpeg_b64=b64,
                detections=detections,
                question=question,
                history=list(history),
            )
            resp = run_analyze(req)

            facts_str = (
                "; ".join(
                    f"{f.label}"
                    + (f"/{f.color}" if f.color else "")
                    + (f" {f.distance_m}m" if f.distance_m is not None else "")
                    + (f" {f.bearing}" if f.bearing else "")
                    for f in resp.facts
                )
                or "(none)"
            )
            print(f"facts: {facts_str}")
            print(f"Q: {question}")
            print(f"A: {resp.reply}")

            # Carry the turn into history so the follow-up has context.
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": resp.reply})

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    frames = [Path(a) for a in argv] if argv else _discover_frames()
    return run(frames)


if __name__ == "__main__":
    raise SystemExit(main())
