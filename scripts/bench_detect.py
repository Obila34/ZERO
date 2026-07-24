"""Benchmark object detectors on the two real deployment paths.

* ``cpu``  — the Pi reflex path: ONNX exports through ``zero.vision.detector``
  on onnxruntime/CPU. Run it on the Pi itself for deployable numbers; x86
  results only rank models against each other.
* ``gpu``  — the server path: ultralytics ``.pt`` checkpoints on CUDA, as
  ``/perceive/detect`` runs them. REFUSES to run with less than ``--min-free``
  MiB of free VRAM so it never destabilizes the live services sharing the card.

Usage::

    python scripts/bench_detect.py cpu  img1.jpg [img2.jpg ...] \
        --model yolo11s.onnx --imgsz 640 [--iters 30] [--conf 0.30]
    python scripts/bench_detect.py gpu  img1.jpg \
        --model yolov8x-worldv2.pt [--vocab scripts/vocab_indoor.txt]

Reports per-image p50/p95 latency and the detected labels, so one run gives
both the latency and the coverage half of a before/after comparison.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _bench(fn, iters: int) -> tuple[float, float]:
    fn()  # warm-up (session/graph build, first CUDA kernels)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = statistics.median(times)
    p95 = times[min(len(times) - 1, int(round(0.95 * len(times))) - 1)]
    return p50, p95


def run_cpu(args) -> int:
    import cv2

    from zero.vision.detector import Detector

    det = Detector(model_path=args.model, confidence=args.conf,
                   imgsz=args.imgsz, device="cpu")
    print(f"[bench cpu] {args.model} imgsz={args.imgsz} conf={args.conf} "
          f"iters={args.iters}")
    for img in args.images:
        frame = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        p50, p95 = _bench(lambda: det.detect(frame), args.iters)
        labels = sorted((d.label, round(d.confidence, 2))
                        for d in det.detect(frame))
        print(f"  {Path(img).name:16s} p50={p50:6.1f}ms p95={p95:6.1f}ms "
              f"n={len(labels)} {labels}")
    return 0


def _free_vram_mib() -> int | None:
    """Free VRAM via nvidia-smi. Deliberately NOT torch.cuda.mem_get_info():
    that call creates a CUDA context, which itself OOMs (and adds pressure)
    exactly when the card is full — the case the guard exists for."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def run_gpu(args) -> int:
    free_mib = _free_vram_mib()
    if free_mib is None:
        print("[bench gpu] nvidia-smi unavailable — no CUDA GPU?"); return 2
    if free_mib < args.min_free:
        print(f"[bench gpu] BLOCKED: only {free_mib} MiB free VRAM "
              f"(< {args.min_free} required). Live services are using the "
              f"card; re-run in a maintenance window.")
        return 3

    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.vocab:
        words = [w.split("#", 1)[0].strip()
                 for w in Path(args.vocab).read_text().splitlines()]
        words = [w for w in words if w]
        t0 = time.perf_counter()
        model.set_classes(words)
        print(f"[bench gpu] set_classes({len(words)}) took "
              f"{time.perf_counter() - t0:.2f}s")
    print(f"[bench gpu] {args.model} conf={args.conf} iters={args.iters}")
    for img in args.images:
        frame = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        p50, p95 = _bench(
            lambda: model.predict(frame, conf=args.conf, verbose=False),
            args.iters)
        r = model.predict(frame, conf=args.conf, verbose=False)[0]
        labels = sorted((r.names[int(b.cls[0])], round(float(b.conf[0]), 2))
                        for b in r.boxes)
        print(f"  {Path(img).name:16s} p50={p50:6.1f}ms p95={p95:6.1f}ms "
              f"n={len(labels)} {labels}")
    print(f"[bench gpu] peak VRAM this process: "
          f"{torch.cuda.max_memory_allocated() // (1024 * 1024)} MiB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["cpu", "gpu"])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--vocab", default=None,
                    help="gpu mode: vocab file for set_classes (YOLO-World)")
    ap.add_argument("--min-free", type=int, default=2500,
                    help="gpu mode: refuse below this many MiB free VRAM")
    args = ap.parse_args()
    return run_cpu(args) if args.mode == "cpu" else run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
