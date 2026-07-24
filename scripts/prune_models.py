"""Compress the small nets ZERO owns — measured, never estimated.

Two jobs, mirroring childhood synaptic pruning on deployable terms:

* ``quantize`` — INT8 dynamic quantization of an ONNX model (the compression
  that actually ships on a Pi: smaller file, integer matmuls). Reports size,
  CPU latency, and — for detectors — label agreement vs the FP32 original on
  supplied images, so accuracy drift is a printed number, not a hope.
* ``sparsity`` — magnitude analysis of the weights: what fraction is close
  enough to zero to prune at given thresholds, and the natural sparsity the
  model already has. Informational: actual magnitude pruning without a
  recovery fine-tune costs accuracy, so we PRINT what pruning would remove
  and only ship quantization.

Usage:
    python scripts/prune_models.py sparsity  model.onnx
    python scripts/prune_models.py quantize  model.onnx img1.jpg img2.jpg \
        [--imgsz 480] [--conf 0.30]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def cmd_sparsity(args) -> int:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(args.model)
    weights = [numpy_helper.to_array(t) for t in model.graph.initializer
               if t.data_type == onnx.TensorProto.FLOAT]
    if not weights:
        print("[sparsity] no float weights found (already quantized?)")
        return 1
    flat = np.concatenate([w.reshape(-1) for w in weights])
    total = flat.size
    print(f"[sparsity] {Path(args.model).name}: {total/1e6:.1f}M float params")
    exact_zero = float(np.count_nonzero(flat == 0.0)) / total
    print(f"  natural exact-zero sparsity: {exact_zero*100:5.2f}%")
    mags = np.abs(flat)
    for frac in (0.2, 0.4, 0.6):
        thresh = np.quantile(mags, frac)
        print(f"  magnitude-prune {int(frac*100)}%: removes |w| <= "
              f"{thresh:.2e} (mean |w| = {mags.mean():.2e})")
    print("  NOTE: pruning without recovery fine-tuning costs accuracy — "
          "informational only; ship quantization instead.")
    return 0


def cmd_quantize(args) -> int:
    import cv2
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from zero.vision.detector import Detector

    src = Path(args.model)
    dst = src.with_name(src.stem + "-int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    # The detector needs the class-name sidecar next to the new file too.
    names = src.with_suffix("").with_suffix(".names.json")
    if names.exists():
        dst.with_suffix("").with_suffix(".names.json").write_bytes(
            names.read_bytes())
    s0, s1 = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
    print(f"[quantize] {src.name} {s0:.1f}MB -> {dst.name} {s1:.1f}MB "
          f"({(1 - s1 / s0) * 100:.0f}% smaller)")

    if not args.images:
        print("[quantize] no images given — skipping latency/agreement check")
        return 0
    fp32 = Detector(model_path=str(src), confidence=args.conf, imgsz=args.imgsz)
    int8 = Detector(model_path=str(dst), confidence=args.conf, imgsz=args.imgsz)
    agree_n = agree_hit = 0
    for img in args.images:
        frame = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        rows = {}
        for name, det in (("fp32", fp32), ("int8", int8)):
            det.detect(frame)                       # warm
            t0 = time.perf_counter()
            for _ in range(args.iters):
                out = det.detect(frame)
            ms = (time.perf_counter() - t0) / args.iters * 1000
            rows[name] = (ms, sorted(d.label for d in out))
        (ms32, l32), (ms8, l8) = rows["fp32"], rows["int8"]
        agree_n += 1
        agree_hit += int(l32 == l8)
        print(f"  {Path(img).name:14s} fp32 {ms32:6.1f}ms {l32}")
        print(f"  {'':14s} int8 {ms8:6.1f}ms {l8}"
              f"   {'== MATCH' if l32 == l8 else '!= DRIFT'}")
    print(f"[quantize] label agreement: {agree_hit}/{agree_n} images")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["quantize", "sparsity"])
    ap.add_argument("model")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()
    return cmd_sparsity(args) if args.cmd == "sparsity" else cmd_quantize(args)


if __name__ == "__main__":
    raise SystemExit(main())
