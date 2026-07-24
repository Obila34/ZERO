# Phase 1 — Open-vocabulary detection (killing the 80-object ceiling)

Date: 2026-07-24. Status: **implemented and validated** (GPU-side latency
benchmark pending a maintenance window — see Flags).

## What changed

| Piece | Before | After |
|---|---|---|
| Pi reflex eye (`config.yaml` `vision.detect`) | `yolo11s.onnx`, COCO-80, imgsz 640, conf 0.35 | `yolov8s-worldv2-480.onnx`, **187-word vocabulary** from `scripts/vocab_indoor.txt`, imgsz 480, conf 0.30 |
| GPU heavy eye (`server/vision/perception.py`) | `yolo11x.pt`, COCO-80, hardcoded | `yolov8x-worldv2.pt` + `set_classes(vocab)`, config-driven (`server/vision/config.yaml` `perceive:` block) |
| Vocabulary | fixed at train time | **live**: `GET/POST /perceive/vocab` adds/removes words on the running server (persisted to `server/vision/vocab.runtime.txt`, atomic write, seeded from `scripts/vocab_indoor.txt`) |
| Few-shot teaching (`zero/vision/learned.py`) | unchanged | unchanged — verified label-source agnostic (test below) |

New/changed files: `server/vision/perception.py`, `server/vision/config.yaml`,
`config.yaml`, `yolov8s-worldv2-480.onnx` + `.names.json`,
`scripts/bench_detect.py`, `tests/test_vocab_endpoint.py`, this note.

## Measured numbers (x86 dev box = zerolabs1's CPU, onnxruntime, 30 iters)

`scripts/bench_detect.py cpu <imgs> --model ... --imgsz ... --conf ...`

| Model | Image | p50 | p95 | Detections |
|---|---|---|---|---|
| yolo11s.onnx @640 conf .35 (before) | cats | 44.0 ms | 46.3 ms | 2 cat, couch, remote |
| yolo11s.onnx @640 conf .35 (before) | office | 44.1 ms | 45.9 ms | clock |
| world-s @640 conf .30 | cats | 99.4 ms | 102.3 ms | 2 cat, couch, remote control |
| world-s @640 conf .30 | office | 100.1 ms | 101.1 ms | — |
| **world-s @480 conf .30 (after)** | cats | **48.7 ms** | 51.3 ms | 2 cat, couch, 2 remote control, bed |
| **world-s @480 conf .30 (after)** | office | **48.4 ms** | 51.0 ms | — |

Read: the deployed 480 export matches the old closed-set latency (~49 vs
~44 ms) while finding *more* objects in the cluttered scene, with a 2.3×
larger, runtime-editable vocabulary. The open-vocab mechanism is proven
end-to-end: it detects classes COCO never had (`remote control`, `boot`,
`shoe` in testing), through the unmodified `zero/vision/detector.py` path
(sidecar `.names.json` auto-load).

Server-side `set_classes` on CPU: 3.1 s for 4 words, **11–14 s for 187 words**
(CLIP text encoding dominates; it re-encodes the whole list, not the delta).

## Coverage

COCO-80 → 187 named classes today, extendable in seconds via
`POST /perceive/vocab {"add": ["fire extinguisher"]}` — no retrain, no
restart, persisted across restarts. Known trade-offs, measured:

* YOLO-World confidences run ~0.2 lower than YOLO11 on the same objects —
  hence conf 0.30 (0.35 dropped real objects).
* The 480 export loses very small objects (office wall clock detected by
  closed-set @640, missed by world @480 *and* @640 at conf 0.30 — it is a
  hard image). Reflex tier accepts this; small-object detail is the GPU eye's
  job.
* Detection quality dilutes as the vocabulary grows; keep it in the low
  hundreds (enforced only by convention today).

## Concurrency model (server)

One `threading.RLock` (`_DET_LOCK`) serializes `model.predict` against
`set_classes` — a swap mid-inference would mismatch class ids and names.
Consequence: **detects stall during a vocab update** (11–14 s worst-case on
CPU; expected 1–2 s on GPU, unverified — see Flags). Vocab updates are rare
admin/learning events, so this is accepted for Phase 1. Vocab file writes are
atomic (tmp + `os.replace`); readers see old or new, never partial.

## Flags (honest gaps)

1. **GPU latency/VRAM for yolov8x-worldv2 is unmeasured.** The 16 GB card is
   fully occupied by live services (3 MiB free at bench time);
   `scripts/bench_detect.py gpu` has a VRAM guard and correctly refuses.
   Budget: ≤ 60 ms p50 inference, ≤ 1.5 GB VRAM, `set_classes` ≤ 2 s. Run in
   a maintenance window:
   `python scripts/bench_detect.py gpu img.jpg --model yolov8x-worldv2.pt --vocab scripts/vocab_indoor.txt`
2. **Pi 5 numbers are proxied.** All CPU numbers above are x86; the ratio
   between models transfers, absolute times do not. Re-run the `cpu` bench on
   the Pi before trusting frame-budget math.
3. **Server restart required** to pick up the new detector default; first
   `/perceive/detect` after restart downloads `yolov8x-worldv2.pt` (~150 MB)
   and runs `set_classes` once.
4. **Env change:** ultralytics auto-installed `clip 1.0` + `ftfy` into
   `.venv` (needed by `set_classes`/export). Add to the GPU node's
   requirements when formalizing.
