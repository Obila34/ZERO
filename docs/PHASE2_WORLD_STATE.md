# Phase 2 — Three perception tiers, one live world state

Date: 2026-07-24. Status: **implemented and validated**; Tier 2 model
inference is VRAM-blocked on current hardware (see Flags).

## Architecture

```
camera frame (every frame)
   │
   ├─ Tier 0  MotionDetector (zero/vision/motion.py)      ~0.4 ms/frame
   │            └─> world.update_motion()   + gates Tier 1 (DetectionGate)
   │
   ├─ Tier 1  open-vocab Detector + IouTracker            detector ~49 ms (x86)
   │            └─> world.update_objects()  (persistent track ids, events)
   │                                        publish overhead ~0.06 ms
   └─ Tier 2  Narrator thread (zero/world/narrator.py)    model-latency bound
                └─> world.update_narration()

                    WorldState (zero/world/state.py)
                      │  versioned immutable snapshots
                      ▼
   readers: LLM context (Eyes.local_context/visual_context), proactive,
   memory (Phase 3) — snapshot() 0.04 µs, describe() 2.6 µs, lock-free
```

## Concurrency/consistency model

The world is ONE immutable `WorldSnapshot`. Writers (three tiers, three
threads, three different frequencies) serialize on a single lock, build a new
snapshot, and atomically swap the reference; every publish bumps a monotonic
`version` and notifies a Condition. Readers do a bare reference read —
lock-free, wait-free, always internally consistent, and a held snapshot never
changes underneath the reader. Event-driven readers use
`wait_for_change(after_version, timeout)` instead of polling. This is what
makes "the answer already exists" true under load: a reader can NEVER see a
half-written world, and never waits.

Sparse firing is explicit: Tier 0 runs always (cheap); Tier 1 runs at full
cadence only during/shortly after motion, else a 2 s keepalive
(`world.gate.*` in config.yaml); Tier 2 self-skips when nothing moved and the
scene graph is unchanged since its last narration.

## Measured (scripts/bench_world.py, x86 dev box; budgets in parentheses)

| Path | Measured | Budget |
|---|---|---|
| Tier 0 motion + publish, per frame | 0.392 ms p50 / 0.449 ms p95 | ≤ 5 ms ✓ |
| Tier 1 track + publish (10 objects) | 0.058 ms p50 | ≤ 2 ms overhead ✓ |
| Tier 1 full event→state | detector (~49 ms x86, Phase 1) + 0.06 ms | — |
| Tier 2 pipeline floor (fake backend) | ~10 ms first-write | — |
| Reader `snapshot()` | 0.04 µs | ≤ 50 µs ✓ |
| Reader `describe()` | 2.62 µs | ≤ 1 ms ✓ |

## What changed

* `zero/vision/motion.py` — new: `MotionDetector` (Tier 0), `DetectionGate`.
* `zero/world/state.py` — new: `WorldState`, `WorldSnapshot`, `WorldObject`,
  `WorldEvent`, `describe()`.
* `zero/world/narrator.py` — new: `Narrator` thread; backends
  `AnalyzeBackend` (existing `/analyze` Qwen2-VL) and `OpenAIVisionBackend`
  (Cosmos 3 Nano Reasoner via vLLM — `server/cosmos/run_cosmos_narrator.sh`).
* `zero/vision/tracker.py` — `tracks()` accessor + per-track confidence.
* `zero/vision/eyes.py` — loop restructured: Tier 0 every frame, motion-gated
  Tier 1, world publishing; narrator lifecycle; fresh narration appended to
  `local_context()`/`visual_context()` so the conversational brain reads the
  world with zero added latency. Behavior unchanged when `world` is None.
* `zero/factory.py`, `config.yaml` (`world:` section) — wiring + knobs.
* `tests/test_world.py` (13 tests), `scripts/bench_world.py`.

Readers wired now: the LLM context path. Proactive/memory get their read
surface (`eyes.world`, `wait_for_change`, structured `WorldEvent`s) — consumed
in Phase 3 (surprise gating decides what's remembered / spoken).

## Flags (honest gaps)

1. **Tier 2 model inference is VRAM-blocked today.** One real `/analyze` call
   was attempted against the live vision server: HTTP 500 (VLM cannot load —
   card at 15.48/16 GB). The narrator therefore ships `enabled: false`.
   Pipeline is fully tested with a fake backend; enabling is a config flip
   once VRAM exists. Cosmos serve script budgets are ESTIMATES until measured.
2. **Gated-frame semantics**: while the gate is closed, the last detections
   are republished with each fresh frame (the scene didn't change — that's
   why the gate closed). Slow drift during total stillness is bounded by the
   2 s keepalive detection.
3. **Pi cadence numbers are x86-proxied** (same caveat as Phase 1); re-run
   `bench_world.py` on the Pi. Tier 0's 0.4 ms x86 ≈ low single-digit ms on
   Pi 5 — comfortably inside budget.
4. Second GPU / edge box is a **drop-in**: Tier 2 is behind a URL
   (`world.narrator.url`); nothing else references the GPU.
