"""End-to-end world-state latency: event-happens -> world-state-reflects-it.

Measures, with real components (no models mocked except where stated):

* Tier 0  — motion: MotionDetector.update + WorldState.update_motion per frame.
* Tier 1  — pipeline overhead: tracker.update + WorldState.update_objects for
  a 10-object scene (detector inference measured separately by
  scripts/bench_detect.py — add the two for the full event->state number).
* Tier 2  — narrator pipeline overhead with an instant fake backend (the
  model-free floor), and optionally ONE real /analyze call against the live
  vision server (--analyze) for the true model latency.
* Readers — WorldState.snapshot() and describe(): the "instant answer" cost
  paid by the conversation hot path.

Run:  python scripts/bench_world.py [--analyze]
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from zero.vision.motion import MotionDetector          # noqa: E402
from zero.vision.tracker import IouTracker             # noqa: E402
from zero.world.state import WorldEvent, WorldObject, WorldState  # noqa: E402


def _stats(times_ms):
    times_ms = sorted(times_ms)
    p50 = statistics.median(times_ms)
    p95 = times_ms[min(len(times_ms) - 1, int(round(0.95 * len(times_ms))) - 1)]
    return p50, p95


def bench_tier0(iters=200):
    det = MotionDetector()
    world = WorldState()
    frames = []
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    for i in range(8):     # cycle of slightly-shifted frames = realistic diffs
        frames.append(np.roll(base, i * 3, axis=1))
    det.update(frames[0])
    times = []
    for i in range(iters):
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        level, active = det.update(f)
        world.update_motion(level, active)
        times.append((time.perf_counter() - t0) * 1000)
    p50, p95 = _stats(times)
    print(f"Tier 0  motion+publish/frame       p50={p50:7.3f}ms p95={p95:7.3f}ms")


class _Det:
    def __init__(self, label, bbox, confidence=0.9):
        self.label = label
        self.bbox = bbox
        self.confidence = confidence


def bench_tier1(iters=200):
    tracker = IouTracker()
    world = WorldState()
    times = []
    t = 1000.0
    for i in range(iters):
        # 10 objects; one of them wanders so tracks stay busy.
        dets = [_Det(f"obj{k}", (k * 60.0 + (i if k == 0 else 0), 50.0,
                                 50.0, 50.0)) for k in range(10)]
        t += 1 / 15
        t0 = time.perf_counter()
        events = tracker.update(dets, t)
        objects = [WorldObject(track_id=tr.tid, label=tr.label,
                               bbox=tuple(tr.bbox), confidence=tr.conf,
                               first_seen=tr.first_ts, last_seen=tr.last_ts)
                   for tr in tracker.tracks()]
        kinds = {"new": "appeared", "gone": "left", "moved": "moved"}
        wevents = [WorldEvent(kind=kinds[k], label=lbl, ts=t)
                   for k, lbl in events if k in kinds]
        world.update_objects(objects, wevents, ts=t)
        times.append((time.perf_counter() - t0) * 1000)
    p50, p95 = _stats(times)
    print(f"Tier 1  track+publish (10 objects) p50={p50:7.3f}ms p95={p95:7.3f}ms"
          f"   (+ detector inference: see bench_detect.py)")


def bench_tier2_pipeline():
    from zero.world.narrator import Narrator

    world = WorldState()
    world.update_motion(0.5, True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class InstantBackend:
        def narrate(self, f, hint):
            return "bench narration"

    n = Narrator(world, lambda: frame, InstantBackend(), interval_s=0.01)
    v0 = world.version
    n.start()
    t0 = time.perf_counter()
    while world.snapshot().narration == "" and time.perf_counter() - t0 < 5:
        time.sleep(0.001)
    latency_ms = (time.perf_counter() - t0) * 1000
    n.stop()
    assert world.version > v0
    print(f"Tier 2  narrator pipeline floor    first-write={latency_ms:7.1f}ms"
          f"   (interval-dominated; model latency is the real cost)")


def bench_tier2_analyze():
    """ONE real /analyze call against the live vision server, if reachable."""
    try:
        from zero.vision.gpu_client import VisionClient

        client = VisionClient(url="http://127.0.0.1:8000",
                              analyze_timeout_s=60.0)
        client.check_health()
    except Exception as e:
        print(f"Tier 2  real /analyze              BLOCKED: {e}")
        return
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 200:400] = (180, 60, 60)
    try:
        t0 = time.perf_counter()
        reply = client.analyze(frame, [], question="Describe this in one sentence.")
        dt = time.perf_counter() - t0
        print(f"Tier 2  real /analyze (Qwen2-VL)   {dt*1000:7.1f}ms  reply={reply[:60]!r}")
    except Exception as e:
        print(f"Tier 2  real /analyze              FAILED (flagged, not faked): {e}")


def bench_readers(iters=100_000):
    world = WorldState()
    world.update_objects([WorldObject(track_id=i, label=f"obj{i}",
                                      bbox=(0, 0, 1, 1), confidence=0.9,
                                      first_seen=time.time(),
                                      last_seen=time.time())
                          for i in range(8)],
                         [WorldEvent("appeared", "mug", time.time())])
    world.update_narration("bench narration")
    t0 = time.perf_counter()
    for _ in range(iters):
        world.snapshot()
    snap_us = (time.perf_counter() - t0) / iters * 1e6
    t0 = time.perf_counter()
    for _ in range(iters // 10):
        world.describe()
    desc_us = (time.perf_counter() - t0) / (iters // 10) * 1e6
    print(f"Reader  snapshot()                 {snap_us:7.2f}µs/call")
    print(f"Reader  describe()                 {desc_us:7.2f}µs/call")


if __name__ == "__main__":
    print("world-state latency benchmark (event-happens -> state-reflects-it)")
    bench_tier0()
    bench_tier1()
    bench_tier2_pipeline()
    if "--analyze" in sys.argv:
        bench_tier2_analyze()
    bench_readers()
