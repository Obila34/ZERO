"""Surprise gating — prediction error decides what is worth attention.

The predictive-coding idea, sized honestly for ZERO: the world state keeps an
ONLINE model of its own statistics — how often each label is present, how
often each (label, event-kind) happens — and scores every new event by how
much it violates those expectations:

    surprise(event) = -log2( P(event) )        # rarity in bits

An event's probability is its smoothed historical share of all events for
that label. First-ever events score highest; daily routine decays toward
zero bits. The stats persist across runs (JSON sidecar) so ZERO's sense of
"normal" accumulates.

Two consumers, per the complementary-learning-systems design:
  * MEMORY — events with surprise >= ``remember_bits`` become 'scene'
    episodes (with the surprise tag) for nightly consolidation. Routine is
    not remembered; the unexpected is.
  * ATTENTION — events with surprise >= ``narrate_bits`` poke the Tier 2
    narrator: wake the expensive model for the unexpected, let it sleep
    through routine.

``SurpriseGate`` runs as a thread on ``world.wait_for_change()`` — it wakes
only when the world actually changes, costs ~µs per event, and never touches
the perception or conversation hot paths.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

from zero.utils.logging import get_logger

log = get_logger("world.surprise")


class SurprisePredictor:
    """Online event statistics -> surprise in bits. Laplace-smoothed counts
    of (label, kind) over total observed events; persisted as JSON."""

    def __init__(self, path: str | None = None, smoothing: float = 1.0):
        self._path = Path(path) if path else None
        self._smoothing = float(smoothing)
        self._counts: dict[str, float] = {}   # "label|kind" -> count
        self._total = 0.0
        self._dirty = 0
        if self._path is not None and self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._counts = {str(k): float(v)
                                for k, v in data.get("counts", {}).items()}
                self._total = float(data.get("total", 0.0))
                log.info("surprise stats loaded (%d event types, %.0f seen)",
                         len(self._counts), self._total)
            except (ValueError, OSError) as e:
                log.warning("bad surprise stats %s — starting fresh: %s",
                            self._path, e)

    @staticmethod
    def _key(label: str, kind: str) -> str:
        return f"{label.lower()}|{kind}"

    def surprise(self, label: str, kind: str) -> float:
        """Bits of surprise for this event, WITHOUT recording it."""
        k = self._key(label, kind)
        # Laplace smoothing with reserved unseen mass (+2 virtual types): a
        # never-seen event is ~1 bit on a cold model and grows toward
        # -log2(1/total) as history accumulates; a routine event decays
        # toward 0 bits. (types+1 alone gives p=1 -> 0 bits for the very
        # first event — backwards.)
        p = ((self._counts.get(k, 0.0) + self._smoothing)
             / (self._total + self._smoothing * (len(self._counts) + 2)))
        return -math.log2(max(1e-12, min(1.0, p)))

    def observe(self, label: str, kind: str) -> float:
        """Score the event, then fold it into the model (so the same thing
        happening daily becomes progressively less surprising)."""
        bits = self.surprise(label, kind)
        self._counts[self._key(label, kind)] = \
            self._counts.get(self._key(label, kind), 0.0) + 1.0
        self._total += 1.0
        self._dirty += 1
        if self._dirty >= 25:                # amortized persistence
            self.save()
        return bits

    def save(self) -> None:
        if self._path is None:
            return
        try:
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(
                {"total": self._total, "counts": self._counts}),
                encoding="utf-8")
            tmp.replace(self._path)
            self._dirty = 0
        except OSError as e:
            log.warning("surprise stats save failed: %s", e)


class SurpriseGate:
    """Thread: score every new world event; route the surprising ones."""

    def __init__(self, world, predictor: SurprisePredictor, *,
                 episodes=None, narrator=None,
                 remember_bits: float = 4.0, narrate_bits: float = 6.0):
        self._world = world
        self._pred = predictor
        self._episodes = episodes
        self._narrator = narrator
        self.remember_bits = float(remember_bits)
        self.narrate_bits = float(narrate_bits)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_through = 0.0             # ts high-water mark for events
        self.scored = 0                      # observability
        self.remembered = 0
        self.narrator_pokes = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="SurpriseGate",
                                        daemon=True)
        self._thread.start()
        log.info("surprise gate running (remember>=%.1f bits, "
                 "narrate>=%.1f bits)", self.remember_bits, self.narrate_bits)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # wait_for_change has a timeout, so the thread notices the stop.
            self._thread.join(timeout=3.0)
            self._thread = None
        self._pred.save()

    def _loop(self) -> None:
        version = self._world.version
        while not self._stop.is_set():
            snap = self._world.wait_for_change(version, timeout=1.0)
            version = snap.version
            for ev in snap.events:
                if ev.ts <= self._seen_through:
                    continue                 # already scored on a prior wake
                self._seen_through = ev.ts
                self._score(ev)

    def _score(self, ev) -> None:
        bits = self._pred.observe(ev.label, ev.kind)
        self.scored += 1
        if bits >= self.narrate_bits and self._narrator is not None:
            self.narrator_pokes += 1
            try:
                self._narrator.poke()
            except Exception as e:
                log.debug("narrator poke failed: %s", e)
        if bits >= self.remember_bits and self._episodes is not None:
            self.remembered += 1
            try:
                self._episodes.add(
                    "scene", {"event": ev.kind, "label": ev.label},
                    surprise=round(bits, 2), ts=ev.ts)
            except Exception as e:
                log.debug("scene episode write failed: %s", e)
        log.debug("event %s/%s -> %.1f bits", ev.label, ev.kind, bits)
