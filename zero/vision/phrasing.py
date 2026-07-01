"""Turn detections / scene facts into compact text for the LLM prompt.

Two flavours:

* ``local_summary`` — a quick, GPU-free line from the always-on detections
  ("in view: a person, a red cup, a laptop"). Cheap enough to attach to *every*
  turn so ZERO always has ambient awareness of the room.
* ``facts_line`` — the grounded line built from GPU ``SceneFact``s, with distance
  and bearing ("a red cup ~1.2 m to your left; a person ~2.0 m ahead"). Attached
  only on visual turns, where the small extra latency of GPU depth is worth it.

The text is injected into the user turn, never the cached system prefix, so the
prompt cache stays warm.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from zero.vision.schemas import Detection, SceneFact

_ARTICLE_PLURAL = {  # irregulars; everything else just gets a trailing "s"
    "person": "people",
}


def _plural(label: str, n: int) -> str:
    if n == 1:
        return label
    return _ARTICLE_PLURAL.get(label, label + "s")


def detector_hint(detections: Sequence[Detection], max_items: int = 10) -> str:
    """Counts + labels only ("3 people, a cup, a laptop"), or '' if empty.

    This is the *hint* handed to the vision LLM alongside the actual frames — a
    nudge so it doesn't undercount, never spoken verbatim. Deliberately carries
    NO distances, bearings or pixel-sampled colors: those read as clinical, and
    the model sees the real colors/positions in the image itself.
    """
    if not detections:
        return ""
    counts: Counter = Counter(d.label for d in detections)
    parts: list[str] = []
    for label, n in counts.most_common(max_items):
        noun = _plural(label, n)
        qty = "a" if n == 1 else str(n)
        parts.append(f"{qty} {noun}")
    return ", ".join(parts)


def local_summary(detections: Sequence[Detection], max_items: int = 6) -> str:
    """Compact GPU-free description of the current scene, or '' if empty."""
    if not detections:
        return ""
    # Count by (color, label) so "two red cups" reads naturally.
    counts: Counter = Counter()
    for d in detections:
        counts[(d.color, d.label)] += 1
    parts: list[str] = []
    for (color, label), n in counts.most_common(max_items):
        noun = _plural(label, n)
        qty = "a" if n == 1 else str(n)
        if color:
            parts.append(f"{qty} {color} {noun}")
        else:
            parts.append(f"{qty} {noun}")
    return ", ".join(parts)


def _fact_phrase(fact: SceneFact) -> str:
    head = f"{fact.color} {fact.label}" if fact.color else fact.label
    parts = [f"a {head}"]
    if fact.distance_m is not None:
        parts.append(f"~{fact.distance_m:.1f} m away")
    if fact.bearing:
        if fact.bearing == "center":
            parts.append("directly ahead")
        else:
            parts.append(f"to your {fact.bearing}")
    return ", ".join(parts)


def facts_line(facts: Iterable[SceneFact], max_items: int = 6) -> str:
    """Grounded one-liner from GPU scene facts, or '' if empty."""
    facts = list(facts)[:max_items]
    if not facts:
        return ""
    return "; ".join(_fact_phrase(f) for f in facts)
