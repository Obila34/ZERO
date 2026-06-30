"""Prompt construction for the local VLM (CP5, GPU side).

The VLM is asked to write the robot's *spoken* reply. To keep it from inventing
distances or colors, we hand it the structured ``SceneFact`` list from CP4 as
plain-text context and instruct it to use **only** those facts for object
identity, color, distance, and direction. The frame itself is attached too, so
the model can phrase naturally ("on the table") while staying grounded.

Two pieces live here:

* ``SYSTEM_PROMPT`` — the fixed instruction that sets voice + grounding rules.
* ``build_user_text`` — folds the facts + the user's question into the user turn.

Both VLM backends (Qwen2-VL chat template, Moondream single-prompt) consume the
same strings, so phrasing rules stay in one place. ``model.py`` owns history and
image plumbing.
"""

from __future__ import annotations

from typing import Iterable

try:
    from shared.schemas import SceneFact
except ImportError:  # pragma: no cover - import path fallback (cwd=repo root)
    from server.shared.schemas import SceneFact


SYSTEM_PROMPT = (
    "You are a robot's voice. Answer briefly and naturally for speech. "
    "Use ONLY the provided facts for object identity, color, distance, and "
    "direction; do not invent measurements. Keep replies to 1-2 sentences."
)


def _fact_phrase(fact: SceneFact) -> str:
    """One fact as a compact clause, e.g. 'red cup, 1.2 m away, to your left'."""
    head = f"{fact.color} {fact.label}" if fact.color else fact.label
    parts = [head]
    if fact.distance_m is not None:
        parts.append(f"{fact.distance_m:.1f} m away")
    if fact.bearing:
        # "center" reads awkwardly as "to your center"; speak it as straight ahead.
        if fact.bearing == "center":
            parts.append("directly ahead")
        else:
            parts.append(f"to your {fact.bearing}")
    return ", ".join(parts)


def format_facts(facts: Iterable[SceneFact]) -> str:
    """Render the scene facts as a bulleted block for the prompt.

    Empty scene yields an explicit marker so the model knows there is nothing to
    describe rather than hallucinating objects.
    """
    facts = list(facts)
    if not facts:
        return "(no objects detected in view)"
    return "\n".join(f"- {_fact_phrase(f)}" for f in facts)


def build_user_text(facts: Iterable[SceneFact], question: str) -> str:
    """Compose the user turn: the grounded facts followed by the question."""
    question = (question or "").strip() or "What are you looking at?"
    return (
        "Facts about what you currently see:\n"
        f"{format_facts(facts)}\n\n"
        f"Question: {question}"
    )


if __name__ == "__main__":
    demo = [
        SceneFact(label="cup", color="red", distance_m=1.2, bearing="left"),
        SceneFact(label="table", color=None, distance_m=1.1, bearing="center"),
        SceneFact(label="person", color=None, distance_m=2.4, bearing="right"),
    ]
    print(SYSTEM_PROMPT)
    print("---")
    print(build_user_text(demo, "what are you looking at?"))
    print("---")
    print(build_user_text([], "what do you see?"))
    print("prompt OK")
