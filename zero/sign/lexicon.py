"""Motion-sign lexicon — KSL signs as hold-move-hold segments.

A sign is not a pose (that mistake is what made the first fingerspelling
build collapse K into P). Following sign phonology (Liddell & Johnson's
hold-movement model), a lexicon entry is an ordered list of SEGMENTS, each
giving, per hand: a handshape (named, from the manual alphabet, or inline
closures), a wrist orientation, and optionally arm joint targets — plus an
optional head marker (non-manual), a movement time into the segment and a
hold time at it. The renderer turns segments into bus keyframes on the sign
track; arm/head parts of a segment simply wait until those joints are live
(steppers are dark until Phase 5) and are refused honestly until then.

THE LEXICON SHIPS EMPTY, deliberately. Signs teach people; an invented or
misremembered sign teaches them wrong. Entries come from a KSL signer via
`data/sign_lexicon.yaml`, validated here on load — a malformed or
unplayable entry is refused at load time with a reason, never at show time
with a wrong movement.

YAML shape (one sign):

  hello:
    review: pending            # pending | signer-approved
    segments:
      - hands: {right: {shape: B, orient: forward}}
        arm:   {right_up_down_joint: 40, right_elbow_joint: -20}
        move_s: 0.5
        hold_s: 0.3
      - hands: {right: {shape: B, orient: in}}
        move_s: 0.4
        hold_s: 0.2
"""
from __future__ import annotations

from zero.arms.hands import FINGERS, WRIST_ORIENT
from zero.sign.handshapes import HANDSHAPES
from zero.utils.logging import get_logger

log = get_logger("sign.lexicon")

_VALID_HEAD = {"nod", "shake", "tilt", "neutral"}


def _check_hand(spec: dict, side: str) -> str | None:
    """None if valid, else the reason it isn't."""
    shape = spec.get("shape")
    closure = spec.get("closure")
    if shape is None and closure is None:
        return f"{side} hand needs a 'shape' or 'closure'"
    if shape is not None and str(shape).upper() not in HANDSHAPES:
        return f"unknown handshape {shape!r}"
    if closure is not None:
        if not isinstance(closure, dict):
            return "closure must be a mapping of finger -> 0..1"
        for f, v in closure.items():
            if f not in FINGERS:
                return f"unknown finger {f!r}"
            try:
                if not 0.0 <= float(v) <= 1.0:
                    return f"closure for {f} outside 0..1"
            except (TypeError, ValueError):
                return f"closure for {f} is not a number"
    orient = spec.get("orient")
    if orient is not None and orient not in WRIST_ORIENT[side]:
        return f"unknown orientation {orient!r}"
    return None


def validate_sign(name: str, entry: dict) -> str | None:
    """None if the entry is playable as data, else the first problem found.
    (Whether the arm joints it names are LIVE is checked at play time —
    an entry may legitimately be authored ahead of stepper bring-up.)"""
    segments = entry.get("segments")
    if not isinstance(segments, list) or not segments:
        return "no segments"
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return f"segment {i} is not a mapping"
        hands = seg.get("hands") or {}
        if not hands and not seg.get("arm") and not seg.get("head"):
            return f"segment {i} moves nothing"
        for side, spec in hands.items():
            if side not in ("left", "right"):
                return f"segment {i}: unknown side {side!r}"
            why = _check_hand(spec or {}, side)
            if why:
                return f"segment {i}: {why}"
        arm = seg.get("arm") or {}
        if not isinstance(arm, dict):
            return f"segment {i}: arm must map joint -> degrees"
        for j, v in arm.items():
            try:
                float(v)
            except (TypeError, ValueError):
                # refuse at LOAD time — the engine float()s these at show
                # time, and a failure there is a swallowed generic apology
                # (audit sign #6)
                return f"segment {i}: arm {j} value {v!r} is not a number"
        head = seg.get("head")
        if head is not None and head not in _VALID_HEAD:
            return f"segment {i}: unknown head marker {head!r}"
        for key in ("move_s", "hold_s"):
            v = seg.get(key, 0.0)
            try:
                if float(v) < 0.0:
                    return f"segment {i}: {key} is negative"
            except (TypeError, ValueError):
                return f"segment {i}: {key} is not a number"
    return None


def load_lexicon(path) -> dict[str, dict]:
    """Signs from a YAML file, invalid entries dropped WITH a logged reason.
    A missing file is an empty lexicon, not an error."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning("sign lexicon %s unreadable (%s) — empty lexicon", path, e)
        return {}
    if not isinstance(raw, dict):
        log.warning("sign lexicon %s is not a mapping — empty lexicon", path)
        return {}
    out: dict[str, dict] = {}
    for name, entry in raw.items():
        why = validate_sign(str(name), entry or {})
        if why:
            log.warning("sign %r invalid (%s) — dropped", name, why)
            continue
        out[str(name).lower()] = entry
    if out:
        log.info("sign lexicon: %d sign(s) loaded", len(out))
        if any(seg.get("head") for e in out.values()
               for seg in e.get("segments", [])):
            # schema-stable, playback-pending: better one honest line at
            # load than a marker that silently never happens (audit sign #6)
            log.info("sign lexicon: head markers present — validated but "
                     "not yet played (non-manual markers land with the "
                     "motion-sign phase)")
    return out
