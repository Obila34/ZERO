#!/usr/bin/env python3
"""Validate (and optionally dry-run) the KSL sign lexicon — the signer's tool.

Usage:
    python scripts/sign_lexicon_check.py [path]           # validate all entries
    python scripts/sign_lexicon_check.py [path] --play X  # dry-run sign X:
        prints the joint keyframes the sign would produce, second by second,
        against a NullTransport — NOTHING MOVES. Watch the numbers, then
        review on the robot with a signer present.

Exit code 0 = every entry valid; 1 = problems found (listed per sign).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/sign_lexicon.yaml")
    ap.add_argument("--play", metavar="SIGN",
                    help="dry-run one sign and print its keyframes")
    args = ap.parse_args()

    import yaml

    from zero.sign.lexicon import validate_sign

    try:
        raw = yaml.safe_load(open(args.path, encoding="utf-8")) or {}
    except FileNotFoundError:
        print(f"{args.path}: not found (an empty lexicon is valid)")
        return 0
    if not isinstance(raw, dict):
        print(f"{args.path}: top level must be a mapping of sign -> entry")
        return 1

    bad = 0
    for name, entry in sorted(raw.items()):
        why = validate_sign(str(name), entry or {})
        review = (entry or {}).get("review", "pending")
        if why:
            bad += 1
            print(f"  INVALID  {name}: {why}")
        else:
            nseg = len(entry["segments"])
            print(f"  ok       {name}: {nseg} segment(s), review={review}")
    print(f"{len(raw)} sign(s), {bad} invalid")

    if args.play and not bad:
        from zero.arms.hands import hand_joint_specs
        from zero.motion.bus import BusJoint, MotionBus
        from zero.motion.transport import NullTransport
        from zero.sign.engine import SignEngine

        class Cfg(dict):
            def get(self, k, d=None):
                return super().get(k, d)

        t = NullTransport()
        bus = MotionBus(t, rate_hz=100.0)
        for jname, s in hand_joint_specs().items():
            bus.register(BusJoint(jname, min_deg=s["min"], max_deg=s["max"],
                                  home_deg=s["home"], batch=True))
        eng = SignEngine(Cfg({"sign.lexicon_path": args.path}), bus)
        spoken = eng.sign(args.play)
        if spoken is None:
            print(f"--play: {args.play!r} not in the lexicon")
            return 1
        print(f"\n{spoken}  (dry run — NullTransport, nothing moves)")
        last: dict = {}
        t0 = time.monotonic()
        while bus.owner("left_indexp1_joint") == "sign" or \
                bus.owner("right_indexp1_joint") == "sign":
            snap = dict(t.posted)
            changed = {k: v for k, v in snap.items()
                       if abs(v - last.get(k, 1e9)) > 2.0}
            if changed:
                print(f"  t={time.monotonic()-t0:5.2f}s  " + "  ".join(
                    f"{k}={v:.0f}" for k, v in sorted(changed.items())))
                last.update(changed)
            time.sleep(0.05)
        bus.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
