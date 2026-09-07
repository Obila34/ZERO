#!/usr/bin/env python3
"""Supervised preview of dictionary signs — the gate before enabling.

Plays chosen signs from the compiled motion dictionary through the REAL
engine and bus, one at a time, with your confirmation between each.
Recorded human arm motion drives the steppers (capped), so this is a
watched bring-up, exactly like the arm carrier's stream probe was.

    .venv/bin/python scripts/sign_dict_probe.py hello please yes no
    .venv/bin/python scripts/sign_dict_probe.py --list water

MOVES HANDS AND ARMS. zero.service stopped, eyes on the robot,
e-stop in reach. If what you see looks right, enable it:
config.yaml -> sign.dictionary.enabled: true
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero.config import load_config  # noqa: E402
from zero.motion.drivers import get_bus, reset_bus  # noqa: E402


class _Cfg:
    """Real config with the dictionary force-enabled for this run only."""

    def __init__(self, base):
        self._b = base

    def get(self, k, d=None):
        if k == "sign.dictionary.enabled":
            return True
        return self._b.get(k, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("glosses", nargs="*", default=[])
    ap.add_argument("--list", metavar="PREFIX",
                    help="list dictionary signs starting with PREFIX")
    a = ap.parse_args()

    cfg = _Cfg(load_config())
    if a.list is not None:
        from zero.sign.dictionary import load_dictionary
        d = load_dictionary(cfg)
        if d is None:
            print("dictionary asset missing")
            return 1
        hits = [g for g in d.glosses() if g.startswith(a.list.lower())]
        print("\n".join(hits) or f"nothing starts with {a.list!r}")
        return 0
    if not a.glosses:
        print("give one or more glosses (or --list PREFIX)")
        return 1
    if str(cfg.get("motion.driver")) != "http":
        print("motion.driver is not http — nothing would move. Abort.")
        return 1

    bus = get_bus(cfg)
    from zero.arms.driver import load_joints, make_arm_driver
    make_arm_driver(cfg, load_joints(cfg))
    # prime a stepper write so the gateway offsets load (muted otherwise)
    prime = {}
    for n in ("right_elbow_joint", "left_elbow_joint"):
        sp = bus.spec(n)
        if sp is not None:
            prime[n] = sp.home_deg
    if prime:
        bus.write("gesture", prime)
        deadline = time.monotonic() + 15.0
        while getattr(bus, "_offsets", None) is None \
                and time.monotonic() < deadline:
            time.sleep(0.25)
        bus.release("gesture")
        if getattr(bus, "_offsets", None) is None:
            print("gateway offsets never arrived — steppers would stay "
                  "mute. Abort.")
            return 1

    from zero.sign.engine import SignEngine
    eng = SignEngine(cfg, bus)
    for gloss in a.glosses:
        input(f"\nEYES ON THE ROBOT — Enter to play {gloss!r} "
              "(Ctrl-C aborts)... ")
        said = eng.sign(gloss)
        if said is None:
            print(f"  {gloss!r} is not in the dictionary — skipped")
            continue
        print(f"  {said}")
        time.sleep(4.0)          # sign + settle + lower
        ans = input("  looked like real signing? [y/n] ").strip().lower()
        print(f"  recorded: {'PASS' if ans == 'y' else 'FAIL'}")
    eng.stop()
    reset_bus()
    print("\nprobe done. To go live: sign.dictionary.enabled: true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
