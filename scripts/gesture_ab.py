#!/usr/bin/env python3
"""Blind A/B: procedural vs neural gesture texture (Phase E, N4 — the gate).

For each trial the SAME synthetic utterance drives the hands twice — once
procedural, once neural — in a random, hidden order. You watch the robot,
type which rendition felt more alive (1/2/tie), and only at the end does
the script unblind and tally. Neural becomes the default ONLY by winning
here; that rule is the plan's, not a preference.

    .venv/bin/python scripts/gesture_ab.py --neural-url http://<gpu>:8200
    .venv/bin/python scripts/gesture_ab.py --neural-url mock   # plumbing only

MOVES THE HANDS. Run on the Pi, eyes on the robot.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from zero.config import load_config  # noqa: E402
from zero.expr.neural import NeuralGestureClient  # noqa: E402
from zero.expr.schedule import HandScheduler  # noqa: E402
from zero.motion.drivers import get_bus, reset_bus  # noqa: E402

SR = 24000

SENTENCES = [
    ("so THAT was the moment everything CHANGED for us", (0.7, 2.1)),
    ("i really THINK we should try the OTHER approach first", (0.9, 2.3)),
    ("it grew SLOWLY at first and then all at ONCE", (0.8, 2.4)),
    ("there was NOTHING left by the time we ARRIVED", (0.6, 2.0)),
]


def make_audio(accents, dur=3.2):
    x = np.zeros(int(dur * SR), dtype=np.float32)

    def place(t, f0, amp, blen):
        n = int(blen * SR)
        tt = np.arange(n) / SR
        f = f0 * (1 + 0.3 * np.sin(np.pi * tt / blen))
        sig = np.hanning(n) * amp * np.sin(2 * np.pi * np.cumsum(f) / SR)
        i = int(t * SR)
        x[i:i + n] += sig[:len(x) - i]

    t = 0.15
    for a in accents:
        while t < a - 0.25:
            place(t, 140, 0.15, 0.14)
            t += 0.33
        place(a - 0.09, 185, 0.5, 0.18)
        t = a + 0.3
    place(t, 140, 0.15, 0.14)
    return x + 0.003 * np.random.randn(len(x)).astype(np.float32)


# Amplitude boost for JUDGING: conversational texture is a 6-8 degree
# finger twitch, nearly invisible across a room (operator, 2026-08-26:
# "nothing happened"). Both renditions get the SAME boost, so the
# comparison stays fair — this scales the motion, not the contest.
_BOOST = {
    "expression.hands.beat.amp": 0.22,
    "expression.hands.engage_closure": 0.12,
    "expression.hands.engage_wrist_deg": 8.0,
    "expression.hands.floor.enabled": True,
    "expression.hands.floor.amp": 0.06,
}


class _Cfg:
    """Real config with per-run neural override + judge-visibility boost."""

    def __init__(self, base, neural_url):
        self._base = base
        self._url = neural_url

    def get(self, k, d=None):
        if k == "expression.hands.neural.enabled":
            return self._url is not None
        if k == "expression.hands.neural.url":
            return self._url or "mock"
        if k in _BOOST:
            return _BOOST[k]
        return self._base.get(k, d)


def play(cfg_base, bus, text, audio, neural_url):
    client = None
    if neural_url is not None:
        client = NeuralGestureClient(_Cfg(cfg_base, neural_url))
    sched = HandScheduler(_Cfg(cfg_base, neural_url), bus, neural=client)
    sched.on_audio(0, text, audio, SR)
    time.sleep(0.6)                     # let inference get ahead
    hop = int(0.05 * SR)
    for i in range(0, len(audio), hop):
        sched.on_playout(0, hop)
        time.sleep(0.05)
    time.sleep(2.0)
    sched.stop()
    if client is not None:
        client.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural-url", required=True,
                    help="sidecar URL, or 'mock' (plumbing test only)")
    ap.add_argument("--trials", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config()
    if str(cfg.get("motion.driver")) != "http":
        print("motion.driver is not http — nothing would move. Abort.")
        return 1
    bus = get_bus(cfg)
    from zero.arms.hands import hand_joint_specs
    from zero.motion.bus import BusJoint

    for name, sp in hand_joint_specs().items():
        if bus.spec(name) is None:
            bus.register(BusJoint(name, min_deg=sp["min"], max_deg=sp["max"],
                                  home_deg=sp["home"], batch=True))

    rng = random.Random()
    results = []
    print(f"{a.trials} trial(s); each plays the SAME utterance twice.\n"
          "Watch the hands. Answer 1, 2, or t (tie).\n")
    for k in range(a.trials):
        text, accents = SENTENCES[k % len(SENTENCES)]
        audio = make_audio(accents)
        order = ["procedural", "neural"]
        rng.shuffle(order)
        for j, mode in enumerate(order, 1):
            input(f"\ntrial {k+1}, rendition {j}: EYES ON THE HANDS, "
                  "then press Enter to play (~6 s)... ")
            print("  playing...")
            play(cfg, bus, text,
                 audio, a.neural_url if mode == "neural" else None)
            print("  done.")
        ans = ""
        while ans not in ("1", "2", "t"):
            ans = input("which felt more alive? [1/2/t] ").strip().lower()
        winner = "tie" if ans == "t" else order[int(ans) - 1]
        results.append(winner)
        print()
    reset_bus()
    n_p = results.count("procedural")
    n_n = results.count("neural")
    n_t = results.count("tie")
    print("=" * 50)
    print(f"UNBLINDED: procedural {n_p} · neural {n_n} · ties {n_t}")
    if n_n > n_p:
        print("Neural WINS this round — it may be promoted "
              "(expression.hands.neural.enabled: true).")
    else:
        print("Neural does not win — it stays off by the plan's own gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
