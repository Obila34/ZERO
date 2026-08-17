#!/usr/bin/env python3
"""Vitals logger — survive-the-reboot diagnostics, no root required.

The head Pi keeps its system journal in RAM, so when it went down unexplained
on 2026-08-17 the evidence died with it. This appends a line every few seconds
to vitals.log on disk, so after any crash the LAST line says what the machine
was doing the moment before: CPU load, temperature, and the SoC's own
undervoltage/throttle flags.

Run it in a second terminal alongside ZERO during a test:

    .venv/bin/python scripts/vitals.py

Reading the throttle word (from `vcgencmd get_throttled`), bit set = now,
bit set >>16 = happened since boot:
    0x1  under-voltage        0x2  arm frequency capped
    0x4  currently throttled  0x8  soft temperature limit

An under-voltage bit next to the last line before a reboot is a power problem,
not a software one — that is the whole point of this file.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "vitals.log")
FLAGS = ((0x1, "UNDER-VOLTAGE"), (0x2, "freq-capped"),
         (0x4, "throttled"), (0x8, "soft-temp-limit"))


def _vcgencmd(arg: str) -> str:
    try:
        out = subprocess.run(["vcgencmd", arg], capture_output=True,
                             text=True, timeout=2.0)
        return out.stdout.strip()
    except Exception:
        return ""


def _decode(word: str) -> str:
    try:
        v = int(word.split("=", 1)[1], 16)
    except (IndexError, ValueError):
        return word or "?"
    if v == 0:
        return "ok"
    now = [n for b, n in FLAGS if v & b]
    since = [n for b, n in FLAGS if v & (b << 16)]
    parts = []
    if now:
        parts.append("NOW:" + ",".join(now))
    if since:
        parts.append("since-boot:" + ",".join(since))
    return " ".join(parts) or "ok"


def sample() -> str:
    load = os.getloadavg()
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read().strip()) / 1000.0
    except Exception:
        temp = float("nan")
    thr = _decode(_vcgencmd("get_throttled"))
    volt = _vcgencmd("measure_volts").replace("volt=", "")
    return (f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"load={load[0]:.2f},{load[1]:.2f} temp={temp:.1f}C "
            f"core={volt} power={thr}")


def main() -> int:
    every = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    print(f"vitals -> {os.path.abspath(LOG)} every {every:.0f}s (Ctrl-C to stop)")
    with open(LOG, "a") as f:
        f.write(f"# session start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.flush()
        try:
            while True:
                line = sample()
                f.write(line + "\n")
                f.flush()          # flush every line: a crash must not lose it
                os.fsync(f.fileno())
                print(line, end="\r")
                time.sleep(every)
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
