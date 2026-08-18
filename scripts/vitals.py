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


def _mem() -> str:
    try:
        vals = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            vals[k] = int(v.split()[0]) // 1024
        return f"mem_avail={vals.get('MemAvailable', 0)}M"
    except Exception:
        return "mem_avail=?"


def _usb() -> str:
    """Which USB devices are present. The BRIO dropping off the bus is the
    single most repeated event in this robot's logs, so its presence is worth
    a column: if it vanishes just before a reset, that is the answer."""
    import glob
    names = []
    for d in sorted(glob.glob("/sys/bus/usb/devices/1-1.*")):
        try:
            with open(os.path.join(d, "product")) as f:
                names.append(f.read().strip().split()[-1][:8])
        except Exception:
            pass
    return "usb=" + (",".join(names) if names else "NONE")


def _top() -> str:
    try:
        out = subprocess.run(["ps", "-eo", "pcpu,comm", "--sort=-pcpu"],
                             capture_output=True, text=True, timeout=2.0)
        line = out.stdout.strip().splitlines()[1].split()
        return f"top={line[1]}:{line[0]}%"
    except Exception:
        return "top=?"


_dmesg_seen: set = set()


def _kernel_new() -> str:
    """NEW kernel messages since the last sample. dmesg is readable without
    root here and is wiped by a reboot, so copying it to disk as we go is the
    only way its final words survive a hard reset — the system journal on this
    Pi lives in RAM and dies with it."""
    try:
        out = subprocess.run(["dmesg"], capture_output=True, text=True,
                             timeout=3.0).stdout.splitlines()
    except Exception:
        return ""
    fresh = []
    for line in out[-80:]:
        key = line[:160]
        if key not in _dmesg_seen:
            _dmesg_seen.add(key)
            low = line.lower()
            if any(w in low for w in ("error", "fail", "reset", "disconnect",
                                      "usb", "under", "throttl", "oom",
                                      "hung", "panic", "warn", "xhci", "mmc")):
                fresh.append(line.strip())
    return "\n".join(f"    KERN {x}" for x in fresh[-6:])


def sample() -> str:
    load = os.getloadavg()
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read().strip()) / 1000.0
    except Exception:
        temp = float("nan")
    thr = _decode(_vcgencmd("get_throttled"))
    volt = _vcgencmd("measure_volts").replace("volt=", "")
    line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"load={load[0]:.2f},{load[1]:.2f} temp={temp:.1f}C "
            f"core={volt} power={thr} {_mem()} {_usb()} {_top()}")
    kern = _kernel_new()
    return line + ("\n" + kern if kern else "")


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
                print(line.splitlines()[0][:150], end="\r")
                time.sleep(every)
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
