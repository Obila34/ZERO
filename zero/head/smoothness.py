"""Objective smoothness metrics for a head-command stream.

'It looks smooth' is not a test. This turns a recorded angle-vs-time series
(the commanded pose per tick, or — later — a fiducial tracked from a 120 fps
phone video, or background optical-flow from the BRIO during a pan) into hard
numbers, so phase 1 can be judged and regressions caught:

  peak_speed / peak_accel / peak_jerk — magnitude of motion and its derivatives.
  reversals   — velocity sign changes above a threshold. A clean point-to-point
                move has <=1 (the settle). Hunting/oscillation shows many; this
                is the direct read of the ±40° 'revolving' failure.
  overshoot   — how far past the target the move goes, as a fraction of the step.
  settling_s  — time from move start until the angle stays within tol of target.
  spectral_concentration — fraction of motion energy below f_cut Hz. Smooth,
                organic motion concentrates energy low (<2-3 Hz); a hunting loop
                puts a peak at its oscillation frequency, dropping this number.

Pure numpy. Works on uneven sampling (uses real timestamps). `simulate()` drives
a HeadController deterministically (no thread) so a harness/test is reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SmoothnessReport:
    peak_speed: float          # °/s
    peak_accel: float          # °/s²
    peak_jerk: float           # °/s³
    reversals: int
    spectral_concentration: float   # 0..1, energy below f_cut
    overshoot: float           # fraction of the step (0 = none); nan if no step
    settling_s: float          # nan if never settles / no step
    samples: int
    duration_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def _derivatives(t: np.ndarray, x: np.ndarray):
    v = np.gradient(x, t)
    a = np.gradient(v, t)
    j = np.gradient(a, t)
    return v, a, j


def count_reversals(v: np.ndarray, *, speed_eps: float = 1.0) -> int:
    """Velocity sign changes while actually moving (|v| > speed_eps °/s). A
    single settle counts as 1; sustained hunting counts many."""
    moving = np.abs(v) > speed_eps
    sign = np.sign(v)
    rev = 0
    last = 0.0
    for s, m in zip(sign, moving):
        if not m:
            continue
        if last != 0.0 and s != 0.0 and s != last:
            rev += 1
        if s != 0.0:
            last = s
    return rev


def spectral_concentration(t: np.ndarray, x: np.ndarray, *, f_cut: float = 2.0) -> float:
    """Fraction of the motion's spectral energy below f_cut Hz, after removing
    the DC/linear trend (we care about how the move is *shaped*, not its size)."""
    n = len(x)
    if n < 8:
        return float("nan")
    # resample to uniform time for a clean FFT
    tu = np.linspace(t[0], t[-1], n)
    xu = np.interp(tu, t, x)
    xu = xu - np.polyval(np.polyfit(tu, xu, 1), tu)   # detrend (remove ramp+DC)
    if np.max(np.abs(xu)) < 1e-9:
        return 1.0   # not moving (only float noise left) = maximally smooth
    dt = (tu[-1] - tu[0]) / (n - 1)
    if dt <= 0:
        return float("nan")
    spec = np.abs(np.fft.rfft(xu)) ** 2
    freqs = np.fft.rfftfreq(n, dt)
    total = spec.sum()
    if total <= 0:
        return 1.0   # perfectly flat = maximally smooth
    return float(spec[freqs <= f_cut].sum() / total)


def analyze_step(t: np.ndarray, x: np.ndarray, *, tol_frac: float = 0.05):
    """Overshoot fraction and settling time for a point-to-point move, inferred
    from the series (start = first sample, target = final settled value)."""
    start = float(x[0])
    target = float(x[-1])
    step = target - start
    if abs(step) < 1e-6:
        return float("nan"), float("nan")
    if step > 0:
        overshoot = max(0.0, (float(x.max()) - target) / step)
    else:
        overshoot = max(0.0, (target - float(x.min())) / (-step))
    tol = abs(step) * tol_frac
    settled_from = t[-1] - t[0]
    # last time it was OUTSIDE tol → settling time is just after that
    outside = np.abs(x - target) > tol
    if outside.any():
        idx = np.where(outside)[0].max()
        settled_from = t[min(idx + 1, len(t) - 1)] - t[0]
    else:
        settled_from = 0.0
    return float(overshoot), float(settled_from)


def summarize(t, x, *, f_cut: float = 2.0, speed_eps: float = 1.0,
              step: bool = True) -> SmoothnessReport:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v, a, j = _derivatives(t, x)
    over, settle = (analyze_step(t, x) if step else (float("nan"), float("nan")))
    return SmoothnessReport(
        peak_speed=float(np.max(np.abs(v))),
        peak_accel=float(np.max(np.abs(a))),
        peak_jerk=float(np.max(np.abs(j))),
        reversals=count_reversals(v, speed_eps=speed_eps),
        spectral_concentration=spectral_concentration(t, x, f_cut=f_cut),
        overshoot=over,
        settling_s=settle,
        samples=len(x),
        duration_s=float(t[-1] - t[0]) if len(t) > 1 else 0.0,
    )


def simulate(controller, schedule, *, dt: float = 0.04, duration: float = 3.0,
             axis: int = 0, gate: str = "track"):
    """Drive a HeadController deterministically (no thread) and capture the
    commanded angle stream. `schedule` is [(t_apply, pan, tilt), ...] setpoints
    applied at/after their time. Returns (t[], angle[]) for the chosen axis
    (0=pan, 1=tilt). Uses controller._step directly so results are reproducible.
    """
    schedule = sorted(schedule, key=lambda s: s[0])
    t0 = 1000.0
    ts, xs = [], []
    si = 0
    n = int(round(duration / dt))
    for k in range(n):
        now = t0 + k * dt
        while si < len(schedule) and schedule[si][0] <= (now - t0):
            _, px, py = schedule[si]
            controller.set_target(px, py)
            si += 1
        _, cx, cy = controller._step(now, gate)
        ts.append(now - t0)
        xs.append(cx if axis == 0 else cy)
    return np.asarray(ts), np.asarray(xs)
