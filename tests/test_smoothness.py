"""The smoothness metrics must discriminate a clean slew from hunting, and the
revived HeadController must pass the clean bar on a saccade-sized step."""
import numpy as np

from zero.head.controller import HeadController
from zero.head.smoothness import (count_reversals, simulate,
                                   spectral_concentration, summarize)


def test_clean_slew_has_at_most_one_reversal():
    c = HeadController(lambda x, y: None, rate_hz=25.0, max_speed_dps=36.0)
    t, x = simulate(c, [(0.1, 40.0, 0.0)], dt=0.04, duration=3.0, axis=0)
    r = summarize(t, x)
    assert r.reversals <= 1                    # no hunting
    assert r.overshoot <= 0.02                 # slew never overshoots
    assert r.spectral_concentration > 0.9      # energy sits low = smooth
    assert r.peak_speed <= 36.0 + 1.0          # respects the slew ceiling


def test_hunting_signal_is_flagged():
    # a ringing step to 40° = the ±40° 'revolving' signature: starts at 0, rings
    # past the target and oscillates in before settling.
    t = np.linspace(0, 3, 300)
    x = 40.0 - 40.0 * np.exp(-1.0 * t) * np.cos(2 * np.pi * 3.0 * t)
    r = summarize(t, x)
    assert r.reversals >= 6                     # many velocity reversals
    assert r.spectral_concentration < 0.5       # energy peaks at the 3 Hz hunt
    assert r.overshoot > 0.1                     # rings past the target


def test_reversal_counter_ignores_micro_jitter():
    # tiny sub-threshold noise on a monotone ramp is not a reversal
    t = np.linspace(0, 2, 200)
    v = np.ones_like(t) * 10.0 + 0.1 * np.sin(50 * t)  # |noise| < speed_eps
    assert count_reversals(v, speed_eps=1.0) == 0


def test_spectral_concentration_flat_is_one():
    t = np.linspace(0, 2, 128)
    x = np.full_like(t, 5.0)                     # not moving at all
    assert spectral_concentration(t, x) >= 0.99
