"""Unit tests for test_microphone.py — the mic diagnostic script.

Covers the pure helpers (device resolution + gain suggestion). Recording and
playback are hardware-driven and not tested here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def mic_mod(monkeypatch_module=None):
    """Import test_microphone.py by path (it lives at the repo root, not on
    sys.path as a package). sounddevice is stubbed BEFORE import so the module
    loads on any dev box."""
    import types

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.query_devices = lambda idx=None: (
        {"name": "Fake", "max_input_channels": 1}
        if idx is not None else _DEFAULT_DEVICES
    )
    fake_sd.default = types.SimpleNamespace(device=(0, 0))
    sys.modules["sounddevice"] = fake_sd

    path = Path(__file__).resolve().parents[1] / "test_microphone.py"
    spec = importlib.util.spec_from_file_location("test_microphone_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DEFAULT_DEVICES = [
    {"name": "Speakers (Realtek)",       "max_input_channels": 0},
    {"name": "Microphone (Realtek)",     "max_input_channels": 1},
    {"name": "Logitech BRIO",            "max_input_channels": 2},
    {"name": "USB Camera (Logitech)",    "max_input_channels": 1},
]


def _patch_devices(mic_mod, devices):
    mic_mod.sd.query_devices = lambda idx=None: (
        devices[idx] if idx is not None else devices
    )


def test_resolve_device_none_returns_none(mic_mod):
    assert mic_mod.resolve_device(None) is None


def test_resolve_device_numeric_string(mic_mod):
    assert mic_mod.resolve_device("2") == 2


def test_resolve_device_integer(mic_mod):
    assert mic_mod.resolve_device(5) == 5


def test_resolve_device_name_substring(mic_mod):
    _patch_devices(mic_mod, _DEFAULT_DEVICES)
    # "Brio" matches index 2 (case-insensitive).
    assert mic_mod.resolve_device("Brio") == 2
    assert mic_mod.resolve_device("brio") == 2


def test_resolve_device_no_match_exits(mic_mod):
    _patch_devices(mic_mod, _DEFAULT_DEVICES)
    with pytest.raises(SystemExit) as exc:
        mic_mod.resolve_device("nonexistent-mic")
    assert exc.value.code == 2


def test_resolve_device_ambiguous_exits(mic_mod):
    _patch_devices(mic_mod, _DEFAULT_DEVICES)
    # "Logitech" matches BOTH the BRIO and the USB Camera.
    with pytest.raises(SystemExit) as exc:
        mic_mod.resolve_device("Logitech")
    assert exc.value.code == 2


def test_resolve_device_ignores_output_only(mic_mod):
    # An output-only device with a matching name must NOT be picked.
    _patch_devices(mic_mod, [
        {"name": "Brio Speakers",  "max_input_channels": 0},  # output — skip
        {"name": "Brio Mic",       "max_input_channels": 1},
    ])
    assert mic_mod.resolve_device("Brio") == 1


def test_gain_suggestion_healthy_level(mic_mod):
    # Peak >= 0.60 -> leave gain off.
    msg = mic_mod._suggest_gain(0.75)
    assert "1.0" in msg and "healthy" in msg.lower()


def test_gain_suggestion_quiet(mic_mod):
    # Peak in the quiet band -> suggest a moderate boost.
    msg = mic_mod._suggest_gain(0.20)
    assert "input_gain" in msg
    # Target is ~0.4, so suggested gain should be ~2.
    assert "2." in msg


def test_gain_suggestion_very_quiet_brio(mic_mod):
    # Peak ~0.03 (documented BRIO level) -> suggest a strong boost around 13.
    msg = mic_mod._suggest_gain(0.03)
    assert "input_gain" in msg
    # Suggested gain = 0.4 / 0.03 ~= 13.3, so expect a two-digit number.
    for token in msg.replace(":", " ").split():
        try:
            val = float(token)
        except ValueError:
            continue
        if 10.0 <= val <= 16.0:
            break
    else:
        pytest.fail(f"expected a gain in [10, 16] range in suggestion: {msg!r}")
