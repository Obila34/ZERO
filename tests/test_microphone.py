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


# -- mics subcommand -----------------------------------------------------

def test_input_devices_filters_output_only(mic_mod):
    _patch_devices(mic_mod, _DEFAULT_DEVICES)
    rows = list(mic_mod._input_devices())
    # 3 of 4 have max_input_channels > 0.
    assert [idx for idx, _ in rows] == [1, 2, 3]


# -- set subcommand: config rewrite --------------------------------------

_CONFIG_SAMPLE = """\
# ZERO configuration
audio:
  sample_rate: 16000
  block_ms: 30
  input_device: Logitech BRIO     # ALSA hw:4,0 — the USB webcam mic (2 in).
                                  # Note: "USB Audio Device" is OUTPUT only.
  input_gain: 12.0
  output_device: pipewire

privacy:
  bystander_mode: guarded
"""


def test_rewrite_input_device_preserves_comment(mic_mod):
    new_text, old_line = mic_mod._rewrite_input_device(_CONFIG_SAMPLE, "USB Audio Device")
    assert "Logitech BRIO" in old_line
    lines = new_text.splitlines()
    # Same LINE INDEX, new value, comment retained.
    changed = [l for l in lines if "input_device:" in l]
    assert len(changed) == 1
    assert "USB Audio Device" in changed[0]
    assert "# ALSA hw:4,0" in changed[0]  # trailing comment preserved
    # Indentation preserved (two spaces).
    assert changed[0].startswith("  input_device:")


def test_rewrite_input_device_leaves_other_lines_alone(mic_mod):
    new_text, _ = mic_mod._rewrite_input_device(_CONFIG_SAMPLE, "2")
    # Every non-input_device line is byte-identical.
    for old, new in zip(_CONFIG_SAMPLE.splitlines(), new_text.splitlines()):
        if "input_device:" in old:
            continue
        assert old == new


def test_rewrite_input_device_numeric_index(mic_mod):
    new_text, _ = mic_mod._rewrite_input_device(_CONFIG_SAMPLE, "2")
    changed = [l for l in new_text.splitlines() if "input_device:" in l]
    assert "input_device: 2" in changed[0]


def test_rewrite_input_device_only_touches_audio_block(mic_mod):
    """A key of the same name outside the audio: block must NOT be rewritten."""
    text = _CONFIG_SAMPLE + """
some_other_section:
  input_device: OTHER   # must remain
"""
    new_text, _ = mic_mod._rewrite_input_device(text, "NEWMIC")
    # Only ONE line changed — the one under audio:.
    assert new_text.count("input_device: NEWMIC") == 1
    assert "input_device: OTHER" in new_text  # sibling section untouched


def test_rewrite_input_device_missing_key_raises(mic_mod):
    empty = "audio:\n  sample_rate: 16000\n"
    with pytest.raises(KeyError):
        mic_mod._rewrite_input_device(empty, "whatever")


# -- device name display --------------------------------------------------

def test_device_display_name_strips_hw_suffix(mic_mod):
    _patch_devices(mic_mod, [
        {"name": "Logitech BRIO: USB Audio (hw:4,0)", "max_input_channels": 2},
    ])
    assert mic_mod._device_display_name(0) == "Logitech BRIO"


def test_device_display_name_strips_paren_suffix(mic_mod):
    _patch_devices(mic_mod, [
        {"name": "Microphone (Realtek Audio)", "max_input_channels": 1},
    ])
    assert mic_mod._device_display_name(0) == "Microphone"


# -- cmd_set: refuses output-only devices --------------------------------

def test_cmd_set_rejects_output_only_device(mic_mod, tmp_path, monkeypatch):
    # Point CONFIG_PATH at a temp file so we don't touch the real one.
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_CONFIG_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(mic_mod, "CONFIG_PATH", cfg)

    _patch_devices(mic_mod, [
        {"name": "USB Audio Device", "max_input_channels": 0},  # output-only!
    ])

    class Args:
        device = "0"
        by = "name"
        dry_run = False

    with pytest.raises(SystemExit) as exc:
        mic_mod.cmd_set(Args())
    assert exc.value.code == 3  # dedicated exit code for "no input channels"
    # Config file untouched.
    assert cfg.read_text(encoding="utf-8") == _CONFIG_SAMPLE


def test_cmd_set_writes_config(mic_mod, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_CONFIG_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(mic_mod, "CONFIG_PATH", cfg)

    _patch_devices(mic_mod, [
        {"name": "Logitech BRIO: USB Audio (hw:4,0)", "max_input_channels": 2},
    ])

    class Args:
        device = "0"
        by = "name"
        dry_run = False

    mic_mod.cmd_set(Args())
    written = cfg.read_text(encoding="utf-8")
    assert "input_device: Logitech BRIO" in written
    assert "# ALSA hw:4,0" in written  # trailing comment survived


def test_cmd_set_dry_run_does_not_write(mic_mod, tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_CONFIG_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(mic_mod, "CONFIG_PATH", cfg)

    _patch_devices(mic_mod, [
        {"name": "Logitech BRIO", "max_input_channels": 2},
    ])

    class Args:
        device = "0"
        by = "name"
        dry_run = True

    mic_mod.cmd_set(Args())
    # File untouched.
    assert cfg.read_text(encoding="utf-8") == _CONFIG_SAMPLE
    out = capsys.readouterr().out
    assert "dry run" in out.lower()


def test_cmd_set_by_index(mic_mod, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_CONFIG_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(mic_mod, "CONFIG_PATH", cfg)

    _patch_devices(mic_mod, [
        {"name": "Logitech BRIO", "max_input_channels": 2},
    ])

    class Args:
        device = "0"
        by = "index"
        dry_run = False

    mic_mod.cmd_set(Args())
    written = cfg.read_text(encoding="utf-8")
    assert "input_device: 0" in written
