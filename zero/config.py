"""Config loading: read config.yaml (+ optional config.local.yaml override).

Returns a plain dict accessed with dotted helpers so call sites read cleanly,
e.g. cfg.get("tts.engine", "piper").
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Dotted-access wrapper around the merged config dict."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        val = self._data.get(name, {})
        return val if isinstance(val, dict) else {}

    def resolve_path(self, dotted: str, default: str | None = None) -> Path | None:
        """Resolve a config path value relative to the project root."""
        val = self.get(dotted, default)
        if val is None:
            return None
        p = Path(val)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    base_path = Path(path) if path else (PROJECT_ROOT / "config.yaml")
    with open(base_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    local_path = PROJECT_ROOT / "config.local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})

    return Config(data)
