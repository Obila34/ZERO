"""Piper TTS — fast, local, the engine actually optimized for Raspberry Pi.

Shells out to the native `piper` binary with --output-raw, which writes int16
PCM mono to stdout at the voice's native sample rate (read from the voice's
sidecar .onnx.json). Emotion/non-verbal expressiveness for this engine is added
by VoiceOrchestrator, not here — Piper just speaks plain words.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from zero.tts.base import TTS
from zero.utils.logging import get_logger

log = get_logger("tts.piper")


class PiperTTS(TTS):
    def __init__(self, binary: str, voice: str, length_scale: float = 1.0):
        # Validate NOW so a missing binary trips FallbackTTS's build-once guard
        # instead of failing every single sentence (the error flood when the
        # Orpheus tunnel drops on a box where Piper was never installed).
        if not self._resolve_binary(binary):
            raise FileNotFoundError(f"piper binary not found: {binary}")
        self.binary = binary
        self.voice = str(voice)
        self.length_scale = length_scale
        self.sample_rate = self._read_sample_rate(voice)
        log.info("Piper ready (voice=%s, %d Hz)", Path(voice).name, self.sample_rate)

    @staticmethod
    def _resolve_binary(binary: str) -> bool:
        if "/" in binary:
            return Path(binary).is_file()
        return shutil.which(binary) is not None

    @staticmethod
    def _read_sample_rate(voice: str) -> int:
        # Piper voices ship "<voice>.onnx.json" with audio.sample_rate.
        cfg_path = Path(str(voice) + ".json")
        if not cfg_path.exists():
            cfg_path = Path(str(voice).replace(".onnx", ".onnx.json"))
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return int(json.load(f)["audio"]["sample_rate"])
        except Exception:  # noqa: BLE001
            return 22050

    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        cmd = [
            self.binary,
            "--model", self.voice,
            "--length_scale", str(self.length_scale),
            "--output-raw",
        ]
        try:
            proc = subprocess.run(
                cmd, input=text.encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.error("Piper synthesis failed: %s", e)
            return np.zeros(0, dtype=np.float32)
        pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return pcm
