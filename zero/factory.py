"""Build pipeline stages from config. The ONE place that maps engine names in
config.yaml to concrete classes — so swapping an engine never touches main.py.
"""
from __future__ import annotations

from pathlib import Path

from zero.config import Config, PROJECT_ROOT
from zero.llm.base import LLM
from zero.stt.base import STT
from zero.tts.base import TTS
from zero.tts.orchestrator import VoiceOrchestrator
from zero.wake.base import WakeWord


def build_wake(cfg: Config) -> WakeWord:
    engine = cfg.get("wake.engine", "openwakeword")
    if engine == "openwakeword":
        from zero.wake.openwakeword_engine import OpenWakeWordEngine

        return OpenWakeWordEngine(
            model=cfg.get("wake.model", "hey_jarvis"),
            threshold=cfg.get("wake.threshold", 0.5),
        )
    raise ValueError(f"unknown wake engine: {engine}")


def build_endpointer(cfg: Config):
    from zero.vad.endpointer import build_endpointer as _build

    return _build(
        engine=cfg.get("vad.engine", "silero"),
        sample_rate=cfg.get("audio.sample_rate", 16000),
        silence_ms=cfg.get("vad.silence_ms", 800),
        max_utterance_ms=cfg.get("vad.max_utterance_ms", 15000),
        speech_pad_ms=cfg.get("vad.speech_pad_ms", 200),
        block_ms=cfg.get("audio.block_ms", 30),
        aggressiveness=cfg.get("vad.aggressiveness", 3),
        energy_threshold=cfg.get("vad.energy_threshold", 350),
    )


def build_stt(cfg: Config) -> STT:
    engine = cfg.get("stt.engine", "whispercpp")
    if engine == "whispercpp":
        from zero.stt.whispercpp_engine import WhisperCppSTT

        model_path = cfg.resolve_path("stt.model_path", "models/whisper/ggml-base.en.bin")
        return WhisperCppSTT(
            model_path=str(model_path),
            initial_prompt=cfg.get("stt.initial_prompt"),
            threads=cfg.get("stt.threads", 4),
            audio_ctx=cfg.get("stt.audio_ctx", 0),
        )
    raise ValueError(f"unknown stt engine: {engine}")


def build_llm(cfg: Config) -> LLM:
    engine = cfg.get("llm.engine", "ollama")
    if engine == "ollama":
        from zero.llm.ollama_engine import OllamaLLM

        return OllamaLLM(
            host=cfg.get("llm.host", "http://127.0.0.1:11434"),
            model=cfg.get("llm.model", "llama3.2:3b"),
            temperature=cfg.get("llm.temperature", 0.7),
            max_tokens=cfg.get("llm.max_tokens", 160),
        )
    raise ValueError(f"unknown llm engine: {engine}")


def _build_tts_engine(cfg: Config) -> tuple[str, TTS]:
    engine = cfg.get("tts.engine", "piper")
    if engine == "piper":
        from zero.tts.piper_engine import PiperTTS

        voice = cfg.resolve_path("tts.piper.voice")
        return "piper", PiperTTS(
            binary=cfg.get("tts.piper.binary", "piper"),
            voice=str(voice),
            length_scale=cfg.get("tts.piper.length_scale", 1.0),
        )
    if engine == "fish":
        from zero.tts.fish_engine import FishTTS

        model_dir = cfg.resolve_path("tts.fish.model_dir")
        return "fish", FishTTS(
            model_dir=str(model_dir),
            precision=cfg.get("tts.fish.precision", "int8"),
            infer_cmd=cfg.get("tts.fish.infer_cmd"),
        )
    raise ValueError(f"unknown tts engine: {engine}")


def build_voice(cfg: Config) -> VoiceOrchestrator:
    engine_name, tts = _build_tts_engine(cfg)
    nonverbals = Path(PROJECT_ROOT) / "zero" / "tts" / "nonverbals"
    return VoiceOrchestrator(engine_name=engine_name, tts=tts, nonverbals_dir=nonverbals)


def build_memory(cfg: Config):
    """Long-term memory store, or None if disabled in config."""
    if not cfg.get("memory.enabled", True):
        return None
    from zero.memory.sqlite_memory import SqliteMemory

    path = cfg.resolve_path("memory.path", "zero_memory.sqlite")
    return SqliteMemory(path=str(path), max_facts=cfg.get("memory.max_facts", 30))
