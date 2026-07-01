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
        min_utterance_rms=cfg.get("vad.min_utterance_rms", 0),
    )


def build_stt(cfg: Config) -> STT:
    engine = cfg.get("stt.engine", "whispercpp")
    if engine == "remote":
        from zero.stt.remote_engine import RemoteSTT

        return RemoteSTT(
            url=cfg.get("stt.remote_url", "http://127.0.0.1:9000/transcribe"),
            timeout=cfg.get("stt.remote_timeout", 30),
        )
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
    if engine == "orpheus":
        from zero.tts.remote_engine import RemoteTTS

        return "orpheus", RemoteTTS(
            url=cfg.get("tts.orpheus.url", "http://127.0.0.1:9100/tts"),
            voice=cfg.get("tts.orpheus.voice", "tara"),
            timeout=cfg.get("tts.orpheus.timeout", 30),
        )
    if engine == "piper":
        from zero.tts.piper_engine import PiperTTS

        voice = cfg.resolve_path("tts.piper.voice")
        binary = cfg.get("tts.piper.binary", "piper")
        # A relative path like "piper/piper" -> resolve against the repo root so it
        # works regardless of CWD or where the repo is cloned. A bare "piper"
        # (on PATH) or an absolute path is left as-is.
        if "/" in binary and not Path(binary).is_absolute():
            binary = str(Path(PROJECT_ROOT) / binary)
        return "piper", PiperTTS(
            binary=binary,
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
    return SqliteMemory(
        path=str(path),
        max_facts=cfg.get("memory.max_facts", 30),
        recent_episodes=cfg.get("memory.recent_episodes", 3),
    )


def build_vision(cfg: Config):
    """Build the always-on Eyes (camera + YOLO + color + GPU client), or None.

    Returns None when vision is disabled in config, or when the optional camera
    stack (OpenCV / Ultralytics) isn't installed — ZERO then runs voice-only.
    """
    from zero.utils.logging import get_logger

    log = get_logger("vision")
    if not cfg.get("vision.enabled", False):
        return None
    try:
        from zero.vision.camera import CameraStream
        from zero.vision.color_namer import ColorNamer
        from zero.vision.detector import Detector
        from zero.vision.eyes import Eyes
        from zero.vision.gpu_client import VisionClient
    except ImportError as e:  # pragma: no cover - package import should not fail
        log.warning("vision enabled but import failed — running voice-only: %s", e)
        return None

    cam = CameraStream(
        index=cfg.get("vision.camera.index", 0),
        width=cfg.get("vision.camera.width", 640),
        height=cfg.get("vision.camera.height", 480),
        request_fps=cfg.get("vision.camera.request_fps", 30),
        mjpg=cfg.get("vision.camera.mjpg", True),
    )
    model_path = cfg.resolve_path("vision.detect.model_path", "yolo11n.onnx")
    detector = Detector(
        model_path=str(model_path),
        confidence=cfg.get("vision.detect.confidence", 0.35),
        iou=cfg.get("vision.detect.iou", 0.45),
        device=cfg.get("vision.detect.device", "cpu"),
        imgsz=cfg.get("vision.detect.imgsz", 640),
        classes=cfg.get("vision.detect.classes"),
    )
    namer = ColorNamer(
        center_crop=cfg.get("vision.color.center_crop", 0.6),
        min_saturation=cfg.get("vision.color.min_saturation", 50),
        min_value=cfg.get("vision.color.min_value", 40),
        white_value=cfg.get("vision.color.white_value", 200),
        min_colorful_ratio=cfg.get("vision.color.min_colorful_ratio", 0.10),
        min_pixels=cfg.get("vision.color.min_pixels", 50),
    )
    client = None
    if cfg.get("vision.gpu.enabled", True):
        client = VisionClient(
            url=cfg.get("vision.gpu.url", "http://127.0.0.1:8000"),
            health_path=cfg.get("vision.gpu.health_path", "/health"),
            facts_path=cfg.get("vision.gpu.facts_path", "/facts"),
            health_timeout_s=cfg.get("vision.gpu.health_timeout_s", 5.0),
            facts_timeout_s=cfg.get("vision.gpu.facts_timeout_s", 15.0),
            jpeg_quality=cfg.get("vision.gpu.jpeg_quality", 80),
        )
    return Eyes(
        cam, detector, namer, client,
        color_top_n=cfg.get("vision.color.top_n", 5),
        max_items=cfg.get("vision.max_items", 6),
        use_gpu=cfg.get("vision.gpu.enabled", True),
        multimodal=cfg.get("vision.multimodal", False),
        jpeg_quality=cfg.get("vision.gpu.jpeg_quality", 80),
        detect_interval_s=cfg.get("vision.detect_interval_s", 0.0),
    )


def build_voiceid(cfg: Config):
    """Speaker verifier + enrolled voiceprint, or (None, None) if off/missing."""
    import numpy as np

    from zero.utils.logging import get_logger

    log = get_logger("voiceid")
    if not cfg.get("voiceid.enabled", False):
        return None, None

    model = cfg.resolve_path("voiceid.model_path", "models/voiceid/voxceleb_ECAPA512_LM.onnx")
    profile = cfg.resolve_path("voiceid.profile_path", "voiceprint.npy")
    if not model.exists() or not profile.exists():
        log.warning("voiceid enabled but model or voiceprint missing — disabling "
                    "(run: python test_voiceid.py enroll)")
        return None, None

    from zero.voiceid.speaker import SpeakerVerifier

    verifier = SpeakerVerifier(str(model), threshold=cfg.get("voiceid.threshold", 0.45))
    return verifier, np.load(str(profile))
