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
        min_speech_for_fast_end_ms=cfg.get("vad.min_speech_for_fast_end_ms", 600),
        silero_model=str(cfg.resolve_path("vad.silero_model",
                                          "models/vad/silero_vad.onnx")),
        silero_threshold=cfg.get("vad.silero_threshold", 0.5),
    )


def _build_whispercpp(cfg: Config) -> STT:
    from zero.stt.whispercpp_engine import WhisperCppSTT

    model_path = cfg.resolve_path("stt.model_path", "models/whisper/ggml-base.en.bin")
    return WhisperCppSTT(
        model_path=str(model_path),
        initial_prompt=cfg.get("stt.initial_prompt"),
        threads=cfg.get("stt.threads", 4),
        audio_ctx=cfg.get("stt.audio_ctx", 0),
    )


def build_stt(cfg: Config) -> STT:
    engine = cfg.get("stt.engine", "whispercpp")
    if engine == "remote":
        from zero.stt.fallback import FallbackSTT
        from zero.stt.remote_engine import RemoteSTT

        remote = RemoteSTT(
            url=cfg.get("stt.remote_url", "http://127.0.0.1:9000/transcribe"),
            timeout=cfg.get("stt.remote_timeout", 30),
            language=cfg.get("stt.language"),
        )
        # Optional local failover so a dropped tunnel doesn't leave ZERO deaf.
        # The local engine is built lazily on the first remote failure.
        builder = None
        if cfg.get("stt.fallback") == "whispercpp":
            builder = lambda: _build_whispercpp(cfg)  # noqa: E731
        return FallbackSTT(remote, builder)
    if engine == "whispercpp":
        return _build_whispercpp(cfg)
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
            num_ctx=cfg.get("llm.num_ctx", 8192),
        )
    raise ValueError(f"unknown llm engine: {engine}")


def _build_piper(cfg: Config) -> TTS:
    from zero.tts.piper_engine import PiperTTS

    voice = cfg.resolve_path("tts.piper.voice")
    binary = cfg.get("tts.piper.binary", "piper")
    # A relative path like "piper/piper" -> resolve against the repo root so it
    # works regardless of CWD or where the repo is cloned. A bare "piper"
    # (on PATH) or an absolute path is left as-is.
    if "/" in binary and not Path(binary).is_absolute():
        binary = str(Path(PROJECT_ROOT) / binary)
    return PiperTTS(
        binary=binary,
        voice=str(voice),
        length_scale=cfg.get("tts.piper.length_scale", 1.0),
    )


def _build_tts_engine(cfg: Config) -> tuple[str, TTS]:
    engine = cfg.get("tts.engine", "piper")
    if engine == "orpheus":
        from zero.tts.remote_engine import RemoteTTS

        tts: TTS = RemoteTTS(
            url=cfg.get("tts.orpheus.url", "http://127.0.0.1:9100/tts"),
            voice=cfg.get("tts.orpheus.voice", "tara"),
            timeout=cfg.get("tts.orpheus.timeout", 30),
        )
        # Optional local failover so a dropped tunnel doesn't leave ZERO mute.
        # Piper is built lazily on the first remote failure.
        if cfg.get("tts.fallback") == "piper":
            from zero.tts.fallback import FallbackTTS

            tts = FallbackTTS(tts, lambda: _build_piper(cfg))
        return "orpheus", tts
    if engine == "piper":
        return "piper", _build_piper(cfg)
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


def build_tools(cfg: Config, llm: LLM, memory, events, context_provider=None):
    """Wrap the LLM with the tool router, or return it untouched if tools are
    off. Returns (llm, registry, timer_manager) — registry/timers are None
    when disabled."""
    if not cfg.get("tools.enabled", True):
        return llm, None, None
    from zero.tools.builtin import (
        RecallTool, RememberTool, ReminderTool, TimeTool, TimerTool,
    )
    from zero.tools.registry import ToolRegistry
    from zero.tools.router import ToolAwareLLM
    from zero.tools.timers import TimerManager

    timers = TimerManager(events)
    registry = ToolRegistry(allow=cfg.get("tools.allow"))
    for tool in (TimeTool(), TimerTool(timers), ReminderTool(timers),
                 RememberTool(), RecallTool()):
        registry.register(tool)
    if cfg.get("tools.websearch.enabled", False):
        from zero.tools.websearch import WebSearchTool

        registry.register(WebSearchTool(
            url=cfg.get("tools.websearch.url", "http://127.0.0.1:8888/search"),
            timeout=cfg.get("tools.websearch.timeout_s", 8.0),
            max_results=cfg.get("tools.websearch.max_results", 3),
        ))
    wrapped = ToolAwareLLM(llm, registry, context_provider=context_provider)
    return wrapped, registry, timers


def build_memory(cfg: Config):
    """Long-term memory store, or None if disabled in config."""
    if not cfg.get("memory.enabled", True):
        return None
    from zero.memory.embeddings import build_embedder
    from zero.memory.sqlite_memory import SqliteMemory

    embedder = build_embedder(
        cfg.get("memory.embeddings.backend", "auto"),
        host=cfg.get("llm.host", ""),
        model=cfg.get("memory.embeddings.ollama_model",
                      cfg.get("llm.model", "")),
        hash_dim=cfg.get("memory.embeddings.hash_dim", 256),
    )
    path = cfg.resolve_path("memory.path", "zero_memory.sqlite")
    return SqliteMemory(
        path=str(path),
        max_facts=cfg.get("memory.max_facts", 30),
        recent_episodes=cfg.get("memory.recent_episodes", 3),
        max_stored_facts=cfg.get("memory.max_stored_facts", 200),
        max_stored_episodes=cfg.get("memory.max_stored_episodes", 100),
        embedder=embedder,
        top_k=cfg.get("memory.retrieval.top_k", 6),
        half_life_days=cfg.get("memory.retrieval.half_life_days", 14.0),
        min_activation=cfg.get("memory.retrieval.min_activation", 0.05),
        forget_max_age_days=cfg.get("memory.forgetting.max_age_days", 90.0),
        forgetting=cfg.get("memory.forgetting.enabled", True),
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
    def _local_detector():
        model_path = cfg.resolve_path("vision.detect.model_path", "yolo11n.onnx")
        # Optional explicit open-vocabulary names file; otherwise the detector
        # auto-loads a sibling "<model>.names.json" or falls back to COCO-80.
        names = None
        names_cfg = cfg.get("vision.detect.names_path")
        if names_cfg:
            import json

            names_path = cfg.resolve_path("vision.detect.names_path", names_cfg)
            try:
                names = json.loads(Path(names_path).read_text(encoding="utf-8"))
                log.info("open-vocab names loaded (%d) from %s",
                         len(names), names_path)
            except (ValueError, OSError) as e:
                log.warning("could not load names_path %s — using model "
                            "default: %s", names_path, e)
        return Detector(
            model_path=str(model_path),
            confidence=cfg.get("vision.detect.confidence", 0.35),
            iou=cfg.get("vision.detect.iou", 0.45),
            device=cfg.get("vision.detect.device", "cpu"),
            imgsz=cfg.get("vision.detect.imgsz", 640),
            classes=cfg.get("vision.detect.classes"),
            names=names,
        )

    perception_client = build_perception_client(cfg)
    if perception_client is not None:
        # GPU-first (Phase 8): YOLO11x on the server; local YOLO11s is the
        # lazily-built tunnel-outage fallback.
        from zero.perception.remote import FallbackDetector, RemoteDetector

        detector = FallbackDetector(RemoteDetector(perception_client),
                                    _local_detector)
    else:
        detector = _local_detector()
    namer = ColorNamer(
        center_crop=cfg.get("vision.color.center_crop", 0.6),
        min_saturation=cfg.get("vision.color.min_saturation", 50),
        min_value=cfg.get("vision.color.min_value", 40),
        white_value=cfg.get("vision.color.white_value", 200),
        min_colorful_ratio=cfg.get("vision.color.min_colorful_ratio", 0.10),
        min_pixels=cfg.get("vision.color.min_pixels", 50),
    )
    learned = None
    if cfg.get("learning.objects.enabled", True):
        try:
            from zero.vision.learned import LearnedObjects, build_object_embedder

            def _local_embedder():
                clip_path = None
                if cfg.get("learning.objects.clip_model_path"):
                    p = cfg.resolve_path("learning.objects.clip_model_path")
                    clip_path = str(p) if p is not None and p.exists() else None
                return build_object_embedder(clip_path)

            if perception_client is not None:
                # GPU-first (Phase 8): real CLIP embeddings server-side; the
                # local (histogram/CLIP-onnx) embedder covers outages. CLIP's
                # tighter space earns a lower match threshold.
                from zero.perception.remote import (
                    FallbackObjectEmbedder, RemoteObjectEmbedder,
                )

                embedder = FallbackObjectEmbedder(
                    RemoteObjectEmbedder(perception_client), _local_embedder)
                threshold = cfg.get("learning.objects.match_threshold_clip", 0.72)
            else:
                embedder = _local_embedder()
                threshold = cfg.get("learning.objects.match_threshold", 0.80)
            learned = LearnedObjects(
                str(cfg.resolve_path("learning.objects.db_path",
                                     "zero_objects.sqlite")),
                embedder,
                match_threshold=threshold,
                max_samples_per_object=cfg.get(
                    "learning.objects.max_samples_per_object", 5),
            )
        except Exception as e:
            log.warning("learned objects unavailable: %s", e)

    client = None
    if cfg.get("vision.gpu.enabled", True):
        client = VisionClient(
            url=cfg.get("vision.gpu.url", "http://127.0.0.1:8000"),
            health_path=cfg.get("vision.gpu.health_path", "/health"),
            facts_path=cfg.get("vision.gpu.facts_path", "/facts"),
            analyze_path=cfg.get("vision.gpu.analyze_path", "/analyze"),
            health_timeout_s=cfg.get("vision.gpu.health_timeout_s", 5.0),
            facts_timeout_s=cfg.get("vision.gpu.facts_timeout_s", 15.0),
            analyze_timeout_s=cfg.get("vision.gpu.analyze_timeout_s", 30.0),
            jpeg_quality=cfg.get("vision.gpu.jpeg_quality", 80),
        )
    return Eyes(
        cam, detector, namer, client,
        color_top_n=cfg.get("vision.color.top_n", 5),
        max_items=cfg.get("vision.max_items", 15),
        use_gpu=cfg.get("vision.gpu.enabled", True),
        multimodal=cfg.get("vision.multimodal", False),
        jpeg_quality=cfg.get("vision.gpu.jpeg_quality", 80),
        detect_interval_s=cfg.get("vision.detect_interval_s", 0.0),
        frames_per_look=cfg.get("vision.frames_per_look", 2),
        look_window_s=cfg.get("vision.look_window_s", 1.0),
        vlm_fallback=cfg.get("vision.gpu.vlm_fallback", False),
        preview=cfg.get("vision.preview", False),
        preview_scale=cfg.get("vision.preview_scale", 1.0),
        preview_mode=cfg.get("vision.preview_mode", "auto"),
        # Default PRIVATE: never expose the unauthenticated camera stream on the
        # LAN unless config.yaml explicitly asks for 0.0.0.0.
        preview_host=cfg.get("vision.preview_host", "127.0.0.1"),
        preview_port=cfg.get("vision.preview_port", 8008),
        learned=learned,
        unknown_conf=cfg.get("learning.objects.unknown_conf", 0.45),
        change_note_cooldown_s=cfg.get("vision.change_note_cooldown_s", 120.0),
    )


def build_perception_client(cfg: Config):
    """Shared HTTP client for the GPU /perceive/* endpoints, or None when the
    remote-perception offload is disabled."""
    if not cfg.get("perception.remote.enabled", True):
        return None
    from zero.perception.remote import PerceptionClient

    return PerceptionClient(
        base_url=cfg.get("perception.remote.base_url", "http://127.0.0.1:8000"),
        timeout=cfg.get("perception.remote.timeout_s", 5.0),
        detect_timeout=cfg.get("perception.remote.detect_timeout_s", 10.0),
        jpeg_quality=cfg.get("perception.remote.jpeg_quality", 80),
    )


def build_identity(cfg: Config):
    """Face+voice identity service, or None (anonymous mode) if disabled or no
    usable model is present. Each channel degrades independently: no face model
    -> voice-only; no voice model -> face-only."""
    from zero.utils.logging import get_logger

    log = get_logger("identity")
    if not cfg.get("identity.enabled", False):
        return None
    from zero.identity.fuser import IdentityFuser
    from zero.identity.registry import PersonRegistry
    from zero.identity.service import IdentityService

    client = build_perception_client(cfg)

    def _local_voice():
        v_path = cfg.resolve_path("identity.voice.model_path",
                                  "models/voiceid/voxceleb_ECAPA512_LM.onnx")
        if v_path is None or not v_path.exists():
            raise FileNotFoundError(f"voice model missing: {v_path}")
        from zero.voiceid.speaker import SpeakerVerifier

        return SpeakerVerifier(str(v_path))

    def _local_face():
        f_path = cfg.resolve_path("identity.face.model_path")
        if f_path is None or not f_path.exists():
            raise FileNotFoundError(f"face model missing: {f_path}")
        from zero.identity.face import FaceRecognizer

        return FaceRecognizer(str(f_path),
                              min_size=cfg.get("identity.face.min_size", 60))

    voice = face = None
    if client is not None:
        # GPU-first (Phase 8): embeddings computed server-side, local models
        # only as lazily-built fallbacks for tunnel outages.
        from zero.perception.remote import (
            FallbackFace, FallbackSpeaker, RemoteFaceRecognizer,
            RemoteSpeakerEmbedder,
        )

        voice = FallbackSpeaker(RemoteSpeakerEmbedder(client), _local_voice)
        face = FallbackFace(RemoteFaceRecognizer(client), _local_face)
    else:
        try:
            voice = _local_voice()
        except Exception as e:
            log.warning("identity voice channel unavailable: %s — face-only", e)
        if cfg.get("identity.face.model_path"):
            try:
                face = _local_face()
            except Exception as e:
                log.warning("identity face channel unavailable: %s — voice-only", e)

    if voice is None and face is None:
        log.warning("identity enabled but no usable models — running anonymous")
        return None

    registry = PersonRegistry(
        str(cfg.resolve_path("identity.db_path", "zero_identity.sqlite")),
        max_samples_per_kind=cfg.get("identity.max_samples_per_kind", 5),
    )
    fuser = IdentityFuser(
        w_face=cfg.get("identity.fusion.w_face", 0.55),
        w_voice=cfg.get("identity.fusion.w_voice", 0.45),
        threshold=cfg.get("identity.fusion.threshold", 0.50),
        face_threshold=cfg.get("identity.fusion.face_threshold", 0.42),
        voice_threshold=cfg.get("identity.fusion.voice_threshold", 0.45),
    )
    return IdentityService(
        registry, fuser, voice_embedder=voice, face_recognizer=face,
        reinforce_threshold=cfg.get("identity.reinforce_threshold", 0.70),
    )


def build_guests(cfg: Config):
    """GuestBook for clustering unfamiliar voices into provisional guests, or
    None when disabled. Only useful alongside identity (which supplies the
    per-utterance voice embeddings it clusters)."""
    if not cfg.get("identity.guests.enabled", True):
        return None
    from zero.identity.guests import GuestBook

    return GuestBook(
        str(cfg.resolve_path("identity.guests.db_path", "zero_guests.sqlite")),
        match_threshold=cfg.get("identity.guests.match_threshold", 0.55),
        max_guests=cfg.get("identity.guests.max_guests", 50),
    )


def build_corpus(cfg: Config):
    """Interaction corpus (day-to-day conversations saved as training data), or
    None when disabled."""
    if not cfg.get("learning.corpus.enabled", True):
        return None
    from zero.learning.corpus import Corpus

    return Corpus(str(cfg.resolve_path("learning.corpus.dir", "data/corpus")))


def build_privacy(cfg: Config):
    """(PrivacyGuard, ListeningIndicator) — both always exist; the guard's
    mode and the LED pin come from config."""
    from zero.privacy.guard import PrivacyGuard
    from zero.privacy.indicator import ListeningIndicator

    guard = PrivacyGuard(mode=cfg.get("privacy.bystander_mode", "guarded"))
    indicator = ListeningIndicator(gpio_pin=cfg.get("privacy.indicator_gpio_pin"))
    return guard, indicator


def build_perception(cfg: Config, identity=None):
    """(AffectEstimator | None, SpeakerTracker | None) per config."""
    affect = None
    if cfg.get("perception.affect.enabled", True):
        from zero.perception.affect import AffectEstimator

        fer_path = None
        if cfg.get("perception.affect.face_model_path"):
            p = cfg.resolve_path("perception.affect.face_model_path")
            fer_path = str(p) if p is not None and p.exists() else None
        affect = AffectEstimator(
            face_model_path=fer_path,
            face_detector=getattr(identity, "face_detector", None),
        )
    tracker = None
    if cfg.get("perception.diarize.enabled", True):
        from zero.perception.diarize import SpeakerTracker

        tracker = SpeakerTracker(
            change_threshold=cfg.get("perception.diarize.change_threshold", 0.40))
    return affect, tracker


def build_proactive(cfg: Config, *, events, eyes, identity, memory, is_idle):
    """The proactive stack: curiosity queue + policy + trigger watcher.
    Returns (triggers, curiosity, policy) — all None when disabled. Greetings
    need eyes+identity; curiosity questions and idle learning still run
    without them (from timers/memory alone there's little to do, so the
    watcher only starts when it has at least one real signal source)."""
    from zero.utils.logging import get_logger

    log = get_logger("proactive")
    if not cfg.get("proactive.enabled", True):
        return None, None, None
    from zero.proactive.curiosity import CuriosityStore
    from zero.proactive.policy import InteractionPolicy
    from zero.proactive.triggers import TriggerSource

    curiosity = None
    if cfg.get("learning.curiosity.enabled", True):
        curiosity = CuriosityStore(
            str(cfg.resolve_path("learning.curiosity.db_path",
                                 "zero_curiosity.sqlite")),
            max_pending=cfg.get("learning.curiosity.max_pending", 20),
        )
    quiet = cfg.get("proactive.quiet_hours", [22, 7])
    engage_unknown = cfg.get("proactive.engage_unknown", False)
    policy = InteractionPolicy(
        enabled=True,
        greet_cooldown_s=cfg.get("proactive.greet_cooldown_s", 4 * 3600),
        curiosity_cooldown_s=cfg.get("proactive.curiosity_cooldown_s", 1800),
        remark_cooldown_s=cfg.get("proactive.remark_cooldown_s", 300),
        max_per_hour=cfg.get("proactive.max_per_hour", 6),
        quiet_hours=tuple(quiet) if quiet else None,
        presence_reset_s=cfg.get("proactive.presence_reset_s", 1200),
        engage_unknown=engage_unknown,
    )
    if eyes is None and identity is None and curiosity is None:
        log.info("proactive on, but no signal sources — watcher not started")
        return None, curiosity, policy
    triggers = TriggerSource(
        events=events, policy=policy, eyes=eyes, identity=identity,
        curiosity=curiosity, memory=memory, is_idle=is_idle,
        check_interval_s=cfg.get("proactive.check_interval_s", 3.0),
        linger_before_question_s=cfg.get("proactive.linger_s", 45.0),
        consolidate_interval_s=cfg.get("proactive.consolidate_interval_s", 1800),
        engage_unknown=engage_unknown,
    )
    return triggers, curiosity, policy


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
