"""Voice orchestrator — the "human voice" layer that sits above a TTS engine.

Responsibilities:
  * Own the SHARED cue vocabulary (must match zero/llm/persona.py::CUES).
  * Sentence-segment a reply so playback can start early (streaming).
  * For the Piper path: strip cue tags, synthesize the words, and splice short
    pre-recorded non-verbal clips (laugh/sigh/…) where a cue appeared, plus light
    prosody from *emphasis* / punctuation.
  * For the Fish path: the engine performs cues natively, so the orchestrator
    just hands text straight through (translation happens in fish_engine).

main.py only ever calls `synthesize(text)` and gets back (waveform, sample_rate).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from zero.tts.base import TTS, resample_linear
from zero.utils.logging import get_logger

log = get_logger("tts.orchestrator")

# Shared cue tag -> non-verbal clip filename (Piper path).
CUE_TO_CLIP = {
    "[laughs]": "laugh.wav",
    "[chuckles]": "chuckle.wav",
    "[sighs]": "sigh.wav",
    "[gasps]": "gasp.wav",
    "[hmm]": "hmm.wav",
    "[pause]": None,  # rendered as a short silence, no clip needed
}

# Mood -> speaking-rate multiplier (Piper length_scale; >1 = slower). Orpheus
# carries emotion in-band via cue tags, so mood is a no-op on that path.
MOOD_RATE = {"down": 1.08, "calm": 1.03, "excited": 0.94, "frustrated": 0.97}

_CUE_RE = re.compile(r"(\[[a-z]+\])")
_SENTENCE_RE = re.compile(r"(.+?(?:[.!?…]+|\n|$))", re.S)


def split_sentences(text: str) -> list[str]:
    """Coarse sentence split for streaming TTS. Cheap and good enough for speech."""
    out = [m.group(1).strip() for m in _SENTENCE_RE.finditer(text)]
    return [s for s in out if s]


def _strip_emphasis(text: str) -> str:
    return re.sub(r"\*(\w+)\*", r"\1", text)


class VoiceOrchestrator:
    def __init__(self, engine_name: str, tts: TTS, nonverbals_dir: Path,
                 pause_ms: int = 350):
        self.engine_name = engine_name
        self.tts = tts
        self.sample_rate = tts.sample_rate
        self.nonverbals_dir = Path(nonverbals_dir)
        self.pause_ms = pause_ms
        self._clip_cache: dict[str, np.ndarray] = {}
        self._mood = "neutral"

    def set_mood(self, label: str | None) -> None:
        """Mirror the speaker's mood in delivery. Piper: pacing via
        length_scale; Orpheus: emotion rides in the cue tags, nothing to do."""
        self._mood = label or "neutral"

    @property
    def degraded(self) -> bool:
        """True while the underlying engine is on its fallback voice —
        self-state narration (Phase 7) reads this to be honest about it."""
        return bool(getattr(self.tts, "degraded", False))

    # -- public API ---------------------------------------------------------
    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        if self.engine_name in ("fish", "orpheus"):
            # These perform cues natively; the engine handles cue translation.
            return self.tts.synthesize(text)
        return self._synthesize_piper(text)

    def synthesize_stream(self, text: str):
        """Yield audio chunks as they're produced (real streaming for Orpheus;
        the Piper path falls back to one chunk after splicing cue clips)."""
        text = text.strip()
        if not text:
            return
        if self.engine_name in ("fish", "orpheus"):
            yield from self.tts.synthesize_stream(text)
        else:
            audio = self._synthesize_piper(text)
            if getattr(audio, "size", 0):
                yield audio

    # -- Piper path: split on cues, synthesize words, splice clips -----------
    def _synthesize_piper(self, text: str) -> np.ndarray:
        rate = MOOD_RATE.get(self._mood)
        base = getattr(self.tts, "length_scale", None)
        if rate is not None and base is not None:
            self.tts.length_scale = base * rate
        try:
            parts = _CUE_RE.split(text)
            pieces: list[np.ndarray] = []
            for part in parts:
                if not part.strip():
                    continue
                if part in CUE_TO_CLIP:
                    pieces.append(self._render_cue(part))
                else:
                    words = _strip_emphasis(part).strip()
                    if words:
                        pieces.append(self.tts.synthesize(words))
            if not pieces:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(pieces)
        finally:
            if rate is not None and base is not None:
                self.tts.length_scale = base

    def _render_cue(self, cue: str) -> np.ndarray:
        clip_name = CUE_TO_CLIP.get(cue)
        if clip_name is None:  # [pause]
            return self._silence(self.pause_ms)
        if clip_name not in self._clip_cache:
            self._clip_cache[clip_name] = self._load_clip(clip_name)
        return self._clip_cache[clip_name]

    def _load_clip(self, name: str) -> np.ndarray:
        path = self.nonverbals_dir / name
        if not path.exists():
            log.warning("non-verbal clip missing: %s (using short pause)", path)
            return self._silence(self.pause_ms)
        import soundfile as sf  # lazy

        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if sr != self.sample_rate:
            data = resample_linear(data, sr, self.sample_rate)
        return data

    def _silence(self, ms: int) -> np.ndarray:
        return np.zeros(int(self.sample_rate * ms / 1000), dtype=np.float32)
