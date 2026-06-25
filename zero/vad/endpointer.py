"""Utterance endpointing: after the wake word, capture audio until the user
stops talking (trailing silence) or a hard length cap is hit.

Two backends:
  * silero  — neural VAD, robust to noise (preferred). Loaded via torch.
  * webrtc  — webrtcvad, ultra-light, no torch. Fallback.

Both expose the same `capture()` API: consume frames from an iterator, return
the collected utterance as a float32 mono array (or None if nothing was said).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("vad")


class _BaseEndpointer:
    def __init__(self, sample_rate: int, silence_ms: int, max_utterance_ms: int,
                 speech_pad_ms: int, block_ms: int, energy_threshold: float = 0.0,
                 min_utterance_rms: float = 0.0):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.silence_blocks = max(1, silence_ms // block_ms)
        self.max_blocks = max(1, max_utterance_ms // block_ms)
        self.pad_blocks = max(0, speech_pad_ms // block_ms)
        # int16 RMS a single frame must exceed to count as speech. Rejects quieter
        # background voices/room noise that the VAD alone would accept. 0 = off.
        self.energy_threshold = energy_threshold
        # int16 RMS the WHOLE utterance must average — a proximity gate: your voice
        # (close to the mic) is loud; people across the room are not. 0 = off.
        self.min_utterance_rms = min_utterance_rms

    def _is_speech(self, frame: np.ndarray) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def _energetic(self, frame: np.ndarray) -> bool:
        if self.energy_threshold <= 0:
            return True
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        return rms >= self.energy_threshold

    def capture(self, frames: Iterable[np.ndarray],
                idle_timeout_s: float | None = None) -> np.ndarray | None:
        """Collect one utterance. If `idle_timeout_s` is set and no speech starts
        within that window, return None (lets the caller sleep the conversation).
        """
        collected: list[np.ndarray] = []
        preroll: list[np.ndarray] = []  # recent frames kept so we don't clip word 1
        trailing_silence = 0
        started = False
        idle_blocks_seen = 0
        idle_limit = int(idle_timeout_s * 1000 / self.block_ms) if idle_timeout_s else None

        for frame in frames:
            is_voice = self._is_speech(frame)  # VAD only

            if not started:
                # Keep a short lead-in so the first word isn't clipped.
                preroll.append(frame)
                if len(preroll) > self.pad_blocks + 1:
                    preroll.pop(0)
                # STARTING requires real speech AND enough loudness — this is where
                # the loudness gate belongs (rejects quiet background onsets).
                if is_voice and self._energetic(frame):
                    started = True
                    collected.extend(preroll)
                    trailing_silence = 0
                else:
                    idle_blocks_seen += 1
                    if idle_limit and idle_blocks_seen >= idle_limit:
                        return None
            else:
                collected.append(frame)
                # CONTINUING uses the VAD only — quiet syllables/short pauses inside
                # a sentence must NOT end it (that was the fragmentation bug).
                if is_voice:
                    trailing_silence = 0
                else:
                    trailing_silence += 1
                    if trailing_silence >= self.silence_blocks:
                        break
                if len(collected) >= self.max_blocks:
                    log.info("utterance hit max length cap")
                    break

        if not collected:
            return None

        pcm_i16 = np.concatenate(collected)
        rms = float(np.sqrt(np.mean(pcm_i16.astype(np.float32) ** 2)))
        peak = int(np.abs(pcm_i16).max())
        log.info("utterance: rms=%.0f peak=%d (%.1fs)", rms, peak,
                 len(collected) * self.block_ms / 1000.0)

        # Proximity gate: a whole utterance that's quiet on average is almost
        # certainly background, not the close speaker — drop it.
        if self.min_utterance_rms and rms < self.min_utterance_rms:
            log.info("dropped: too quiet (rms %.0f < %.0f) — likely background",
                     rms, self.min_utterance_rms)
            return None

        return pcm_i16.astype(np.float32) / 32768.0


class SileroEndpointer(_BaseEndpointer):
    def __init__(self, **kw):
        super().__init__(**kw)
        import torch  # lazy

        self._torch = torch
        # silero-vad ships via torch.hub; cache after first download.
        model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._model = model
        self._threshold = 0.5
        log.info("silero VAD loaded")

    def _is_speech(self, frame: np.ndarray) -> bool:
        # silero wants float32 in [-1, 1]; it expects 512-sample chunks at 16 kHz.
        x = self._torch.from_numpy(frame.astype(np.float32) / 32768.0)
        prob = float(self._model(x, self.sample_rate).item())
        return prob >= self._threshold


class WebrtcEndpointer(_BaseEndpointer):
    def __init__(self, aggressiveness: int = 3, **kw):
        super().__init__(**kw)
        import webrtcvad  # lazy

        # 0-3, higher = more aggressive at rejecting non-speech (background murmur).
        self._vad = webrtcvad.Vad(int(aggressiveness))
        log.info("webrtc VAD loaded (aggressiveness=%d, energy_threshold=%.0f)",
                 aggressiveness, self.energy_threshold)

    def _is_speech(self, frame: np.ndarray) -> bool:
        # webrtcvad needs 10/20/30 ms int16 frames at 8/16/32/48 kHz.
        return self._vad.is_speech(frame.tobytes(), self.sample_rate)


def build_endpointer(engine: str, **kw) -> _BaseEndpointer:
    if engine == "webrtc":
        return WebrtcEndpointer(**kw)
    return SileroEndpointer(**kw)
