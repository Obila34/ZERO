"""Utterance endpointing: after the wake word, capture audio until the user
stops talking (trailing silence) or a hard length cap is hit.

Two backends:
  * silero  — neural VAD, robust to noise (preferred). Loaded via torch.
  * webrtc  — webrtcvad, ultra-light, no torch. Fallback.

Both expose the same `capture()` API: consume frames from an iterator, return
the collected utterance as a float32 mono array (or None if nothing was said).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("vad")


class _BaseEndpointer:
    def __init__(self, sample_rate: int, silence_ms: int, max_utterance_ms: int,
                 speech_pad_ms: int, block_ms: int, energy_threshold: float = 0.0,
                 min_utterance_rms: float = 0.0,
                 min_speech_for_fast_end_ms: int = 0,
                 semantic_hold_wait_ms: int = 600):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.silence_blocks = max(1, silence_ms // block_ms)
        self.max_blocks = max(1, max_utterance_ms // block_ms)
        self.pad_blocks = max(0, speech_pad_ms // block_ms)
        # How long past the normal endpoint we're willing to wait for the
        # speculative transcript to arrive, when should_hold() says "don't know
        # yet" (None). Without this the mid-thought check raced the STT round
        # trip and lost — the answer arrived just after the endpoint committed,
        # so half-sentences were shipped to the LLM.
        self.hold_wait_blocks = max(0, semantic_hold_wait_ms // block_ms)
        # Adaptive endpoint: an utterance with less speech than this waits DOUBLE
        # the trailing silence before ending — slow starters ("um...") aren't cut
        # off, while a finished sentence still commits fast. 0 = off.
        self.fast_end_speech_blocks = max(0, min_speech_for_fast_end_ms // block_ms)
        # Speculative pause hook: this far into a silence run (well before the
        # endpoint), capture() calls on_speech_pause(audio_so_far) so STT can
        # start while we're still waiting to confirm the turn is over.
        self.spec_blocks = max(1, 180 // block_ms)
        # int16 RMS a single frame must exceed to count as speech. Rejects quieter
        # background voices/room noise that the VAD alone would accept. 0 = off.
        self.energy_threshold = energy_threshold
        # int16 RMS the WHOLE utterance must average — a proximity gate: your voice
        # (close to the mic) is loud; people across the room are not. 0 = off.
        self.min_utterance_rms = min_utterance_rms

    def _is_speech(self, frame: np.ndarray) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def _on_capture_start(self) -> None:
        """Reset per-utterance VAD state at the start of each capture. No-op by
        default; stateful backends (silero) override it."""

    def _energetic(self, frame: np.ndarray) -> bool:
        if self.energy_threshold <= 0:
            return True
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        return rms >= self.energy_threshold

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        """Frame-level speech check for callers outside capture() (barge-in)."""
        try:
            return self._is_speech(frame)
        except Exception:
            return False

    def capture(self, frames: Iterable[np.ndarray],
                idle_timeout_s: float | None = None,
                on_speech_pause=None,
                should_hold=None) -> np.ndarray | None:
        """Collect one utterance from the owner. Returns the audio, or None only on
        a true idle timeout (no speech for `idle_timeout_s`). A too-quiet utterance
        (background) is NOT an idle timeout — we drop it and keep listening, so a
        stray background blip doesn't end the conversation.
        """
        self._on_capture_start()   # reset stateful VAD (silero) per utterance
        idle_limit = int(idle_timeout_s * 1000 / self.block_ms) if idle_timeout_s else None
        idle_blocks_seen = 0
        frames_iter = iter(frames)

        while True:
            collected: list[np.ndarray] = []
            preroll: list[np.ndarray] = []  # lead-in so the first word isn't clipped
            trailing_silence = 0
            started = False
            hold_until = 0  # semantic hold: extended endpoint (blocks), 0 = none

            for frame in frames_iter:
                is_voice = self._is_speech(frame)  # VAD only

                if not started:
                    preroll.append(frame)
                    if len(preroll) > self.pad_blocks + 1:
                        preroll.pop(0)
                    # STARTING needs real speech AND enough loudness (rejects quiet
                    # background onsets).
                    if is_voice and self._energetic(frame):
                        started = True
                        collected.extend(preroll)
                        trailing_silence = 0
                    else:
                        idle_blocks_seen += 1
                        # Heartbeat: proves the listener is alive AND receiving
                        # frames while it waits for you to start talking. If this
                        # ticks but your speech isn't caught -> VAD/level issue.
                        # If it never ticks on a turn -> no frames are arriving
                        # (the mic stream died), a different problem entirely.
                        if idle_blocks_seen % max(1, int(5000 / self.block_ms)) == 0:
                            log.info("listening… (%.0fs, no speech yet)",
                                     idle_blocks_seen * self.block_ms / 1000.0)
                        if idle_limit and idle_blocks_seen >= idle_limit:
                            return None
                else:
                    collected.append(frame)
                    # CONTINUING uses the VAD only — quiet syllables/short pauses
                    # inside a sentence must NOT end it (the fragmentation bug).
                    if is_voice:
                        trailing_silence = 0
                        hold_until = 0  # speech resumed — a fresh endpoint decision
                    else:
                        trailing_silence += 1
                        if (on_speech_pause is not None
                                and trailing_silence == self.spec_blocks):
                            try:  # speculative STT must never break capture
                                on_speech_pause(np.concatenate(collected))
                            except Exception as e:
                                log.debug("speech-pause hook failed: %s", e)
                        need = self.silence_blocks
                        if (self.fast_end_speech_blocks
                                and len(collected) - trailing_silence
                                < self.fast_end_speech_blocks):
                            need = self.silence_blocks * 2
                        if hold_until:
                            need = max(need, hold_until)
                        if trailing_silence >= need:
                            # Semantic hold (once per pause): the early partial
                            # transcript says whether the user is mid-thought
                            # ("...and"). Tri-state: True = hold one more silence
                            # window; False = commit now; None = the transcript
                            # is STILL IN FLIGHT — wait (bounded) instead of
                            # racing it, so half-sentences aren't shipped just
                            # because STT was a beat slower than the endpoint.
                            decision = False
                            if hold_until == 0 and should_hold is not None:
                                try:
                                    decision = should_hold()
                                except Exception as e:
                                    log.debug("hold hook failed: %s", e)
                            if decision is None:
                                if trailing_silence < need + self.hold_wait_blocks:
                                    continue  # transcript pending — wait a beat
                                decision = False  # waited long enough; commit
                            if decision:
                                hold_until = trailing_silence + self.silence_blocks
                                log.debug("semantic hold: mid-thought pause")
                            else:
                                break
                    if len(collected) >= self.max_blocks:
                        log.info("utterance hit max length cap")
                        break

            if not collected:
                return None  # frames exhausted

            pcm_i16 = np.concatenate(collected)
            rms = float(np.sqrt(np.mean(pcm_i16.astype(np.float32) ** 2)))
            peak = int(np.abs(pcm_i16).max())
            log.info("utterance: rms=%.0f peak=%d (%.1fs)", rms, peak,
                     len(collected) * self.block_ms / 1000.0)

            # Proximity gate: a whole utterance that's quiet on average is almost
            # certainly background. Drop it and KEEP LISTENING (don't sleep).
            if self.min_utterance_rms and rms < self.min_utterance_rms:
                log.info("dropped: too quiet (rms %.0f < %.0f) — still listening",
                         rms, self.min_utterance_rms)
                idle_blocks_seen = 0  # there was activity; give a fresh idle window
                continue

            return pcm_i16.astype(np.float32) / 32768.0


class SileroEndpointer(_BaseEndpointer):
    """Silero-VAD via ONNX Runtime — no torch (keeps the Pi torch-free).

    Silero v5 requires exactly 512-sample windows at 16 kHz, but our frames are
    480 (30 ms), so we buffer and feed whole 512-windows in order, carrying the
    model's recurrent state between them. The ONNX I/O is smoke-tested at
    construction, so an API/version mismatch raises here and the factory falls
    back to webrtc rather than silently deafening the mic at runtime.
    """

    _WINDOW = 512  # samples @ 16 kHz that silero v5 demands

    def __init__(self, model_path: str | None = None, threshold: float = 0.5, **kw):
        super().__init__(**kw)
        import onnxruntime as ort  # already on the Pi (wake word / YOLO)

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"silero VAD model missing: {model_path}")
        self._sess = ort.InferenceSession(model_path,
                                          providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._sess.get_inputs()}
        self._threshold = float(threshold)
        self._sr = np.array(self.sample_rate, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        # Validate the ONNX interface NOW (deaf-mic insurance) — a mismatch
        # raises and build_endpointer() falls back to webrtc.
        self._run(np.zeros(self._WINDOW, dtype=np.float32))
        self._on_capture_start()
        log.info("silero VAD (onnx) loaded (%s)", model_path)

    def _on_capture_start(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0

    def _run(self, chunk: np.ndarray) -> float:
        feed = {"input": chunk[None, :].astype(np.float32), "sr": self._sr}
        if "state" in self._input_names:
            feed["state"] = self._state
        out = self._sess.run(None, feed)
        prob = float(np.asarray(out[0]).reshape(-1)[0])
        if "state" in self._input_names and len(out) > 1:
            self._state = np.asarray(out[1], dtype=np.float32)
        return prob

    def _is_speech(self, frame: np.ndarray) -> bool:
        samples = frame.astype(np.float32) / 32768.0
        self._buf = (np.concatenate([self._buf, samples])
                     if self._buf.size else samples)
        while self._buf.size >= self._WINDOW:
            chunk = self._buf[: self._WINDOW]
            self._buf = self._buf[self._WINDOW:]
            try:
                self._last_prob = self._run(chunk)
            except Exception as e:  # a transient run error must not drop the turn
                log.debug("silero run failed: %s", e)
        return self._last_prob >= self._threshold


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


def build_endpointer(engine: str, *, aggressiveness: int = 3,
                     silero_model: str | None = None,
                     silero_threshold: float = 0.5, **kw) -> _BaseEndpointer:
    """Build the configured endpointer. Silero falls back to webrtc if its
    model/runtime isn't available, so a missing model never breaks the mic."""
    if engine == "silero":
        try:
            return SileroEndpointer(model_path=silero_model,
                                    threshold=silero_threshold, **kw)
        except Exception as e:
            log.warning("silero VAD unavailable (%s) — falling back to webrtc", e)
    return WebrtcEndpointer(aggressiveness=aggressiveness, **kw)
