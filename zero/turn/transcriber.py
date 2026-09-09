"""Transcriber — everything that turns AUDIO into TEXT for a turn.

Second slice of the main.py surgery (docs/ARCHITECTURE_TRACKS_PLAN.md
M4.2a), moved verbatim from the Zero god-object:

  * the shared STT lock — speculative, interrupt-classify, afterthought
    and external-turn transcriptions all serialize against the final
    transcript through ONE lock, exactly as before;
  * the LIVE session lifecycle — streaming ASR opened per turn, torn
    down without leaking sockets, fed by a tee that holds back room tone
    until speech starts (bandwidth) without double-feeding the stateful
    VAD (the hop-buffer desync incident);
  * afterthoughts — gap remarks transcribed the moment they finish and
    merged into the turn later, with the 20 s expiry that keeps a stale
    remark from resurfacing in a turn that no longer exists;
  * the batch rescue path for when the live stream returns nothing.

The turn LOOP (who calls what, when) stays in main — this module owns
the how, not the choreography.
"""
from __future__ import annotations

import threading
import time

from zero.utils.logging import get_logger

log = get_logger("turn.stt")


class Transcriber:
    def __init__(self, cfg, engine):
        self._cfg = cfg
        self.engine = engine
        # Serialize speculative vs final STT — one recogniser, one queue.
        self.lock = threading.Lock()
        self._live = None              # live ASR session for the current turn
        self._afterthoughts: list = []  # (t, text) gap remarks awaiting merge

    # ── batch paths ─────────────────────────────────────────────────────────
    def transcribe(self, audio_f32, sr: int) -> str:
        """Locked batch transcription. Raises what the engine raises —
        call sites keep their own failure policies."""
        with self.lock:
            return (self.engine.transcribe(audio_f32, sr) or "").strip()

    def rescue(self, audio_f32, sr: int) -> str | None:
        """The second-opinion path for a failed live stream, when the
        engine offers one. None = no rescue engine configured."""
        fn = getattr(self.engine, "rescue_transcribe", None)
        if not callable(fn):
            return None
        with self.lock:
            return (fn(audio_f32, sr) or "").strip()

    # ── live session lifecycle ──────────────────────────────────────────────
    def open_live(self, sr: int):
        """A live ASR session for this turn, or None when streaming is off /
        the engine can't do it / it fails to open. Never raises — a failure
        here just means the old record-then-transcribe path."""
        if not self._cfg.get("stt.streaming", True):
            return None
        maker = getattr(self.engine, "live_session", None)
        if maker is None:  # unwrap FallbackSTT to reach the primary
            maker = getattr(getattr(self.engine, "_primary", None),
                            "live_session", None)
        if maker is None:
            return None
        try:
            self.close_live()   # never leak a previous turn's socket
            session = maker(sr)
            session.start()
            self._live = session
            return session
        except Exception as e:
            log.warning("live STT unavailable (%s) — using batch path", e)
            return None

    def close_live(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            try:
                live.close()
            except Exception as e:
                log.debug("live STT close failed: %s", e)

    def tee_to_live(self, frames, live):
        """Pass mic frames through to the endpointer while also feeding the
        live recogniser. push() never blocks or raises, so the capture loop's
        timing is unaffected.

        Room tone is held back rather than streamed: frames go into a short
        ring until speech actually starts, then the ring is flushed so the
        first word still has its lead-in. Waiting silently for someone to
        speak used to cost ~960 kbps to a machine across the internet."""
        import numpy as _np

        def _rms(f):
            return float(_np.sqrt(_np.mean(
                _np.asarray(f, dtype=_np.float32) ** 2)))

        # Well below the VAD's own start gate: this only decides when the wire
        # opens, so erring open costs a little bandwidth, while erring closed
        # would clip the start of a sentence.
        gate = max(40.0, self._cfg.get("vad.energy_threshold", 150) * 0.4)
        pre: list = []
        speaking = False
        pad = max(1, self._cfg.get("vad.speech_pad_ms", 200)
                  // max(1, self._cfg.get("audio.block_ms", 30)))
        for frame in frames:
            if speaking:
                live.push(frame)
            else:
                pre.append(frame)
                if len(pre) > pad + 1:
                    pre.pop(0)
                # STATELESS level check on purpose. Calling the endpointer's
                # VAD here double-fed it: capture() runs the same frames
                # through the same stateful TEN VAD, so each frame was
                # consumed twice and the model's hop buffer desynchronised —
                # which silently broke both utterance detection and barge-in.
                if _rms(frame) >= gate:
                    speaking = True
                    for f in pre:      # flush the lead-in, keep the first word
                        live.push(f)
                    pre = []
            yield frame

    # ── afterthoughts (gap remarks) ─────────────────────────────────────────
    def note_afterthought(self, frames) -> None:
        """Monitor thread: transcribe a finished gap remark right away, so
        the merge point (just before speaking) finds text, not raw audio."""
        import numpy as np

        try:
            audio = np.concatenate(frames).astype("float32") / 32768.0
            text = self.transcribe(
                audio, self._cfg.get("audio.sample_rate", 16000))
            if text:
                log.info("afterthought heard: %r", text)
                self._afterthoughts.append((time.monotonic(), text))
        except Exception as e:  # a lost afterthought must never break playback
            log.debug("afterthought stt failed: %s", e)

    def pop_afterthoughts(self) -> str:
        """Everything transcribed from gap remarks since the turn committed,
        joined — '' when there were none. One-shot, and EXPIRING: an
        afterthought older than ~20 s belongs to a turn that no longer
        exists. During the 2026-08-31 STT outage a stale remark survived
        several aborted turns and was merged, minutes later, into an empty
        transcript — ZERO answered words nobody had just said."""
        max_age = float(self._cfg.get("stt.afterthought_max_age_s", 20.0))
        now = time.monotonic()
        parts, self._afterthoughts = self._afterthoughts, []
        fresh, stale = [], 0
        for item in parts:
            t, text = item if isinstance(item, tuple) else (now, item)
            if text and now - t <= max_age:
                fresh.append(text)
            elif text:
                stale += 1
        if stale:
            log.info("dropped %d stale afterthought(s) (>%.0fs old)",
                     stale, max_age)
        return " ".join(fresh).strip()

    def clear_afterthoughts(self) -> None:
        self._afterthoughts = []
