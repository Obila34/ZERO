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

# Moods that get the "smile voice": a smile shortens the vocal tract and
# brightens 2-4 kHz — audible over a phone with no video. A gentle high-shelf
# fakes the same physics on the synthesized waveform.
_SMILE_MOODS = frozenset(("excited",))


class _SmileShelf:
    """RBJ high-shelf biquad (+`db` above `f0`), direct-form-II-transposed with
    carried state so streamed chunks join without a seam. Pure numpy — this
    must add ~zero latency and no dependencies to the synthesis path."""

    def __init__(self, sample_rate: int, f0: float = 2800.0, db: float = 2.5):
        import math

        a = 10.0 ** (db / 40.0)
        w0 = 2.0 * math.pi * f0 / sample_rate
        cw, sw = math.cos(w0), math.sin(w0)
        alpha = sw / 2.0 * math.sqrt((a + 1 / a) * (1 / 0.9 - 1) + 2)
        b0 = a * ((a + 1) + (a - 1) * cw + 2 * math.sqrt(a) * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - 2 * math.sqrt(a) * alpha)
        a0 = (a + 1) - (a - 1) * cw + 2 * math.sqrt(a) * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - 2 * math.sqrt(a) * alpha
        self._b = np.array([b0, b1, b2], dtype=np.float64) / a0
        self._a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        self._zi = np.zeros(2, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        # scipy's C lfilter (already a dependency via mic resampling) — a pure
        # Python loop here would cost more than the whole TTS chunk on a Pi.
        from scipy.signal import lfilter

        y, self._zi = lfilter(self._b, self._a, x.astype(np.float64),
                              zi=self._zi)
        return np.clip(y, -1.0, 1.0).astype(np.float32)

_CUE_RE = re.compile(r"(\[[a-z]+\])")
_SENTENCE_RE = re.compile(r"(.+?(?:[.!?…]+|\n|$))", re.S)


# Clause boundaries inside a long sentence. Speech is chunked here as well as
# at full stops for two reasons that turned out to be the same problem:
#
#   * Playback cannot start until the FIRST chunk is complete. A model that
#     opens with a 40-word sentence made the listener wait for all of it —
#     `llm+tts` was measured at 1.8-2.5s, the largest remaining block.
#   * A queued barge-in yields "at the sentence end". With one enormous
#     sentence that meant an 11-second wait after someone had already started
#     talking, which reads as being ignored.
#
# Splitting at commas/dashes/colons fixes both: audio starts sooner, and there
# is a natural place to hand over much sooner. The minimum length stops it
# chopping "Yes, sure." into fragments that sound clipped.
_CLAUSE_RE = re.compile(r"(.+?(?:[,;:—–]|\s-\s)\s*)", re.S)
_CLAUSE_MIN_CHARS = 28
_LONG_SENTENCE_CHARS = 90


def split_sentences(text: str, clause_split: bool = True) -> list[str]:
    """Coarse sentence split for streaming TTS. Cheap and good enough for
    speech. Long sentences are further broken at clause boundaries so playback
    can start — and can hand over — without waiting for a full stop."""
    out = [m.group(1).strip() for m in _SENTENCE_RE.finditer(text)]
    out = [s for s in out if s]
    if not clause_split:
        return out
    split: list[str] = []
    for sentence in out:
        if len(sentence) <= _LONG_SENTENCE_CHARS:
            split.append(sentence)
            continue
        buf = ""
        end = 0                      # how much of the sentence is consumed
        for m in _CLAUSE_RE.finditer(sentence):
            buf += m.group(1)
            end = m.end()
            if len(buf.strip()) >= _CLAUSE_MIN_CHARS:
                split.append(buf.strip())
                buf = ""
        # Whatever follows the last clause marker was never matched — without
        # this the end of every long sentence would be silently dropped.
        tail = (buf + sentence[end:]).strip()
        if tail:
            if len(tail) < _CLAUSE_MIN_CHARS and split:
                split[-1] = f"{split[-1]} {tail}"   # don't leave a stub
            else:
                split.append(tail)
    return [s for s in split if s] or out


_SENT_END_TAIL_RE = re.compile(r"[.!?…\n]\s*$")


def split_stream(text: str, eager_first: bool = False) -> tuple[list[str], str]:
    """Incremental split for streaming TTS: (complete_sentences, remainder).

    Unlike split_sentences, the remainder keeps its ORIGINAL whitespace. The
    old producer pattern (`buffer = split_sentences(buffer)[-1]`) stripped the
    buffer's trailing space, so when the tokenizer put the joining space at the
    END of a chunk ("I'm " + "still"), it was silently lost — live replies came
    out as "I'mstill" / "Heythere" and were spoken that way.

    ``eager_first``: when nothing has been spoken yet, also emit a leading
    CLAUSE ("Well, that's a good question,") the moment one is long enough,
    instead of holding all audio until the first full stop — the first sound
    starts several LLM tokens sooner.
    """
    complete: list[str] = []
    consumed = 0
    for m in _SENTENCE_RE.finditer(text):
        seg = m.group(1)
        if m.end() >= len(text) and not _SENT_END_TAIL_RE.search(seg):
            break  # incomplete tail — stays in the buffer, spacing intact
        # Reuse the long-sentence clause splitting on each complete sentence.
        complete.extend(split_sentences(seg.strip()))
        consumed = m.end()
    remainder = text[consumed:]
    if eager_first and not complete:
        buf = ""
        end = 0
        for m in _CLAUSE_RE.finditer(remainder):
            buf += m.group(1)
            end = m.end()
            if len(buf.strip()) >= _CLAUSE_MIN_CHARS:
                complete.append(buf.strip())
                remainder = remainder[end:]
                break
    return [s for s in complete if s], remainder


def strip_asides(chunks):
    """Drop non-speech the model writes into a streamed reply, statefully
    across chunk and sentence boundaries:

    * (parentheses) — imitations of the sensory-note protocol, including a
      paren the model never closes;
    * *multi-word star spans* — stage directions ("*smiles slightly*").
      Single-word *emphasis* keeps its word (the engines strip the stars).

    Whatever survives here is what gets SPOKEN, so this is the last gate."""
    depth = 0
    star: list | None = None  # buffered content of an open *span*
    for chunk in chunks:
        out = []
        for ch in chunk:
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth:
                    depth -= 1
            elif depth:
                continue
            elif ch == "*":
                if star is None:
                    star = []
                else:
                    span = "".join(star)
                    if span and " " not in span.strip():
                        out.append(f"*{span}*")  # emphasis — keep for engines
                    star = None  # multi-word = stage direction — dropped
            elif star is not None:
                star.append(ch)
            else:
                out.append(ch)
        if out:
            yield "".join(out)
    if star:  # unclosed * at end of reply — flush as ordinary words
        yield "".join(star)


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

    def _smile(self) -> "_SmileShelf | None":
        """Fresh smile filter when the current mood calls for one. Fresh per
        reply/sentence so filter state can never leak across utterances; the
        try guards a missing scipy (then the voice just doesn't brighten)."""
        if self._mood not in _SMILE_MOODS:
            return None
        try:
            return _SmileShelf(self.sample_rate)
        except Exception as e:
            log.debug("smile shelf unavailable: %s", e)
            return None

    # -- public API ---------------------------------------------------------
    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        if self.engine_name in ("fish", "orpheus", "kyutai"):
            # These perform cues natively; the engine handles cue translation.
            audio = self.tts.synthesize(text)
        else:
            audio = self._synthesize_piper(text)
        shelf = self._smile()
        if shelf is not None and getattr(audio, "size", 0):
            audio = shelf.process(audio)
        return audio

    def synthesize_stream(self, text: str):
        """Yield audio chunks as they're produced (real streaming for Orpheus;
        the Piper path falls back to one chunk after splicing cue clips)."""
        text = text.strip()
        if not text:
            return
        shelf = self._smile()
        if self.engine_name in ("fish", "orpheus", "kyutai"):
            for chunk in self.tts.synthesize_stream(text):
                if shelf is not None and getattr(chunk, "size", 0):
                    chunk = shelf.process(chunk)
                yield chunk
        else:
            audio = self._synthesize_piper(text)
            if getattr(audio, "size", 0):
                yield shelf.process(audio) if shelf is not None else audio

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
