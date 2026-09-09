"""Canned speech — fillers and recovery lines, pre-synthesized at startup.

First slice of the main.py surgery (docs/ARCHITECTURE_TRACKS_PLAN.md
M4.1): the self-contained "lines ZERO keeps in its pocket" subsystem,
extracted verbatim from the Zero god-object.

Two kinds of pocket lines:
  * FILLERS — "Good question, let me think." — played only if the real
    reply's audio hasn't arrived within the grace window, picked to fit
    what the user just said (question / short ack / neutral).
  * RECOVERY — "Give me one second." — spoken when a turn fails. Rendered
    once at startup and kept as WAVEFORMS, because when the exhibition
    wifi drops, TTS is gone too: anything synthesised on demand would
    also fail. These are already audio.

This module owns synthesis and selection; PLAYBACK stays with the
caller (it owns the speaker and the barge-in rules).
"""
from __future__ import annotations

import random

from zero.utils.logging import get_logger

log = get_logger("voicelines")

RECOVERY_LINES = {
    "retry": ["Give me one second.", "Hang on, let me try that again."],
    "lost": ["Sorry, I lost my train of thought there. Say that again?",
             "I didn't quite catch that — one more time?"],
    "slow": ["My connection is being slow right now — bear with me.",
             "Give me a moment, I'm having a slow moment."],
}

DEFAULT_FILLERS = {
    "question": ["Good question, let me think.",
                 "Hmm, let me think about that.",
                 "Let me think for a second."],
    "default": ["Okay, let me see.", "Right, one moment.", "Let's see."],
    "ack": ["Mm-hmm.", "Sure."],
}

_QUESTION_WORDS = {
    "what", "why", "how", "when", "who", "where", "which", "whose", "can",
    "could", "would", "do", "does", "did", "is", "are", "should", "tell",
    "explain", "describe",
}


class VoiceLines:
    def __init__(self, cfg, synthesize):
        """`synthesize`: str -> waveform (or raises/returns empty) — the
        live TTS at startup time. Never called again after presynth."""
        self._synth = synthesize
        self._filler_prob = cfg.get("conversation.filler_probability", 0.5)
        self._filler_sets = cfg.get("conversation.fillers", DEFAULT_FILLERS)
        self._fillers: dict[str, list] = {}
        self._recovery: dict[str, list] = {}

    # ── startup synthesis ───────────────────────────────────────────────────
    def presynth(self) -> None:
        self._fillers = self._presynth_fillers()
        self._recovery = self._presynth_recovery()

    def _presynth_fillers(self) -> dict:
        out: dict[str, list] = {}
        total = 0
        misses = 0  # consecutive empty synths — the TTS is cold/down/mute
        for category, phrases in self._filler_sets.items():
            audios = []
            for phrase in phrases:
                if misses >= 2:  # stop hammering a dead TTS at 30s/call
                    break
                try:
                    audio = self._synth(phrase)
                except Exception as e:  # never block startup on a filler
                    audio = None
                    log.debug("filler synth failed for %r: %s", phrase, e)
                if getattr(audio, "size", 0):
                    audios.append(audio)
                    total += 1
                    misses = 0
                else:
                    misses += 1
            out[category] = audios
        if misses >= 2:
            log.warning("filler pre-synth aborted — TTS not responding; "
                        "fillers off this session (the real reply voice is "
                        "unaffected)")
        log.info("pre-synthesized %d fillers across %d categories",
                 total, len(out))
        return out

    def _presynth_recovery(self) -> dict:
        out: dict[str, list] = {}
        for kind, lines in RECOVERY_LINES.items():
            clips = []
            for line in lines:
                try:
                    audio = self._synth(line)
                except Exception as e:
                    audio = None
                    log.debug("recovery synth failed for %r: %s", line, e)
                if getattr(audio, "size", 0):
                    clips.append(audio)
            out[kind] = clips
        total = sum(len(v) for v in out.values())
        if total:
            log.info("pre-synthesized %d recovery lines (offline-safe)",
                     total)
        else:
            log.warning("NO recovery lines cached — a failed turn will be "
                        "SILENT. Check the TTS service before going live.")
        return out

    # ── selection ───────────────────────────────────────────────────────────
    @staticmethod
    def filler_category(text: str) -> str:
        """The filler that FITS what the user just said, so it sounds
        aware: a question gets 'Good question…'; a one-word reply a quick
        'Mm-hmm.'; everything else a neutral 'Let's see.'"""
        t = (text or "").lower().strip()
        words = t.split()
        if t.endswith("?") or (words and words[0] in _QUESTION_WORDS):
            return "question"
        if len(words) <= 2:
            return "ack"
        return "default"

    def pick_filler(self, user_text: str):
        """One pre-synthesized filler matched to what the user said, or
        None (probabilistic — silence is a valid filler too)."""
        if random.random() > self._filler_prob:
            return None
        category = self.filler_category(user_text)
        audios = (self._fillers.get(category)
                  or self._fillers.get("default") or [])
        if not audios:
            return None
        log.debug("filler category: %s", category)
        return random.choice(audios)

    def clips(self, category: str) -> list:
        """All cached filler clips of one category (backchannel murmurs)."""
        return list(self._fillers.get(category) or [])

    def recovery_clip(self, kind: str = "retry"):
        """A cached recovery waveform, or None. Never synthesises."""
        clips = (self._recovery.get(kind)
                 or self._recovery.get("retry") or [])
        return random.choice(clips) if clips else None
