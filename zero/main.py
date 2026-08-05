"""ZERO entry point — the IDLE -> LISTENING -> THINKING -> SPEAKING loop.

Run on the Pi (after scripts/setup_pi.sh) with:
    python -m zero.main

The loop streams the LLM reply sentence-by-sentence and speaks each sentence as
soon as it's ready, so the first words come out while the rest is still being
generated — the main trick for keeping a fully-local pipeline feeling responsive.
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import queue
import random
import re
import sys
import threading
import time

from zero.audio.interrupt import InterruptKind, classify_interrupt
from zero.speculate import Speculation, worth_speculating
from zero.config import load_config
from zero.conversation import Conversation
from zero.events import EventBus
from zero.factory import (
    build_corpus, build_endpointer, build_guests, build_identity,
    build_learning_loop, build_llm, build_memory, build_perception,
    build_privacy, build_proactive, build_stt, build_tools,
    build_turn_detector, build_vision, build_voice, build_voiceid, build_wake,
)
from zero.privacy.guard import parse_forget_command
from zero.identity.service import parse_enroll_command, parse_enrollment
from zero.memory.preferences import (apply_rate_delta, parse_preference,
                                     parse_volume)
from zero.perception.affect import MoodTracker
from zero.tools.base import ToolContext
from zero.vision.learned import parse_object_teach
from zero.llm.persona import build_system_prompt
from zero.state import State, can_transition
from zero.tts.orchestrator import split_stream, strip_asides
from zero.utils.logging import get_logger, setup_logging

log = get_logger("main")


# Stop phrases that end the conversation and return to wake-word mode. Matched on
# word boundaries AND only in short utterances, so a sentence that merely mentions
# one ("tell me about the movie Goodbye Lenin") doesn't put ZERO to sleep.
STOP_PHRASES = (
    "goodbye", "good bye", "go to sleep", "stop listening", "stop talking",
    "that's all", "thats all", "see you later", "bye zero", "bye jarvis",
    "bye for now", "talk later", "shut down",
)
_STOP_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in STOP_PHRASES) + r")\b")
_STOP_MAX_WORDS = 5  # stop commands are short; longer sentences just mention the words


def is_stop_phrase(text: str) -> bool:
    t = text.lower().strip(" .!?,")
    if len(t.split()) > _STOP_MAX_WORDS:
        return False
    return bool(_STOP_RE.search(t))


# Words/phrases that signal the user is asking about what ZERO can SEE. On these
# we hand the multimodal LLM a few recent keyframes so it can actually look;
# every other turn just carries the cheap text detector hint. Kept deliberately
# tight, two ways: bare demonstratives ("this", "that", "there") appear in most
# ordinary sentences, and polysemous verbs ("see", "look", "watch", "picture")
# are mostly non-visual in speech ("I see", "looking forward to it", "picture
# this scenario") — those live in the PHRASES list in their genuinely visual
# forms instead.
_VISUAL_WORDS = {
    "color", "colour", "colors", "colours", "wearing", "holding", "camera",
    "recognize", "recognise",
}
_VISUAL_PHRASES = (
    "what is this", "what's this", "what is that", "what's that",
    "who is this", "who's this", "who is that", "who's that",
    "who am i", "what am i", "in front of you", "how many",
    "over there", "over here", "around you", "the room", "this room",
    "next to", "read this", "read that", "am i wearing", "in my hand",
    # The visual uses of the demoted polysemous words:
    "do you see", "can you see", "what do you see", "what can you see",
    "have you seen", "how do i look", "take a look", "look at",
    "look around", "looking at", "watch me", "watch this",
)

# Detected labels that are useless as visual signals: "person" is on screen
# almost always, and the word turns up constantly in abstract speech ("he's a
# nice person") — matching it would ship keyframes on ordinary chat.
_GENERIC_LABELS = {"person"}


def _mentions_visible_object(text_lower: str, labels) -> bool:
    """True if the utterance names an object the camera can see RIGHT NOW.

    'What color is the cup?' should be visual whenever a cup is actually in
    frame — no fixed word list can enumerate every object, but the live
    detections can. Matches the full label and its head noun ('cell phone' ->
    'phone'), plus a naive plural, on word boundaries.
    """
    for label in labels:
        label = str(label).lower().strip()
        if not label or label in _GENERIC_LABELS:
            continue
        candidates = {label, label.split()[-1]}
        for cand in candidates:
            pattern = r"\s+".join(re.escape(part) for part in cand.split())
            if re.search(rf"\b{pattern}(?:e?s)?\b", text_lower):
                return True
    return False


# Words that end a transcript when the thought ISN'T finished — conjunctions,
# prepositions, articles, fillers. Used by semantic endpointing: a pause after
# "...and" means keep listening, not "reply now". Deliberately excludes words
# that legitimately end sentences (pronouns, "it", "that").
_MID_THOUGHT_WORDS = frozenset((
    "and", "or", "but", "so", "because", "then", "if", "when", "while",
    "with", "without", "to", "of", "for", "from", "in", "on", "at", "by",
    "as", "than", "the", "a", "an", "my", "your", "our", "their", "his",
    "um", "uh", "like", "plus", "also", "though", "although", "unless",
    "until", "whether", "gonna", "wanna", "versus",
))


def ends_mid_thought(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # Whisper writes a literal trailing "..." when speech trails off — the
    # clearest "I wasn't finished" signal there is.
    if t.endswith((",", "-", "—", ":", "...", "…")):
        return True
    words = re.findall(r"[a-z']+", t)
    return bool(words) and words[-1] in _MID_THOUGHT_WORDS


def is_visual_question(text: str, visible_labels=()) -> bool:
    t = text.lower()
    words = {w.strip(" .!?,;:'\"") for w in t.split()}
    if words & _VISUAL_WORDS:
        return True
    if any(p in t for p in _VISUAL_PHRASES):
        return True
    return _mentions_visible_object(t, visible_labels)


class Zero:
    def __init__(self, config_path: str | None = None, text_mode: bool = False):
        self.cfg = load_config(config_path)
        self.text_mode = text_mode
        self._abort_if_already_running()
        sr = self.cfg.get("audio.sample_rate", 16000)

        # The LLM is all text mode needs. Audio capture, wake word, STT and the
        # voice are only built for the full voice pipeline — so text mode runs on
        # a box with no mic/Piper/whisper installed (just to test brain + memory).
        log.info("loading engines...")
        self.events = EventBus()
        self.memory = build_memory(self.cfg)  # long-term SQLite store (or None)
        self.corpus = build_corpus(self.cfg)  # day-to-day interactions -> training data
        base_llm = build_llm(self.cfg)
        self.llm, self.tool_registry, self.timers = build_tools(
            self.cfg, base_llm, self.memory, self.events,
            context_provider=self._tool_context,
        )
        if not text_mode:
            # Lazy import so text mode doesn't require sounddevice/numpy/portaudio.
            from zero.audio.capture import MicCapture
            from zero.audio.playback import Speaker

            self.mic = MicCapture(
                sample_rate=sr,
                block_ms=self.cfg.get("audio.block_ms", 30),
                device=self.cfg.get("audio.input_device"),
                gain=self.cfg.get("audio.input_gain", 1.0),
            )
            echo_ref = None
            if self.cfg.get("audio.aec.enabled", False):
                from zero.audio.aec import EchoReference, SpeexAEC

                try:
                    echo_ref = EchoReference(sr)
                    self.mic.aec = SpeexAEC(
                        echo_ref, frame_size=self.mic.block_size,
                        sample_rate=sr,
                        filter_ms=self.cfg.get("audio.aec.filter_ms", 200))
                except Exception as e:  # missing speexdsp — run without AEC
                    echo_ref = None
                    log.warning("AEC unavailable (pip install speexdsp): %s", e)
            # Jitter prebuffer sized to the ACTIVE engine. The Orpheus value
            # (300ms) was applied to every engine — Kyutai streams at 2.9x
            # realtime and never underruns, so those 300ms were pure added
            # latency on every reply's first sound.
            _tts_engine = self.cfg.get("tts.engine", "piper")
            _prebuffer = (self.cfg.get("tts.kyutai.prebuffer_ms", 100)
                          if _tts_engine == "kyutai"
                          else self.cfg.get("tts.orpheus.prebuffer_ms", 0))
            self.speaker = Speaker(
                device=self.cfg.get("audio.output_device"),
                echo_ref=echo_ref,
                prebuffer_ms=_prebuffer,
                output_gain=self.cfg.get("audio.output_gain", 1.0))
            from zero.audio.room import RoomSense

            self.room = RoomSense(
                block_ms=self.cfg.get("audio.block_ms", 30),
                quiet_ref=self.cfg.get("audio.room.quiet_rms", 120),
                loud_ref=self.cfg.get("audio.room.loud_rms", 900),
                max_boost=self.cfg.get("audio.room.max_boost", 1.6),
                min_boost=self.cfg.get("audio.room.min_boost", 0.75))
            self.wake = build_wake(self.cfg)
            self.endpointer = build_endpointer(self.cfg)
            # Audio-first end-of-turn detector (Smart Turn v3); None = use the
            # text-based ends_mid_thought heuristic. Read from audio at each pause.
            self.turn = build_turn_detector(self.cfg)
            self.stt = build_stt(self.cfg)
            self.voice = build_voice(self.cfg)
            # Eyes: always-on camera + detection (or None if vision disabled /
            # the camera stack isn't installed). Started in run().
            self.eyes = build_vision(self.cfg)
            # One line of truth about what ACTUALLY loaded. Every stage above
            # degrades silently by design (missing model -> fallback), and
            # three silent fallbacks once stacked into "the upgrade changed
            # nothing". This makes the real pipeline a 2-second log read.
            log.info(
                "engines live: vad=%s | turn=%s | speech-bargein=%s | wake=%s",
                type(self.endpointer).__name__,
                "smart-turn-v3" if self.turn is not None else "TEXT-HEURISTIC "
                "(smart-turn model missing?)",
                "on" if self.cfg.get("conversation.barge_in_on_speech", True)
                else "OFF (config)",
                type(self.wake).__name__)
        else:
            self.eyes = None
            self.room = None

        tool_block = (self.tool_registry.spec_block()
                      if self.tool_registry is not None else "")
        lang_block = ""
        if self.cfg.get("conversation.multilingual", True):
            lang_block = (
                "Language: if the person speaks Swahili, or mixes Swahili and "
                "English (Sheng-style), reply in the same language or mix, "
                "naturally. Never point out the switch — just flow with it.")
        # Only nudge toward the web when the tool actually exists this session,
        # so the model never reaches for a tool it doesn't have.
        web_block = ""
        if (self.cfg.get("tools.websearch.enabled", False)
                and self.tool_registry is not None
                and self.tool_registry.get("web_search") is not None):
            import datetime as _dt

            today = _dt.date.today().strftime("%A, %B %d, %Y")
            web_block = (
                f"Today's date is {today} — use it to resolve 'today', "
                "'this year', 'yesterday' and the like; don't ask the person "
                "which year they mean.\n"
                "Staying current: when you're unsure of a fact — especially "
                "anything current like news, sports scores, release dates or "
                "prices — use the web_search tool to check before answering, "
                "rather than guessing or telling the person to look it up "
                "themselves. Only say you don't know if the search comes back "
                "empty.")
        system_prompt = build_system_prompt(tool_block, lang_block, web_block)
        self.convo = Conversation(
            system_prompt=system_prompt,
            history_turns=self.cfg.get("llm.history_turns", 3),
            trim_at_turns=self.cfg.get("llm.history_trim_at", 8),
        )
        self.state = State.IDLE
        self._interrupt = False   # HARD stop: cut playback mid-word (correction/wake)
        self._soft_stop = False   # POLITE stop: finish the sentence, then yield
        self._bargein_frames = None   # interrupting audio without a transcript
        self._queued_turn = None      # classified interruption: {text, frames, kind}
        self._interrupt_note = None   # one-shot note telling the LLM HOW it was cut off
        self._was_interrupted = False  # tell the LLM it got cut off, once
        self._last_backchannel = 0.0   # cooldown clock for "mm-hmm" while user speaks
        self._t_utterance_end = 0.0    # end-of-speech marker for the latency budget
        self._bg_monitor = None        # live duplex monitor (stop_event, thread)
        self._speculation = None       # in-flight bet on an unfinished sentence
        self._live_stt = None          # live ASR session for the current turn
        self._llm_unreachable = False  # last turn failed to reach the model
        self._degenerate = False       # last reply collapsed into repetition
        # Voice level ASKED FOR ("talk quietly", "speak up"). Persists across
        # turns and multiplies with the room's own Lombard gain, so "quietly"
        # still means quietly in a loud hall — just not inaudibly.
        self._voice_level = 1.0
        self._stage_marks: list = []   # per-turn latency breakdown
        self._stage_t = None
        self._afterthoughts: list = [] # transcribed gap remarks awaiting merge
        self._overheard: list = []     # backchannels heard mid-reply -> next context
        # Speculative prefill needs the RAW engine (the tool router doesn't
        # proxy it) — None simply means the old first-token latency.
        self._llm_prefill = (getattr(base_llm, "prefill", None)
                             if self.cfg.get("llm.speculative_prefill", True)
                             else None)
        self._mood = MoodTracker()     # cross-turn emotional state
        self._stt_lock = threading.Lock()  # serialize speculative vs final STT
        self._memory_thread: threading.Thread | None = None  # background fact save
        self._summary_thread: threading.Thread | None = None  # rolling compaction
        self._face_name = None         # who the camera recognises (log/perception only)
        self._last_id_key = None       # last identity note attached (dedup across turns)
        self._turn_durable_pid = None  # speaker to CREDIT durable memory this turn (or None)
        self._turn_speaker = None      # who spoke, for the corpus (real pid / guest / None)
        self._turn_voice_emb = None    # unfamiliar voiceprint awaiting the quality gate
        self._session_log: list[tuple] = []  # (durable_pid, role, text) for per-person save
        self._corpus_log: list[tuple] = []    # (speaker, role, text) for training data
        self._welcomed: set[int] = set()  # people greeted "welcome back" this session

        # External control (the AF1 fusion surface — zero/control.py). One lock
        # serializes external turns against each other; the busy-wait in
        # external_turn_text keeps them off the native loop's think/speak phase.
        self._ext_lock = threading.Lock()
        self._last_ext: dict = {}   # last external turn, for /zero/status
        self._control = None        # ControlServer when control.enabled

        # Voice-only extras (need the voice/mic): owner verification + spoken fillers.
        self._filler_prob = self.cfg.get("conversation.filler_probability", 0.5)
        self._filler_grace_s = self.cfg.get("conversation.filler_grace_ms", 600) / 1000.0
        self._fillers = {}
        self._person = None  # IdentityResult of the current speaker (or None)
        if not text_mode:
            self.voiceid, self._voiceprint = build_voiceid(self.cfg)
            if self.voiceid is not None:
                log.info("voice ID active — only the enrolled voice will be answered")
            self.identity = build_identity(self.cfg)
            # Cluster unfamiliar voices into provisional guests so different
            # strangers (and their training data) stay separate.
            self.guests = (build_guests(self.cfg)
                           if self.identity is not None else None)
            self.privacy, self.indicator = build_privacy(self.cfg)
            self.affect, self.speaker_tracker = build_perception(
                self.cfg, self.identity)
            self.proactive, self.curiosity, self.policy = build_proactive(
                self.cfg, events=self.events, eyes=self.eyes,
                identity=self.identity, memory=self.memory,
                is_idle=lambda: self.state == State.IDLE,
            )
            self._fillers = self._presynth_fillers()
            self._recovery = self._presynth_recovery()
        else:
            self.voiceid, self._voiceprint = None, None
            self._recovery = {}
            self.identity = None
            self.guests = None
            self.privacy, self.indicator = None, None
            self.affect, self.speaker_tracker = None, None
            self.proactive, self.curiosity, self.policy = None, None, None
        # Experience spine + reward tagging (Phase 3): every turn becomes a
        # reward-tagged episode; proactive outcomes feed the policy's adaptive
        # cooldowns. Works in text mode too (policy is None there — fine).
        self.episodes, self.reward = build_learning_loop(self.cfg,
                                                         policy=self.policy)

    def _self_state_notes(self) -> list[str]:
        """One-time notes when a pipeline stage degrades to (or recovers from)
        its local fallback — so ZERO can say 'I'm on my backup voice' instead
        of pretending nothing changed."""
        notes: list[str] = []
        if not hasattr(self, "_degraded_flags"):
            self._degraded_flags = {"tts": False, "stt": False}
        checks = (
            ("tts", bool(getattr(self.voice, "degraded", False)),
             "(Your main voice is down — you're speaking through your local "
             "backup voice right now. If it comes up, be upfront about it.)"),
            ("stt", bool(getattr(self.stt, "degraded", False)),
             "(Your fast hearing is down — you're transcribing locally, which "
             "is slower. If asked why you're slow, that's why.)"),
        )
        for key, now_degraded, note in checks:
            if now_degraded and not self._degraded_flags[key]:
                notes.append(note)
            self._degraded_flags[key] = now_degraded
        return notes

    def _tool_context(self) -> ToolContext:
        """Fresh, narrow context for a tool run — who's speaking now, plus the
        memory store and event bus."""
        person = self._person
        return ToolContext(
            memory=self.memory, events=self.events,
            person_id=person.person_id if person is not None else None,
            person_name=person.name if person is not None else None,
        )

    def _drain_events(self) -> bool:
        """Speak any queued announcements (timers, reminders, proactive nudges).
        Called at safe moments only: idle wake-wait and turn boundaries, so an
        announcement never talks over a reply. Returns True when an event asked
        to OPEN A CONVERSATION (a greeting or a curiosity question expects a
        reply — ZERO should listen, not go back to sleep)."""
        if self.text_mode:
            return False
        open_conversation = False
        for event in self.events.drain():
            log.info("announcing %s: %s", event.kind, event.text)
            prev = self.state
            self._to(State.SPEAKING)
            try:
                self._speak_one(event.text)
                if (self.reward is not None
                        and event.kind in ("greet", "curiosity", "remark")):
                    # Await the human's reaction: their next words (or their
                    # silence) score this proactive kind's bandit outcome.
                    self.reward.on_proactive(event.kind, event.text,
                                             person_id=event.person_id)
                if event.meta.get("open_conversation"):
                    open_conversation = True
                    # The opener is real dialogue — _converse() folds it into
                    # the fresh history so the LLM knows it said it.
                    self._pending_opener = event.text
            except Exception as e:  # an announcement must never kill the loop
                log.warning("announcement failed: %s", e)
            self._to(prev if prev != State.SPEAKING else State.IDLE)
        return open_conversation

    # Spoken when a turn produces nothing — a dropped network, an unreachable
    # model, a reply cut for degeneracy. Synthesised at startup while the
    # network is known good and held in RAM, so they still play when every
    # remote service is down. Silence in front of an audience reads as "it's
    # broken"; a short human line reads as "it's thinking".
    _RECOVERY_LINES = {
        "retry": ["Give me one second.", "Hang on, let me try that again."],
        "lost": ["Sorry, I lost my train of thought there. Say that again?",
                 "I didn't quite catch that — one more time?"],
        "slow": ["My connection is being slow right now — bear with me.",
                 "Give me a moment, I'm having a slow moment."],
    }

    _DEFAULT_FILLERS = {
        "question": ["Good question, let me think.", "Hmm, let me think about that.",
                     "Let me think for a second."],
        "default": ["Okay, let me see.", "Right, one moment.", "Let's see."],
        "ack": ["Mm-hmm.", "Sure."],
    }

    def _presynth_fillers(self) -> dict:
        sets = self.cfg.get("conversation.fillers", self._DEFAULT_FILLERS)
        out: dict[str, list] = {}
        total = 0
        misses = 0  # consecutive empty synths — the TTS is cold/down/mute
        for category, phrases in sets.items():
            audios = []
            for phrase in phrases:
                if misses >= 2:  # stop hammering a dead TTS at 30s/call
                    break
                try:
                    audio = self.voice.synthesize(phrase)
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
            log.warning("filler pre-synth aborted — TTS not responding; fillers "
                        "off this session (the real reply voice is unaffected)")
        log.info("pre-synthesized %d fillers across %d categories", total, len(out))
        return out

    # -- state transition helper -------------------------------------------
    def _to(self, dst: State) -> None:
        if dst == self.state:
            return  # already there (e.g. SPEAKING filler -> SPEAKING reply)
        if not can_transition(self.state, dst):
            log.warning("illegal transition %s -> %s", self.state, dst)
        log.debug("state: %s -> %s", self.state, dst)
        self.state = dst
        if self.indicator is not None:   # the visible "I can hear you" signal
            self.indicator.set_state(dst.name.lower())

    def _start_conversation(self) -> None:
        """Fresh history + load long-term memory (injected once, cache-friendly)."""
        self.convo.reset()
        if self.memory is not None:
            self.convo.set_memory(self.memory.as_block())

    def _warmup_messages(self) -> list:
        """The exact prefix the first conversation will send (system prompt +
        memory block), so warmup pre-fills the prompt cache and the first turn
        skips the multi-second prefill."""
        self._start_conversation()
        return [*self.convo.messages(), {"role": "user", "content": "hi"}]

    # -- text mode ----------------------------------------------------------
    def run_text(self) -> None:
        """Type-to-chat: tests the brain (LLM + memory + conversation) with no mic,
        Piper or Whisper needed. Useful for validating the GPU LLM offload."""
        self._start_control()  # text-mode turns still serve /zero/turn_text
        warmup = getattr(self.llm, "warmup", None)
        if callable(warmup):
            warmup(self._warmup_messages())
        if self._control is not None:
            self._control.ready = True
        print("\nZERO text mode — type to chat. Say 'goodbye' to reset, Ctrl-C to quit.\n")
        self._start_conversation()
        try:
            while True:
                try:
                    text = input("you> ").strip()
                except EOFError:
                    break
                if not text:
                    continue
                if self._is_stop(text):
                    print("zero> Okay, talk to you later.\n")
                    self._end_conversation()       # extract facts to memory
                    self._start_conversation()      # fresh chat, reload memory
                    continue
                self._maybe_remember(text)
                self.convo.add_user(text)
                # Text mode has no identity — turns are anonymous, but they must
                # still reach the session/corpus logs or nothing gets persisted.
                self._session_log.append((None, "user", text))
                self._corpus_log.append((None, "user", text))
                self._t_reply_start = time.monotonic()
                parts, first = [], True
                for chunk in self.llm.stream(self.convo.messages()):
                    if first and chunk.strip():
                        print(f"  [first token: {time.monotonic()-self._t_reply_start:.2f}s]")
                        first = False
                    parts.append(chunk)
                reply = "".join(parts).strip()
                if reply:
                    print(f"zero> {reply}\n")
                    self.convo.add_assistant(reply)
                    self._session_log.append((None, "assistant", reply))
                    self._corpus_log.append((None, "assistant", reply))
                else:
                    print("zero> (no reply — see log; check the tunnel/model)\n")
                self._maybe_compact()  # rolling summary keeps context bounded
        except KeyboardInterrupt:
            print()
        self._end_conversation()
        self._join_memory_thread()

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        # AF1 fusion surface — up before warmup so /health answers immediately;
        # `ready` flips true once the model is pinned.
        self._start_control()
        # Pin the LLM in RAM now — with the real prefix, so the first reply
        # pays neither the cold load nor the persona+memory prefill.
        warmup = getattr(self.llm, "warmup", None)
        if callable(warmup):
            warmup(self._warmup_messages())
        if self._control is not None:
            self._control.ready = True
        self.mic.start()
        # Open the eyes BEFORE the wake loop so perception is already running when
        # the user speaks — the scene is pre-computed, never on the critical path.
        if self.eyes is not None:
            try:
                self.eyes.start()
            except Exception as e:  # a missing camera must not stop voice working
                log.warning("could not start eyes — running voice-only: %s", e)
                self.eyes = None
        # Surprise gate (Phase 3): scores world events by prediction error —
        # the unexpected becomes episodes and wakes the narrator. Built here
        # (not __init__) because it needs the eyes to have survived startup.
        self._surprise = None
        if self.eyes is not None and self.episodes is not None:
            try:
                from zero.factory import build_surprise_gate

                self._surprise = build_surprise_gate(
                    self.cfg, eyes=self.eyes, episodes=self.episodes)
                if self._surprise is not None:
                    self._surprise.start()
            except Exception as e:  # learning must never block startup
                log.warning("surprise gate unavailable: %s", e)
        if self.proactive is not None:
            self.proactive.start()  # presence greetings + curiosity + idle learning
        log.info("ZERO ready. Say the wake word to start talking. (Ctrl-C to quit)")
        try:
            while True:
                self._wait_for_wake()   # IDLE: wake word OR a proactive opener
                self._converse()        # free-flowing multi-turn until sleep/stop
        except KeyboardInterrupt:
            print()
            log.info("shutting down")
        finally:
            if self.proactive is not None:
                self.proactive.stop()
            self.mic.stop()
            if getattr(self, "_surprise", None) is not None:
                self._surprise.stop()
            if self.eyes is not None:
                self.eyes.stop()
            if self.indicator is not None:
                self.indicator.close()
            self._join_memory_thread()

    def _wait_for_wake(self) -> None:
        self._to(State.IDLE)
        self.wake.reset()
        # Defensive: the stop-phrase path in _converse() returns with the mic
        # paused, so without this the callback would drop every frame and
        # frames() would block forever — leaving ZERO deaf to the wake word.
        self.mic.resume()
        self.mic.drain()
        for frame in self.mic.frames():
            if self.events.peek_pending():
                opened = self._drain_events()  # timers/greetings fire in IDLE
                if opened:
                    # A proactive opener (greeting / curiosity question) expects
                    # an answer — enter the conversation without a wake word.
                    log.info("proactive opener — conversation open")
                    return
                self._to(State.IDLE)
                self.wake.reset()
                self.mic.resume()
                self.mic.drain()               # drop our own announcement audio
            _room = getattr(self, "room", None)
            if _room is not None:
                _room.observe(frame)   # idle mic = the room itself
                _room.maybe_log(time.monotonic())
            if self.wake.process(frame):
                log.info("wake word! let's talk.")
                return

    # -- conversation -------------------------------------------------------
    def _is_stop(self, text: str) -> bool:
        return is_stop_phrase(text)

    def _converse(self) -> None:
        """After a single wake, stay in a flowing conversation: listen -> reply,
        repeatedly, with NO wake word between turns. History is kept across turns.
        Ends on a stop phrase or `sleep_timeout` of silence, then returns to IDLE.
        """
        self.convo.reset()
        # Load durable facts from past sessions and inject them ONCE (keeps the
        # prefix stable, so the cache stays warm through the conversation).
        if self.memory is not None:
            self.convo.set_memory(self.memory.as_block())
        # Prefill visibility: the system prefix is what every cold prefill
        # pays for. Logged so prompt growth is a measured fact, not a guess.
        _sys = self.convo.messages()[0]["content"]
        log.info("system prefix: %d chars (~%d tokens)", len(_sys), len(_sys) // 4)
        # A proactive opener ("Hey David, welcome back.") started this
        # conversation — seed it as the first assistant turn.
        opener = getattr(self, "_pending_opener", None)
        if opener:
            self.convo.add_assistant(opener)
            self._pending_opener = None
        if self.speaker_tracker is not None:
            self.speaker_tracker.reset()   # a new conversation, a clean slate
        self._mood.reset()
        self._reset_turn_state()  # stale interrupts/ducks must not leak across chats
        self._face_name = None
        self._last_id_key = None       # fresh conversation: re-introduce who's here
        self._session_log = []         # per-turn (durable_pid, role, text) for memory
        self._corpus_log = []          # per-turn (speaker, role, text) for training data
        self._welcomed = set()         # welcome-back fires once per person
        sr = self.cfg.get("audio.sample_rate", 16000)
        idle_s = self.cfg.get("conversation.sleep_timeout_ms", 30000) / 1000.0
        log.info("conversation open — just talk (say 'goodbye' to stop)")

        while True:
            self._stop_monitor()  # whatever path got us here, one mic consumer
            self._stage_marks = []   # a turn that never replied must not leak
            self._stage_t = None     # its stages into the next turn's report
            self._drain_events()  # timers/reminders land at turn boundaries
            self._maybe_compact()  # fold trimmed turns into a rolling summary

            # A classified interruption from the last reply? Its words were
            # ALREADY transcribed while ZERO wound down its sentence — skip
            # capture AND STT: the response to a queued/correcting interruption
            # starts with zero listening latency.
            import numpy as np

            queued, self._queued_turn = self._queued_turn, None
            pre_text = (queued or {}).get("text")
            spec: dict = {}
            if pre_text:
                frames = queued.get("frames") or []
                utterance = (np.concatenate(frames).astype("float32") / 32768.0
                             if frames else np.zeros(1, dtype="float32"))
                self.mic.pause()
                self._to(State.THINKING)
                self._t_utterance_end = time.monotonic()
            else:
                self._to(State.LISTENING)
                self.mic.resume()  # re-open the mic for the user's turn
                # On an untranscribed barge-in the interrupting audio was still
                # captured — feed it into this capture so the user never has to
                # repeat themselves.
                stash, self._bargein_frames = self._bargein_frames, None
                if stash:
                    frames_src = itertools.chain(stash, self.mic.frames())
                else:
                    self.mic.drain()  # drop audio captured while speaking/thinking
                    frames_src = self.mic.frames()
                # Speculative STT: the moment the user PAUSES (well before the
                # endpoint confirms), start transcribing the audio so far — the
                # STT round trip overlaps the silence wait. Skipped when a gate
                # (voiceid / strict privacy) must run before any transcription.
                # Speculation only pays off when STT is the FAST remote. While
                # it's degraded to the slow local CPU engine, speculating just
                # runs that ~5s job twice — so skip it and transcribe once.
                allow_spec = (self.voiceid is None
                              and not getattr(self.stt, "degraded", False)
                              and not (self.privacy is not None
                                       and getattr(self.privacy, "mode", "")
                                       == "strict"))

                # Live streaming ASR: the mic feeds the recogniser WHILE the
                # person talks, so the transcript is essentially written by the
                # time they stop — instead of a ~0.5-1.2s transcription that
                # only starts once they've finished.
                live = self._open_live_stt(sr)
                if live is not None:
                    frames_src = self._tee_to_live(frames_src, live)

                def _on_pause(audio_i16):
                    # Always record the pause audio + the turn-model verdict —
                    # Smart Turn must keep working even when speculative STT is
                    # gated off (the old coupling silently disabled it).
                    spec["audio"] = audio_i16
                    p = None
                    if self.turn is not None:
                        p = self.turn.probability(
                            audio_i16.astype("float32") / 32768.0)
                    spec["turn_p"] = p
                    self._maybe_backchannel(audio_i16, p)
                    if live is not None:
                        # The live transcript is already here — no STT round
                        # trip to wait on. Use it for the endpoint decision AND
                        # to start guessing at a reply before they finish.
                        spec["live_text"] = lt = (live.text() or "").strip()
                        self._maybe_speculate(lt, p)
                        # Speculative LLM PREFILL on the live path. This block
                        # used to live only in the batch-STT branch below,
                        # which the live path returns before — so with Kyutai
                        # streaming (the deployed config) the KV cache was
                        # never warmed and every turn paid full prefill on the
                        # real request. A full speculative reply supersedes it
                        # (its own stream prefills); otherwise one throwaway
                        # token here makes commit-time first-token near-instant.
                        if (lt and self._llm_prefill is not None
                                and self._speculation is None
                                and spec.get("prefilled") != lt):
                            spec["prefilled"] = lt

                            def _prefill(text=lt):
                                try:
                                    self._llm_prefill(
                                        [*self.convo.messages(),
                                         {"role": "user", "content": text}])
                                except Exception as e:
                                    log.debug("speculative prefill failed "
                                              "(harmless): %s", e)
                            threading.Thread(target=_prefill,
                                             name="llm-prefill",
                                             daemon=True).start()
                        return
                    if not allow_spec:
                        return
                    res: dict = {}

                    def run():
                        with self._stt_lock:
                            try:
                                res["text"] = self.stt.transcribe(
                                    audio_i16.astype("float32") / 32768.0, sr)
                            except Exception as e:
                                log.debug("speculative stt failed: %s", e)
                        # Speculative LLM prefill: warm the KV cache with this
                        # transcript while the endpoint is still waiting out
                        # the silence — by commit time the model has usually
                        # already read the prompt and first-token is nearly
                        # instant. Wrong speculation costs one throwaway token.
                        text = (res.get("text") or "").strip()
                        if (text and self._llm_prefill is not None
                                and spec.get("prefilled") != text):
                            spec["prefilled"] = text
                            try:
                                self._llm_prefill(
                                    [*self.convo.messages(),
                                     {"role": "user", "content": text}])
                            except Exception as e:
                                log.debug("speculative prefill failed: %s", e)
                    spec["res"] = res
                    spec["thread"] = t = threading.Thread(
                        target=run, name="stt-spec", daemon=True)
                    t.start()

                def _hold() -> bool | None:
                    # Audio-first end-of-turn (Smart Turn v3): judge whether the
                    # turn is finished straight from the waveform at THIS pause —
                    # prosody, intonation, rhythm — no transcript needed. The
                    # verdict was computed once at the pause hook; reuse it.
                    p = spec.get("turn_p")
                    if p is not None:
                        done = p >= self.turn.threshold
                        # Cross-check the audio verdict against the live words:
                        # prosody can read "finished" on a trailing "...and",
                        # and the transcript is free here (already streamed).
                        lt = spec.get("live_text")
                        if done and lt and ends_mid_thought(lt):
                            return True   # hold — the words say otherwise
                        return not done
                    # Text fallback, tri-state: the speculative transcript for
                    # THIS pause says finished (False), mid-thought (True) — or
                    # it's STILL IN FLIGHT (None), in which case the endpointer
                    # waits a bounded beat instead of racing the STT round trip.
                    res = spec.get("res") or {}
                    text = res.get("text")
                    if text is None:
                        t = spec.get("thread")
                        if t is not None and t.is_alive():
                            return None  # transcript pending — worth a short wait
                        return False     # no speculation ran; commit normally
                    return ends_mid_thought(text)

                early_thr = self.cfg.get("vad.early_commit_threshold", 0.85)

                def _early(audio_i16) -> bool:
                    # Predictive endpointing: when the turn model is CONFIDENT
                    # the thought is complete at the ~180 ms pause mark, commit
                    # now — the human ~200 ms turn gap — instead of waiting out
                    # the full silence window.
                    if not self.cfg.get("vad.early_commit", True):
                        return False
                    p = spec.get("turn_p")
                    return p is not None and p >= early_thr

                use_hold = (self.turn is not None
                            or (allow_spec
                                and self.cfg.get("vad.semantic_hold", True)))
                utterance = self.endpointer.capture(
                    frames_src, idle_timeout_s=idle_s,
                    on_speech_pause=_on_pause,
                    should_hold=_hold if use_hold else None,
                    early_commit=_early if self.turn is not None else None)
                self._t_utterance_end = time.monotonic()

                if utterance is None or getattr(utterance, "size", 0) == 0:
                    log.info("Sleeping… (no speech for %.0fs — say the wake "
                             "word to talk again)", idle_s)
                    self._end_conversation()
                    return

                self.mic.pause()
                self._to(State.THINKING)
            self._stage("capture->pause")

            # Duplex from the moment the turn commits: the monitor runs through
            # THINKING as well as SPEAKING, so the old deaf window (identity +
            # STT + prefill, ~0.5-1.5s) is gone. Gap remarks become
            # afterthoughts (merged below, before a word is spoken); speech
            # continuing into playback becomes a carry barge-in. When speech
            # barge-in is config-disabled, _start_bargein leaves the mic paused
            # and only the wake word interrupts — the old behavior.
            self._bg_monitor = self._start_bargein()
            self._stage("arm-bargein")

            # "Only my voice": skip anything that isn't the enrolled owner — before
            # STT, so we don't even transcribe other people / background voices.
            if self.voiceid is not None:
                score, is_owner = self.voiceid.verify(self._voiceprint, utterance)
                if not is_owner:
                    log.info("ignored: not the owner (voice score %.2f)", score)
                    continue
                log.debug("owner verified (voice score %.2f)", score)

            # Transcribe IN PARALLEL with identity/diarization below — they only
            # need the audio, not the text. Reuse the speculative transcription
            # when its pause became the actual endpoint (the audio it saw is a
            # verbatim prefix of the final utterance and only silence follows).
            stt_result: dict = {}
            stt_thread = None
            stt_rescue = False   # live path already failed on this audio
            spec_a = spec.get("audio")
            # Tail bound: more than ~2x the silence window after the spec pause
            # means speech resumed / max-cap fired — the spec text is a prefix.
            slack = int(sr * (2 * self.cfg.get("vad.silence_ms", 450)
                              + self.cfg.get("vad.speech_pad_ms", 200) + 400) / 1000)
            if pre_text:
                stt_result = {"text": pre_text}  # transcribed at interrupt time
            elif live is not None:
                # Streamed while they spoke — only the model's lookahead tail
                # is left to flush, so this returns almost immediately.
                # MUST run before the session is closed: closing first kills
                # the sender, so the end-of-turn marker never goes out, the
                # tail never flushes (the last word of every turn is lost) and
                # finalize() returns instantly on a dead session — which is
                # what "settled in 0 ms" on every turn meant.
                _t_fin = time.monotonic()
                stt_result = {"text": live.finalize(
                    timeout=self.cfg.get("stt.finalize_timeout", 3.0))}
                self._stage("transcript")
                self._close_live_stt()   # tail is in — now release the socket
                if not stt_result["text"]:
                    # Socket died mid-turn OR the model finalized with zero
                    # words on real speech (Kyutai's venue failure mode).
                    # Either way the live path has already failed on THIS
                    # audio — route to the batch rescue below, which goes
                    # fallback-first instead of paying the primary again.
                    if live.failed is not None:
                        log.warning("live STT failed (%s) — batch rescue",
                                    live.failed)
                    stt_result = {}
                    stt_rescue = True
            elif (spec_a is not None and spec.get("thread") is not None
                    and 0 <= utterance.size - spec_a.size <= slack
                    and bool((utterance[:spec_a.size]
                              == spec_a.astype("float32") / 32768.0).all())):
                stt_thread, stt_result = spec["thread"], spec["res"]
                log.debug("speculative STT reused (%.1fs head start)",
                          (utterance.size - spec_a.size) / sr)
            elif not (self.privacy is not None
                      and getattr(self.privacy, "mode", "") == "strict"):
                def _stt(u=utterance):
                    with self._stt_lock:
                        try:
                            stt_result["text"] = self.stt.transcribe(u, sr)
                        except Exception as e:
                            log.warning("stt failed: %s", e)
                stt_thread = threading.Thread(target=_stt, name="stt", daemon=True)
                stt_thread.start()

            # Who is speaking? Face (current frame) + voice (this utterance),
            # fused. Local + fast; never on the critical path of a failure.
            frame = self.eyes.current_frame() if self.eyes is not None else None
            self._turn_notes = []
            if self._was_interrupted:
                self._was_interrupted = False
                self._turn_notes.append(self._interrupt_note or (
                    "(You were interrupted mid-sentence a moment ago — don't "
                    "restart the old answer; just respond to what they say now.)"))
                self._interrupt_note = None
            heard, self._overheard = self._overheard[-3:], []
            if heard:
                murmurs = "; ".join(f"'{h}'" for h in heard)
                self._turn_notes.append(
                    f"(While you were speaking they murmured: {murmurs} — "
                    "active listening, not an interruption. Let it shape you: "
                    "engagement means you can go a level deeper; don't comment "
                    "on the murmur itself.)")
            self._turn_durable_pid = None  # only a CONFIDENT voice credits memory
            self._turn_speaker = None      # who to tag this turn's training data
            # Identity runs under a HARD TIME BUDGET. Voice embedding +
            # diarization + registry lookup measured ~0.5-1.0s and sat
            # squarely on the reply path — a third of the latency budget
            # spent deciding who is talking, which the ANSWER does not need.
            # It runs on a thread; if it misses the budget the turn proceeds
            # with the identity already known (it rarely changes mid-chat)
            # and the result lands for the next turn. Correctness is
            # unchanged, the wait is not.
            def _identity_work():
                if self.identity is not None:
                    # Sessions are owned by VOICE — the speaker's voice decides whose
                    # memories this turn belongs to; the face is perception only.
                    # (voice_only: false restores the legacy face+voice fusion.)
                    if self.cfg.get("identity.session.voice_only", True):
                        ident, face_name = self.identity.identify_speaker(
                            audio=utterance, frame_rgb=frame)
                    else:
                        ident = self.identity.identify(audio=utterance, frame_rgb=frame)
                        face_name = (ident.name if ident.is_known
                                     and "face" in ident.via else None)
                    self._person = ident if ident.is_known else None
                    self._face_name = face_name
                    # One voiceprint for this utterance, reused by diarization AND
                    # guest clustering (embedding is not free).
                    voice_emb = self.identity.voice_embedding(utterance)
                    if ident.is_known:
                        self._turn_speaker = ident.person_id
                        # Persist this turn under the speaker ONLY when the voice is
                        # confident enough: a borderline match still talks (live), but
                        # must not write into someone else's permanent memory. This is
                        # what keeps a multi-speaker session from cross-contaminating.
                        write_min = self.cfg.get("identity.session.write_min_score", 0.55)
                        if ident.score >= write_min:
                            self._turn_durable_pid = ident.person_id
                        welcome = self._welcome_back_note(ident)
                        if welcome:
                            self._turn_notes.append(welcome)
                    elif self.guests is not None:
                        # An unfamiliar voice: remember the voiceprint, but DON'T
                        # mint a guest yet — the transcript isn't in, and noise /
                        # STT hallucinations were creating phantom guests. The
                        # quality-gated assignment happens after the text check.
                        self._turn_voice_emb = voice_emb
                    # Diarization: notice when a DIFFERENT person takes over.
                    if self.speaker_tracker is not None:
                        change_note = self.speaker_tracker.update(
                            person_id=ident.person_id if ident.is_known else None,
                            name=ident.name if ident.is_known else None,
                            voice_emb=voice_emb,
                        )
                        if change_note:
                            self._turn_notes.append(change_note)


            _id_t = threading.Thread(target=_identity_work, name="identity",
                                     daemon=True)
            _id_t.start()
            _id_budget = self.cfg.get("identity.budget_ms", 150) / 1000.0
            _id_t.join(timeout=_id_budget)
            self._stage("identity")
            if _id_t.is_alive():
                log.debug("identity over budget (%dms) — replying now, the "
                          "result lands for the next turn", int(_id_budget * 1000))

            # Bystander gate BEFORE transcription: in strict mode an unknown
            # voice isn't even transcribed — nothing to act on, nothing kept.
            decision = None
            if self.privacy is not None and self.identity is not None:
                decision = self.privacy.decide(self._person)
                if not decision.respond:
                    continue
            self._memory_allowed = decision.store_memory if decision else True

            if stt_thread is not None:
                stt_thread.join(timeout=self.cfg.get("stt.remote_timeout", 30) + 5)
                text = (stt_result.get("text") or "").strip()
            elif stt_result.get("text"):
                text = stt_result["text"].strip()  # queued turn: already transcribed
            else:
                rescue = getattr(self.stt, "rescue_transcribe", None)
                if stt_rescue and callable(rescue):
                    # The streaming session already came up empty on this very
                    # audio — go straight to the engine that can hear it
                    # (Whisper) instead of paying the primary a second time.
                    text = rescue(utterance, sr).strip()
                else:
                    text = self.stt.transcribe(utterance, sr).strip()
            if not text:
                # A turn that produces nothing used to vanish in silence — a
                # visitor spoke and got no answer, with no trace of why.
                log.info("no transcript for a %.1fs utterance (rms %.0f) — "
                         "nothing said, or the recogniser missed it",
                         getattr(utterance, "size", 0) / max(1, sr),
                         float(np.sqrt(np.mean(np.square(
                             utterance.astype("float64")))) * 32768.0)
                         if getattr(utterance, "size", 0) else 0.0)
                continue  # misfire / noise — keep listening, stay in conversation

            # Provisional guest — only now that the turn has PROVEN itself real:
            # a transcript with substance, enough speech, enough level. Without
            # these gates, silence hallucinations were minting phantom guests.
            if (self._turn_speaker is None and self.guests is not None
                    and getattr(self, "_turn_voice_emb", None) is not None
                    and self._guest_worthy(utterance, text, sr)):
                self._turn_speaker = self.guests.assign(self._turn_voice_emb)
            self._turn_voice_emb = None

            # Erasure on request: "forget that" / "forget everything about me".
            forget = parse_forget_command(text)
            if forget and self.memory is not None:
                pid = self._person.person_id if self._person is not None else None
                if forget == "me":
                    n = self.memory.forget_person(pid) if pid is not None else 0
                    if self._person is not None and self.identity is not None:
                        self.identity.forget(self._person.name)
                        line = (f"Done — I've forgotten everything about you, "
                                f"{self._person.name}, including your face and "
                                "voice.")
                        self._person = None
                    else:
                        line = ("I don't have anything stored about you "
                                "specifically." if n == 0 else "Done.")
                else:
                    what = self.memory.forget_last(
                        self._person.person_id if self._person else None)
                    line = ("Forgotten." if what
                            else "There's nothing recent to forget.")
                self._to(State.SPEAKING)
                self._speak_one(line)
                self.convo.add_user(text)
                self.convo.add_assistant(line)
                continue

            # Affect: per-turn read folded into a cross-turn mood (EMA). The
            # mood steers the LLM's tone note AND the voice's delivery.
            turn_affect = None
            if self.affect is not None:
                read = self.affect.estimate(utterance, sr, text, frame_rgb=frame)
                turn_affect = read     # reward tagging reads the same signal
                label, mood_note = self._mood.update(read)
                if mood_note:
                    self._turn_notes.append(mood_note)
                set_mood = getattr(self.voice, "set_mood", None)
                if callable(set_mood):
                    set_mood(label)

            # Self-state narration: when a stage is running on its fallback,
            # ZERO knows — and can be upfront instead of sounding "off".
            self._turn_notes.extend(self._self_state_notes())

            # Enrolment. Two ways in:
            #   * name introduction   — "I'm David" / "my name is David"
            #   * explicit command    — "remember my face", "learn me as David"
            # Both run the guided multi-angle capture so recognition is robust.
            if self.identity is not None:
                cmd = parse_enroll_command(text)         # explicit request
                name_intro = parse_enrollment(text) if cmd is None else None
                if cmd is not None or name_intro:
                    if cmd:                              # explicit + name
                        name = cmd
                    elif name_intro:                     # "I'm David"
                        name = name_intro
                    elif self._person is not None:       # "remember my face" + known
                        name = self._person.name
                    else:                                # command, but no name known
                        name = None
                    if name:
                        line = self._guided_enroll(name, first_audio=utterance)
                    else:
                        line = ("Sure — tell me your name while you look at me, "
                                "and I'll remember your face.")
                    self._to(State.SPEAKING)
                    self._speak_one(line)
                    self.convo.add_user(text)
                    self.convo.add_assistant(line)
                    continue

            # Object teaching: "this is a french press" — the article is what
            # separates an OBJECT from a person ("this is Peter"). Binds the
            # current crop to the name; the detector hint uses it from now on.
            obj_name = parse_object_teach(text) if self.eyes is not None else None
            if obj_name:
                pid = self._person.person_id if self._person is not None else None
                ok = self.eyes.teach_object(obj_name, person_id=pid)
                line = (f"Got it — that's a {obj_name}. I'll recognise it now."
                        if ok else
                        "I want to learn that, but I can't get a good look "
                        "right now — hold it up for me and tell me again.")
                self._to(State.SPEAKING)
                self._speak_one(line)
                self.convo.add_user(text)
                self.convo.add_assistant(line)
                continue

            # Behavioural corrections: "speak slower", "keep it short" — stored
            # as standing preferences and applied to engine knobs where one exists.
            vol = parse_volume(text)
            if vol is not None:
                pref_text, level = vol
                self._voice_level = level
                if self.memory is not None and self.cfg.get(
                        "preferences.enabled", True):
                    self.memory.set_preference(
                        pref_text,
                        person_id=(self._person.person_id
                                   if self._person is not None else None))
                log.info("voice level -> x%.2f (%s)", level, pref_text)
                line = ("Okay, I'll keep it down." if level < 1.0
                        else "Sure, I'll speak up." if level > 1.0
                        else "Okay, back to normal.")
                self._to(State.SPEAKING)
                self._speak_one(line)
                self.convo.add_user(text)
                self.convo.add_assistant(line)
                continue

            if self.memory is not None and self.cfg.get("preferences.enabled", True):
                pref = parse_preference(text)
                if pref is not None:
                    pref_text, rate_delta = pref
                    pid = (self._person.person_id
                           if self._person is not None else None)
                    self.memory.set_preference(pref_text, person_id=pid)
                    if rate_delta is not None:
                        apply_rate_delta(self.voice, rate_delta)
                    line = f"Okay — I'll {pref_text} from now on."
                    self._to(State.SPEAKING)
                    self._speak_one(line)
                    self.convo.add_user(text)
                    self.convo.add_assistant(line)
                    continue

            if self._is_stop(text):
                self._to(State.SPEAKING)
                self._speak_one("Okay, talk to you later.")
                log.info("Sleeping… (stop phrase)")
                self._end_conversation()
                return

            self._maybe_remember(text)  # explicit "remember that ..."

            # Store the RAW utterance so history stays clean and cache-friendly.
            self.convo.add_user(text)
            # Tag the turn with its (confident) speaker so end-of-session memory can
            # be split per person instead of lumped onto one owner; the corpus log
            # keeps the fuller speaker (incl. provisional guests) for training data.
            # Privacy is PER TURN: a turn the guard says not to store never enters
            # either log — so a bystander's words are never persisted, and one
            # stranger at the end can't void a known person's whole session.
            if self._memory_allowed:
                self._session_log.append((self._turn_durable_pid, "user", text))
                self._corpus_log.append((self._turn_speaker, "user", text))
                if self.reward is not None:
                    # Retro-tag: these words may be a verdict on the last reply
                    # (and resolve a pending proactive outcome). Same privacy
                    # gate as the logs — an unstorable turn tags nothing.
                    self.reward.on_user(text)
            self._stage("stt-total")
            self._t_reply_start = time.monotonic()  # end-of-STT marker for timing
            # Afterthought merge, round 1: a remark finished during the STT/
            # identity work ("...oh and make it two") joins the turn BEFORE the
            # LLM ever sees it — the cheapest conversation-turn there is.
            extra = self._pop_afterthoughts()
            if extra:
                log.info("afterthought merged pre-LLM: %r", extra)
                text = f"{text} {extra}"
                self.convo.amend_last_user(extra)
                for lg in (self._session_log, self._corpus_log):
                    if lg and lg[-1][1] == "user":
                        lg[-1] = (lg[-1][0], "user", text)
            # Pre-open the first sentence's TTS socket NOW, so its connect
            # (~300-700ms to the Kyutai server over the tailnet) runs under
            # the vision/recall build and the LLM's first token instead of
            # after them. Non-blocking; a no-op for engines without prewarm.
            _pw = getattr(self.voice, "prewarm", None)
            if callable(_pw):
                _pw()
            # Fold in what ZERO currently sees as an EPHEMERAL note on THIS turn
            # only — the note + keyframes are attached to the outgoing copy of the
            # messages, never saved to history, so the cached prefix and future
            # turns stay clean and image-free.
            messages = self._attach_vision(self.convo.messages(), text)
            self._stage("vision+recall")
            # Kick the LLM off in the BACKGROUND so its prefill overlaps the spoken
            # filler — the model is already generating while the filler plays, so the
            # real answer flows in with little or no added delay.
            # A bet placed before they finished only survives if the finished
            # sentence is word-for-word what it was placed on — otherwise it is
            # discarded and we generate normally. Nothing half-relevant is ever
            # allowed to reach the speaker.
            taken = self._take_speculation(text)
            if taken is not None:
                chunks, llm_stop = taken
            else:
                chunks, llm_stop = self._stream_in_background(messages)
            # Degeneracy guard on the FINAL stream. The engine-level guard only
            # sees its own socket; the tool router re-prompts through a
            # separate path, which is exactly where "thought thought thought"
            # reached the speaker. Everything spoken passes through here.
            chunks = self._guard_degenerate(chunks, llm_stop)
            # Afterthought merge, round 2: a remark that landed while the
            # vision/recall context was being built. The in-flight stream is
            # abandoned and restarted with the completed thought — the warm
            # prefix makes the restart cost roughly one first-token.
            extra = self._pop_afterthoughts()
            if extra:
                llm_stop.set()   # covers a speculative stream too
                log.info("afterthought merged, reply restarted: %r", extra)
                text = f"{text} {extra}"
                self.convo.amend_last_user(extra)
                for lg in (self._session_log, self._corpus_log):
                    if lg and lg[-1][1] == "user":
                        lg[-1] = (lg[-1][0], "user", text)
                messages = self._attach_vision(self.convo.messages(), text)
                chunks, llm_stop = self._stream_in_background(messages)
            _room = getattr(self, "room", None)
            # Level = what they ASKED for x what the ROOM needs. A request to
            # be quiet still holds in a noisy hall; it just isn't taken to the
            # point of being inaudible.
            _lvl = getattr(self, "_voice_level", 1.0)
            self.speaker.gain = (_lvl * _room.speech_gain()
                                 if _room is not None else _lvl)
            self._to(State.SPEAKING)
            # The monitor has been live since the turn committed; it covers the
            # filler AND the reply, so speech or the wake word can cut ZERO off
            # at any point while it's making sound.
            try:
                # The filler RACES the real reply: it only plays if no reply
                # audio arrived within the grace window, so a fast answer is
                # never delayed by a canned "let me think".
                reply = self._speak_streaming(chunks, llm_stop,
                                              filler_audio=self._pick_filler(text))
            finally:
                self._stop_monitor()
            if self._interrupt or self._soft_stop:
                llm_stop.set()  # stop generating a reply nobody is listening to
                self._was_interrupted = True
                log.info("barge-in: %s", "hard stop (correction/wake)"
                         if self._interrupt else "yielded at the sentence end")
            # NEVER GO SILENT. A turn that produced no audio means the model
            # was unreachable, the stream died, or the reply was cut for
            # degeneracy. In a room with people, dead air reads as broken —
            # say something human and stay in the conversation.
            if self._degenerate:
                self._degenerate = False
                self._say_recovery("lost")
                log.warning("reply was degenerate — spoke a recovery line")
                continue
            if not reply and not self._interrupt and not self._soft_stop:
                spoke = self._say_recovery("slow" if self._llm_unreachable
                                           else "lost")
                log.warning("empty reply — spoke a recovery line (%s)",
                            "ok" if spoke else "NO CLIP AVAILABLE")
                self._llm_unreachable = False
                continue

            # Store only what was actually SPOKEN — after a barge-in the model must
            # not "remember" saying sentences the user never heard.
            if reply:
                self.convo.add_assistant(reply)
                if self._memory_allowed:
                    self._session_log.append(
                        (self._turn_durable_pid, "assistant", reply))
                    self._corpus_log.append(
                        (self._turn_speaker, "assistant", reply))
                    if self.reward is not None:
                        # One reward-tagged episode per exchange: tone while
                        # speaking + barge-in + engagement now; an explicit
                        # verdict may retro-tag it next utterance.
                        self.reward.on_turn(
                            text, reply, affect=turn_affect,
                            barged_in=self._interrupt or self._soft_stop,
                            person_id=self._turn_durable_pid)
                log.info("reply: %r", reply)

    # -- vision -------------------------------------------------------------
    def _is_visual(self, text: str) -> bool:
        labels: set[str] = set()
        if self.eyes is not None:
            try:
                labels = self.eyes.visible_labels()
            except Exception:  # a scene read must never break a turn
                pass
        return is_visual_question(text, labels)

    def _look(self, text: str) -> tuple[str, list[str]]:
        """Ephemeral (note, keyframes) for this turn — never persisted, never raises.

        Ambient turns get only the cheap text hint; visual turns also pull a few
        recent keyframes so the multimodal LLM can actually see.
        """
        if self.eyes is None:
            return "", []
        try:
            t0 = time.monotonic()
            visual = self._is_visual(text)
            t1 = time.monotonic()
            if visual:
                ctx = self.eyes.visual_context(question=text)
                out = (ctx.text, ctx.images)
            else:
                out = (self.eyes.local_context(), [])
            t2 = time.monotonic()
            if t2 - t0 > 0.3:
                log.info("look breakdown: is_visual=%.2fs context=%.2fs "
                         "(visual=%s)", t1 - t0, t2 - t1, visual)
            return out
        except Exception as e:  # vision must never break a conversation turn
            log.debug("vision look failed: %s", e)
            return "", []

    def _attach_vision(self, messages: list, text: str) -> list:
        """Attach the live-sight note + keyframes to the FINAL user message of a
        throwaway copy of ``messages``. The persona prompt ("Your eyes") tells the
        model to treat the parenthetical as its own perception, not user text.
        """
        _t0 = time.monotonic()
        note, images = self._look(text)
        _t_look = time.monotonic() - _t0
        ident = self._person

        # Ground PRESENCE in the detector (truth), not the LLM's imagination —
        # otherwise it happily says "yes I see you" to an empty room. YOLO is
        # authoritative for "is a person in frame right now".
        person_present = None
        if self.eyes is not None:
            try:
                person_present = "person" in self.eyes.visible_labels()
            except Exception:  # a scene read must never break a turn
                person_present = None

        # Identity note — only claim to SEE someone when their FACE is in the
        # current frame (perception, so it's face-driven and independent of who
        # OWNS the session). Voice-only recognition (they're talking off-camera)
        # must NOT let ZERO claim it can see them.
        id_note = ""
        id_key = None
        face_name = getattr(self, "_face_name", None)
        if face_name:
            id_key = ("face", face_name)
            id_note = f"(You can see {face_name} — you recognise their face.)"
        elif ident is not None and ident.is_known:
            id_key = ("voice", ident.name)
            id_note = (f"(You recognise {ident.name}'s voice, but you can't "
                       "see their face in the camera right now.)")
        # Attach it only when recognition CHANGES (first sighting, a new person,
        # a face appearing) or the user asks a visual question. Repeating it every
        # turn fed the model greeting-fodder — it kept re-greeting mid-topic
        # ("hey Greg, you look good today") instead of staying on the subject.
        if (id_note and id_key == getattr(self, "_last_id_key", None)
                and not self._is_visual(text)):
            id_note = ""
        self._last_id_key = id_key

        # Decisive presence fact so "can you see me?" is answered from reality.
        presence_note = ""
        if self.eyes is None:
            # Camera never came up this session — ZERO is BLIND. The persona
            # assumes it can see, so without this it invents a scene (a phantom
            # "bookshelf"). Only bother the LLM with it on a visual question.
            if self._is_visual(text):
                presence_note = (
                    "(Your camera is offline right now — you cannot see anything "
                    "at all. If asked what you see, say honestly that your eyes "
                    "aren't working at the moment. Do NOT invent or describe a "
                    "scene.)")
        elif person_present is False:
            presence_note = (
                "(Your camera view has NO people in it right now. If asked "
                "whether you can see someone, say honestly that you can't see "
                "anyone at the moment — do NOT pretend to.)")
        elif person_present is True and not id_note:
            presence_note = "(There's a person in your camera view right now.)"

        # Relevance recall: what a human would "think of" hearing this turn —
        # ephemeral, attached to the outgoing copy only, never history.
        # HARD TIME BUDGET: recall may hit the GPU embedder, and a slow embed
        # sitting in the reply path showed up as multi-second first-token lag.
        # If it doesn't come back inside the budget, this turn just goes
        # without the note — never slower.
        recall_note = ""
        if self.memory is not None:
            budget_s = self.cfg.get("memory.retrieval.budget_ms", 300) / 1000.0
            pid = ident.person_id if ident is not None else None
            res: dict = {}

            convo = getattr(self, "convo", None)
            already = convo.memory_block if convo is not None else ""

            def _recall():
                try:
                    res["block"] = self.memory.relevant_block(
                        text, person_id=pid, exclude=already)
                except Exception as e:  # recall must never break a turn
                    log.debug("recall failed: %s", e)

            t = threading.Thread(target=_recall, name="recall", daemon=True)
            t.start()
            t.join(timeout=budget_s)
            recalled = res.get("block", "")
            if t.is_alive():
                log.debug("recall skipped this turn (over %dms budget)",
                          int(budget_s * 1000))
            if recalled:
                recall_note = f"(This reminds you of things you know: {recalled}.)"
        _t_recall = time.monotonic() - _t0
        turn_notes = list(getattr(self, "_turn_notes", []) or [])

        # Spontaneous visual awareness: debounced scene changes ("a guitar just
        # came into view") surface as a note the LLM may mention — or not.
        # NEVER while the person is asking something: answering "which planet
        # did Thanos visit" with "is that a remote on the table?" reads as not
        # listening. Ungated, the changes stay queued (and self-expire) until a
        # calmer turn.
        if self.eyes is not None and self._filler_category(text) != "question":
            try:
                changes = self.eyes.scene_changes()
                if changes:
                    turn_notes.append(
                        f"(You just noticed: {'; '.join(changes)}. Mention it "
                        "only if the current topic has wound down — never "
                        "derail what they're talking about; otherwise just "
                        "let it go.)")
            except Exception:  # a scene read must never break a turn
                pass
        _t_total = time.monotonic() - _t0
        log.info("turn context: look=%.2fs id+recall=%.2fs changes=%.2fs "
                 "total=%.2fs", _t_look, _t_recall - _t_look,
                 _t_total - _t_recall, _t_total)

        if not (note or images or id_note or presence_note or recall_note
                or turn_notes):
            return messages
        last = dict(messages[-1])
        extras = []
        if id_note:
            extras.append(id_note)
        if presence_note:
            extras.append(presence_note)
        extras.extend(turn_notes)   # speaker change, affect — this turn only
        if recall_note:
            extras.append(recall_note)
        if note:
            extras.append(
                f"(Right now, through your camera, you can see: {note}.)")
        if extras:
            last["content"] = last["content"] + "\n\n" + "\n".join(extras)
        if images:
            last["images"] = images  # THIS turn only — not stored in history
        return [*messages[:-1], last]

    def _guest_worthy(self, utterance, text: str, sr: int) -> bool:
        """Should this turn mint (or match) a provisional guest? Real speech
        only: enough words, enough duration, enough level. Whisper hallucinates
        plausible text from near-silence ('Obrigado', subtitle credits) — those
        junk turns must not create phantom guests or pollute the corpus."""
        if len(text.split()) < self.cfg.get("identity.guests.min_words", 2):
            return False
        dur_ms = 1000.0 * getattr(utterance, "size", 0) / max(1, sr)
        if dur_ms < self.cfg.get("identity.guests.min_ms", 1200):
            return False
        import numpy as np

        rms = float(np.sqrt(np.mean(np.square(
            utterance.astype("float64"))))) * 32768.0
        return rms >= self.cfg.get("identity.guests.min_rms", 150)

    # -- returning-person recall --------------------------------------------
    @staticmethod
    def _humanize_gap(seconds: float) -> str:
        """A natural 'how long since' phrase for the welcome-back note."""
        d = max(0.0, seconds) / 86400.0
        if d < 0.5:
            return "earlier today"
        if d < 1.5:
            return "yesterday"
        if d < 10:
            return f"{round(d)} days ago"
        if d < 45:
            return f"about {max(1, round(d / 7))} weeks ago"
        if d < 350:
            return f"about {max(1, round(d / 30))} months ago"
        years = d / 365.0
        return "about a year ago" if years < 1.5 else f"about {round(years)} years ago"

    def _welcome_back_note(self, ident) -> str:
        """One-time 'welcome back' the first time a known voice is heard this
        conversation: surfaces the durable last-conversation record so ZERO picks
        up where it left off — even after years. '' when there's nothing to say."""
        if (self.memory is None
                or not self.cfg.get("memory.last_conversation.enabled", True)):
            return ""
        pid = ident.person_id
        if pid in self._welcomed:
            return ""
        self._welcomed.add(pid)  # fire once per person, even if there's no record
        try:
            last = self.memory.last_conversation(pid)
        except Exception as e:  # recall must never break a turn
            log.debug("last-conversation lookup failed: %s", e)
            return ""
        if not last:
            return ""
        summary, when = last
        gap = self._humanize_gap(time.time() - (when or time.time()))
        return (f"(You've spoken with {ident.name} before. Last time ({gap}) you "
                f"talked about: {summary} Greet them like you remember them, and "
                "pick that thread back up if it fits — naturally, don't recite it.)")

    # -- long-term memory ---------------------------------------------------
    def _maybe_remember(self, text: str) -> None:
        """Explicit 'remember that ...' — store immediately so it survives even if
        the conversation never ends cleanly."""
        if self.memory is None or not getattr(self, "_memory_allowed", True):
            return  # guarded/strict privacy: strangers are never remembered
        low = text.lower()
        pid = self._person.person_id if self._person is not None else None
        for trigger in ("remember that ", "remember "):
            if low.startswith(trigger):
                fact = text[len(trigger):].strip()
                if fact:
                    self.memory.remember(f"note ({int(time.time())})", fact,
                                         person_id=pid, importance=7.0)
                return

    # -- in-session compaction ----------------------------------------------
    def _maybe_compact(self) -> None:
        """Fold trimmed-away turns into a rolling summary on a background thread,
        so a long session's context stays small (bounded KV cache) instead of the
        shared GPU filling up. Single-flight; the summariser easily keeps pace
        with the occasional trim."""
        if not self.cfg.get("conversation.compaction.enabled", True):
            return
        if self._summary_thread is not None and self._summary_thread.is_alive():
            return
        snap = self.convo.pending_snapshot()
        if snap is None:
            return
        prev_summary, pending = snap
        self._summary_thread = threading.Thread(
            target=self._compact, args=(prev_summary, pending),
            name="compaction", daemon=True)
        self._summary_thread.start()

    def _compact(self, prev_summary: str, pending: list) -> None:
        """Merge the summary-so-far with the freshly trimmed turns into one short
        rolling summary, then install it (dropping the turns it now covers)."""
        try:
            cap = self.cfg.get("conversation.compaction.max_summary_chars", 600)
            turns = "\n".join(f"{m['role']}: {m['content']}" for m in pending)
            body = (f"Summary so far:\n{prev_summary}\n\nNewer turns:\n{turns}"
                    if prev_summary else turns)
            prompt = [
                {"role": "system", "content": (
                    "Maintain a running summary of this conversation so the "
                    "assistant can keep the thread without the full transcript. "
                    "Merge the summary-so-far with the newer turns into ONE concise "
                    f"summary under {cap} characters — key facts, decisions, open "
                    "threads and the current topic, from the assistant's point of "
                    "view. Output only the summary.")},
                {"role": "user", "content": body},
            ]
            summary = "".join(self.llm.stream(prompt)).strip()[:cap]
            if summary:
                self.convo.apply_summary(summary, pending)
                log.debug("compacted %d turns into rolling summary (%d chars)",
                          len(pending), len(summary))
        except Exception as e:  # compaction must never break the loop
            log.warning("compaction failed: %s", e)

    def _end_conversation(self) -> None:
        """On sleep/stop: split the chat by speaker and save each person's durable
        memory on a BACKGROUND thread, so ZERO goes back to listening for the wake
        word immediately instead of being deaf for seconds. The speaker log is
        snapshotted first — the next conversation may reset it while the save is
        still running."""
        self._stop_monitor()  # a stop-phrase exit arrives with the monitor live
        if self.reward is not None:
            # A proactive nudge nobody answered scores against its kind, and
            # per-session tagging state resets.
            self.reward.end_session()
        # Privacy is enforced PER TURN at log time (a guarded stranger's words
        # never entered these logs), so anything present here may be persisted.
        if ((self.memory is not None or self.corpus is not None)
                and (self._session_log or self._corpus_log)):
            session_log = list(self._session_log)
            corpus_log = list(self._corpus_log)
            # Let a still-running previous save finish first — two concurrent
            # savers could interleave corpus lines and double-consolidate.
            prev = self._memory_thread
            self._memory_thread = threading.Thread(
                target=self._persist_session,
                args=(session_log, corpus_log, prev),
                name="memory-save", daemon=True,
            )
            self._memory_thread.start()
        # Clear immediately so a mode without _converse's reset (text mode)
        # can't re-save this session's turns at the next end.
        self._session_log = []
        self._corpus_log = []
        self._person = None  # identity does not persist across conversations
        self._reset_turn_state()  # no interrupt/duck/queued-turn survives a session
        self._to(State.IDLE)

    def _persist_session(self, session_log: list, corpus_log: list,
                         prev: threading.Thread | None = None) -> None:
        """Background: save this session's training corpus AND per-person memory.
        Waits for the previous session's save (if still running) so writers never
        overlap; the corpus write is cheap (file append) and runs first so a slow
        memory pass can't lose the raw data."""
        if prev is not None and prev.is_alive():
            prev.join(timeout=60.0)
        try:
            self._save_corpus(corpus_log)
        except Exception as e:  # corpus must never break the loop
            log.warning("corpus save failed: %s", e)
        if self.memory is not None and session_log:
            self._save_memories(session_log)

    def _save_corpus(self, corpus_log: list) -> None:
        """Split this session by speaker (real person, provisional guest, or
        anonymous) and append it to the interaction corpus for offline training."""
        if self.corpus is None or not corpus_log:
            return
        by_speaker: dict = {}
        for speaker, role, text in corpus_log:
            by_speaker.setdefault(speaker, []).append({"role": role, "text": text})
        self.corpus.add_session(by_speaker, meta={"source": "voice"})

    def _save_memories(self, session_log: list | None = None) -> None:
        """Per-speaker durable memory. The session is voice-owned, so we split it
        by speaker and draw facts + a summary + the last-conversation record from
        ONLY each person's own turns — nobody's memory is contaminated by another
        speaker who was in the room. Turns without a confident speaker fall into a
        global (anonymous) bucket that still records a rough episode."""
        session_log = session_log or []
        by_person: dict = {}
        for pid, role, text in session_log:
            by_person.setdefault(pid, []).append(f"{role}: {text}")
        for pid, lines in by_person.items():
            # A stray one-word turn isn't a conversation worth mining.
            user_chars = sum(len(ln) for ln in lines if ln.startswith("user: "))
            if user_chars < 15:
                continue
            transcript = "\n".join(lines)
            try:
                self._extract_facts(transcript, pid)
            except Exception as e:  # never let memory break the loop
                log.warning("fact extraction failed for %s: %s", pid, e)
            summary = None
            try:
                summary = self._summarize_session(transcript, pid)  # episodic memory
            except Exception as e:
                log.warning("session summary failed for %s: %s", pid, e)
            # Durable 'last conversation' for THIS person, from THIS person's turns.
            if (summary and pid is not None
                    and self.cfg.get("memory.last_conversation.enabled", True)):
                try:
                    self.memory.set_last_conversation(pid, summary)
                except Exception as e:
                    log.warning("last-conversation save failed for %s: %s", pid, e)
        try:
            # Sleep-phase pass: gentle forgetting + reflection over recent
            # episodes into higher-level insights. Same background thread, so
            # ZERO is already back listening while it "dreams".
            stats = self.memory.consolidate(reflect_fn=self._reflect)
            if stats.get("insights") or stats.get("forgotten"):
                log.info("consolidated memory: %s", stats)
        except Exception as e:
            log.warning("memory consolidation failed: %s", e)

    def _reflect(self, episodes_text: str) -> list[str]:
        """Distil recent episode summaries into up to 3 higher-level insights."""
        prompt = [
            {"role": "system", "content": (
                "Below are one-line summaries of your recent conversations with "
                "the user. Infer up to 3 higher-level, durable insights about "
                "them (habits, routines, ongoing situations, relationships). "
                "One short sentence per line, no bullets. If none, output "
                "exactly NONE.")},
            {"role": "user", "content": episodes_text},
        ]
        result = "".join(self.llm.stream(prompt)).strip()
        if not result or result.upper().startswith("NONE"):
            return []
        return [ln for ln in result.splitlines() if ln.strip()]

    def _join_memory_thread(self, timeout: float = 30.0) -> None:
        """Give an in-flight background memory save a chance to finish on shutdown."""
        t = self._memory_thread
        if t is not None and t.is_alive():
            log.info("finishing memory save...")
            t.join(timeout=timeout)

    def _extract_facts(self, transcript: str, person_id: int | None = None) -> None:
        prompt = [
            {"role": "system", "content": (
                "Extract durable facts about the USER from the conversation. "
                "Output only lines shaped 'key: value | N' where N is 1-10 — "
                "how important this fact is to remember long-term (name/family "
                "= 9-10, preferences/projects = 6-8, passing details = 2-4). "
                "Short lowercase keys. If there are no durable facts, output "
                "exactly NONE and nothing else.")},
            {"role": "user", "content": transcript},
        ]
        result = "".join(self.llm.stream(prompt)).strip()
        if not result or result.upper().startswith("NONE"):
            return
        for line in result.splitlines():
            if ":" not in line:
                continue
            key, _, rest = line.partition(":")
            value, _, imp_s = rest.partition("|")
            try:
                importance = float(imp_s.strip())
            except ValueError:
                importance = 5.0
            self.memory.remember(key.strip("-• ").strip(), value.strip(),
                                 person_id=person_id, importance=importance)

    def _summarize_session(self, transcript: str,
                           person_id: int | None = None) -> str | None:
        """Write a one-line summary of the conversation into episodic memory, so
        next time ZERO can recall the ARC of past chats, not just isolated facts.
        Returns the summary (also reused as each participant's durable
        last-conversation record), or None if nothing of substance was said."""
        prompt = [
            {"role": "system", "content": (
                "Summarize this conversation in ONE short sentence — what you and "
                "the user talked about or what happened — from your point of view "
                "as the assistant. No preamble, just the sentence. If nothing of "
                "substance was said, output exactly NONE.")},
            {"role": "user", "content": transcript},
        ]
        summary = "".join(self.llm.stream(prompt)).strip()
        if summary and not summary.upper().startswith("NONE"):
            self.memory.add_episode(summary, person_id=person_id)
            return summary
        return None

    def _stream_in_background(self, messages) -> tuple:
        """Start the LLM streaming on a worker thread feeding a queue. Returns
        (generator, stop_event). Calling this begins prefill immediately (a bare
        generator would wait until first consumed), so it overlaps the filler.
        Setting the stop event makes the worker abandon the stream — used on
        barge-in so the GPU stops generating a reply nobody will hear.
        """
        q: "queue.Queue" = queue.Queue()
        stop = threading.Event()

        def worker():
            stream = self.llm.stream(messages)
            try:
                for chunk in stream:
                    if stop.is_set():
                        break  # barge-in: abandon the rest of the reply
                    q.put(chunk)
            except Exception as e:  # never let the worker die silently
                log.error("LLM stream error: %s", e)
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()  # closes the HTTP stream so the server stops generating
                q.put(None)  # sentinel = done

        threading.Thread(target=worker, name="llm-stream", daemon=True).start()

        def gen():
            while True:
                item = q.get()
                if item is None:
                    return
                yield item

        return gen(), stop

    _QUESTION_WORDS = {
        "what", "why", "how", "when", "who", "where", "which", "whose", "can",
        "could", "would", "do", "does", "did", "is", "are", "should", "tell",
        "explain", "describe",
    }

    def _filler_category(self, text: str) -> str:
        """Pick the filler that FITS what the user just said, so it sounds aware:
        a question gets 'Good question, let me think.'; a one-word reply gets a
        quick 'Mm-hmm.'; everything else gets a neutral 'Let's see.'"""
        t = text.lower().strip()
        words = t.split()
        if t.endswith("?") or (words and words[0] in self._QUESTION_WORDS):
            return "question"
        if len(words) <= 2:
            return "ack"
        return "default"

    def _pick_filler(self, user_text: str):
        """One pre-synthesized filler matched to what the user said, or None.
        Handed to _speak_streaming, which plays it only if the real reply's
        audio hasn't arrived within the grace window."""
        if random.random() > self._filler_prob:
            return None
        category = self._filler_category(user_text)
        audios = self._fillers.get(category) or self._fillers.get("default") or []
        if not audios:
            return None
        log.debug("filler category: %s", category)
        return random.choice(audios)

    def _speak_streaming(self, chunks, llm_stop: threading.Event,
                         filler_audio=None) -> str:
        """Stream the reply: a producer thread turns the LLM text into sentences and
        streams each sentence's AUDIO CHUNKS onto a queue, while a single gapless
        output stream plays them as they arrive. First audio lands ~200ms after the
        first sentence starts generating, and there are no inter-sentence pauses.

        Returns the text that was actually SPOKEN. On barge-in the whole pipeline
        is shut down — LLM worker, TTS producer, queue — so nothing keeps
        generating/synthesizing a reply nobody is listening to, and only the
        sentences whose audio reached the speaker are returned for history.
        """
        full: list[str] = []          # every sentence handed to TTS, in order
        audio_q: "queue.Queue" = queue.Queue(maxsize=32)
        stop_evt = threading.Event()  # tells the producer to abandon synthesis
        played = -1                   # index into `full` of the last sentence heard
        # Anything the model writes in (parens) is a hallucinated "note" — it
        # must never be spoken, and never enter history via `full`.
        chunks = strip_asides(chunks)

        def put_piece(item) -> bool:
            """Queue an audio piece without ever blocking forever: on barge-in the
            consumer stops draining, and a plain put() would wedge this thread."""
            while not stop_evt.is_set():
                try:
                    audio_q.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def producer():
            buffer = ""
            first_token = True
            try:
                for chunk in itertools.chain(chunks, ["\n"]):  # sentinel flush
                    if stop_evt.is_set():
                        return
                    if first_token and chunk.strip():
                        log.info("LLM first token: %.2fs",
                                 time.monotonic() - self._t_reply_start)
                        first_token = False
                    buffer += chunk
                    # Rescan when this chunk could complete a sentence — or,
                    # before ANY audio has been queued, a clause: the first
                    # sound then starts at the first comma instead of waiting
                    # for the whole opening sentence (split_stream eager_first).
                    eager = not full
                    if not any(c in chunk for c in ".!?…\n") and not (
                            eager and any(c in chunk for c in ",;:—–")):
                        continue
                    # split_stream keeps the remainder's original whitespace —
                    # taking split_sentences' stripped tail as the new buffer
                    # lost the space between chunks ("I'm " + "still" was
                    # spoken as "I'mstill").
                    complete, buffer = split_stream(buffer, eager_first=eager)
                    for sentence in complete:
                        full.append(sentence)
                        idx = len(full) - 1
                        for piece in self.voice.synthesize_stream(sentence):
                            if not put_piece((idx, piece)):
                                return
                if buffer.strip():
                    full.append(buffer.strip())
                    idx = len(full) - 1
                    for piece in self.voice.synthesize_stream(buffer):
                        if not put_piece((idx, piece)):
                            return
            finally:
                # Deliver the "done" sentinel RELIABLY. put_nowait dropped it
                # whenever the queue was full — which happens the moment synthesis
                # outpaces playback (e.g. fast GPU Orpheus filling the 32 slots).
                # A dropped sentinel left the consumer blocked on get() forever:
                # playback never returned and the whole conversation froze in
                # SPEAKING (ZERO went deaf after the first reply). Block until it
                # lands; if a barge-in already set stop_evt, the drain path cleans
                # up instead, so we don't wait on a stalled consumer.
                while not stop_evt.is_set():
                    try:
                        audio_q.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        continue

        prod = threading.Thread(target=producer, name="tts-producer", daemon=True)
        prod.start()

        spoke_any = False

        def audio_gen():
            nonlocal played, spoke_any
            pending_filler = filler_audio
            while True:
                if pending_filler is not None:
                    try:  # race: real audio within the grace window wins
                        item = audio_q.get(timeout=self._filler_grace_s)
                    except queue.Empty:
                        yield pending_filler
                        pending_filler = None
                        continue
                    pending_filler = None
                else:
                    item = audio_q.get()
                if item is None:
                    return
                idx, piece = item
                # Soft stop (a QUEUE interruption): finish the sentence being
                # spoken, then yield the floor at the boundary — the polite
                # human ending, not a mid-word cut.
                if self._soft_stop and spoke_any and idx != played:
                    return
                if not spoke_any:
                    now = time.monotonic()
                    # The end-to-end number that matters: silence from the
                    # user's last word to ZERO's first sound.
                    gap = (now - self._t_utterance_end
                           if self._t_utterance_end else float("nan"))
                    log.info("first audio out: %.2fs after STT, %.2fs after "
                             "end of speech",
                             now - self._t_reply_start, gap)
                    # Where did that time actually go? Printed once per turn,
                    # right when the first sound leaves the speaker.
                    if self._t_utterance_end:
                        self._stage("llm+tts")
                        self._stage_report((now - self._t_utterance_end) * 1000.0)
                    spoke_any = True
                played = idx
                yield piece

        # Barge-in: the monitor (started by the caller) keeps the mic live and
        # listens for the wake word; saying it cuts ZERO off mid-reply (echo-safe:
        # ZERO never says its own wake word, so its voice won't false-trigger).
        self.speaker.play_stream(audio_gen(), self.voice.sample_rate,
                                 should_stop=self._should_interrupt)

        if self._interrupt or self._soft_stop:
            # Shut the whole pipeline down and unblock the producer, then return
            # only the sentences the user actually heard (a soft stop ended at
            # a sentence boundary, so `played` is exactly the last one spoken).
            llm_stop.set()
            stop_evt.set()
            self._drain_audio_queue(audio_q, prod)
            return " ".join(full[: played + 1]).strip()
        prod.join(timeout=5.0)
        return " ".join(full).strip()

    @staticmethod
    def _drain_audio_queue(q: "queue.Queue", producer: threading.Thread) -> None:
        """Empty the audio queue until the producer's sentinel arrives (or the
        producer dies), so a producer blocked on a full queue always unblocks."""
        while True:
            try:
                if q.get(timeout=0.2) is None:
                    return
            except queue.Empty:
                if not producer.is_alive():
                    return

    def _should_interrupt(self) -> bool:
        """Barge-in hook: True stops playback the instant the wake word fires."""
        return self._interrupt

    def _presynth_recovery(self) -> dict:
        """Render the recovery lines once, at startup, and keep the WAVEFORMS.
        This is the whole point: when the exhibition wifi drops, TTS is gone
        too, so anything synthesised on demand would also fail. These are
        already audio."""
        out: dict[str, list] = {}
        for kind, lines in self._RECOVERY_LINES.items():
            clips = []
            for line in lines:
                try:
                    audio = self.voice.synthesize(line)
                except Exception as e:
                    audio = None
                    log.debug("recovery synth failed for %r: %s", line, e)
                if getattr(audio, "size", 0):
                    clips.append(audio)
            out[kind] = clips
        total = sum(len(v) for v in out.values())
        if total:
            log.info("pre-synthesized %d recovery lines (offline-safe)", total)
        else:
            log.warning("NO recovery lines cached — a failed turn will be "
                        "SILENT. Check the TTS service before going live.")
        return out

    def _say_recovery(self, kind: str = "retry") -> bool:
        """Speak a cached recovery line. Never raises, never synthesises."""
        clips = (self._recovery.get(kind) or self._recovery.get("retry") or [])
        if not clips:
            return False
        try:
            self.speaker.play(random.choice(clips), self.voice.sample_rate,
                              should_stop=lambda: False)
            return True
        except Exception as e:
            log.debug("recovery playback failed: %s", e)
            return False

    def _guard_degenerate(self, chunks, llm_stop):
        """Wrap a reply stream and cut it if it collapses into repetition.
        Sits at the last point before sentence-splitting, so it covers the
        plain engine AND the tool router. Trips the recovery line rather than
        letting a broken reply reach the room."""
        from zero.llm.openai_engine import _DegenerateGuard

        guard = _DegenerateGuard()
        for chunk in chunks:
            for tok in chunk.split(" "):
                if tok and guard.feed(tok):
                    self._degenerate = True
                    try:
                        llm_stop.set()
                    except Exception:
                        pass
                    return
            yield chunk

    def _stage(self, name: str) -> None:
        """Record a stage boundary for the per-turn latency breakdown. The
        end-to-end number told us WHAT the latency was but never WHERE it
        went, which is how ~570ms stayed invisible behind two stages that
        looked individually reasonable."""
        now = time.monotonic()
        prev = getattr(self, "_stage_t", None) or self._t_utterance_end or now
        self._stage_marks.append((name, (now - prev) * 1000.0))
        self._stage_t = now

    def _stage_report(self, first_audio_ms: float) -> None:
        if not self._stage_marks:
            return
        parts = " ".join(f"{n}={ms:.0f}" for n, ms in self._stage_marks)
        total = sum(ms for _, ms in self._stage_marks)
        log.info("LATENCY ms: %s | accounted=%.0f first-audio=%.0f "
                 "UNACCOUNTED=%.0f", parts, total, first_audio_ms,
                 first_audio_ms - total)
        self._stage_marks = []
        self._stage_t = None

    def _reset_turn_state(self) -> None:
        """Clear every cross-turn interrupt artefact — a stale queued turn, a
        leftover duck, a half-set flag. Called at conversation boundaries so
        nothing can leak from one conversation (or a crashed turn) into the
        next."""
        self._interrupt = False
        self._soft_stop = False
        self._bargein_frames = None
        self._queued_turn = None
        self._interrupt_note = None
        self._was_interrupted = False
        self._last_backchannel = 0.0
        self._afterthoughts = []
        self._overheard = []
        self._degenerate = False
        if self._speculation is not None:
            self._speculation.abandon()
            self._speculation = None
        self._close_live_stt()
        if not self.text_mode:
            self.speaker.unduck()

    def _restore_level(self) -> None:
        """Undo the courtesy dip, back to asked-for x room level."""
        try:
            room = getattr(self, "room", None)
            lvl = getattr(self, "_voice_level", 1.0)
            self.speaker.gain = (lvl * room.speech_gain()
                                 if room is not None else lvl)
        except Exception as e:
            log.debug("level restore failed: %s", e)

    def _stop_monitor(self) -> None:
        """Idempotently stop the duplex monitor (if one is running). Safe from
        every path — loop top, post-reply, conversation end — so no early
        `continue` can leave two consumers fighting over the mic queue."""
        m, self._bg_monitor = self._bg_monitor, None
        if m is not None:
            self._stop_bargein(m)

    def _pop_afterthoughts(self) -> str:
        """Everything transcribed from gap remarks since the turn committed,
        joined — '' when there were none. One-shot."""
        parts, self._afterthoughts = self._afterthoughts, []
        return " ".join(p for p in parts if p).strip()

    def _transcribe_afterthought(self, frames) -> None:
        """Monitor thread: transcribe a finished gap remark right away, so the
        merge point (just before speaking) finds text, not raw audio."""
        import numpy as np

        try:
            audio = np.concatenate(frames).astype("float32") / 32768.0
            with self._stt_lock:
                text = (self.stt.transcribe(
                    audio, self.cfg.get("audio.sample_rate", 16000)) or "").strip()
            if text:
                log.info("afterthought heard: %r", text)
                self._afterthoughts.append(text)
        except Exception as e:  # a lost afterthought must never break playback
            log.debug("afterthought stt failed: %s", e)

    def _close_live_stt(self) -> None:
        live, self._live_stt = self._live_stt, None
        if live is not None:
            try:
                live.close()
            except Exception as e:
                log.debug("live STT close failed: %s", e)

    def _open_live_stt(self, sr: int):
        """A live ASR session for this turn, or None when streaming is off /
        the engine can't do it / it fails to open. Never raises — a failure
        here just means the old record-then-transcribe path."""
        if not self.cfg.get("stt.streaming", True):
            return None
        engine = self.stt
        maker = getattr(engine, "live_session", None)
        if maker is None:  # unwrap FallbackSTT to reach the primary
            maker = getattr(getattr(engine, "_primary", None),
                            "live_session", None)
        if maker is None:
            return None
        try:
            self._close_live_stt()   # never leak a previous turn's socket
            session = maker(sr)
            session.start()
            self._live_stt = session
            return session
        except Exception as e:
            log.warning("live STT unavailable (%s) — using batch path", e)
            return None

    def _tee_to_live(self, frames, live):
        """Pass mic frames through to the endpointer while also feeding the
        live recogniser. push() never blocks or raises, so the capture loop's
        timing is unaffected.

        Room tone is held back rather than streamed: frames go into a short
        ring until speech actually starts, then the ring is flushed so the
        first word still has its lead-in. Waiting silently for someone to
        speak used to cost ~960 kbps to a machine across the internet."""
        import numpy as _np

        def _rms(f):
            return float(_np.sqrt(_np.mean(_np.asarray(f, dtype=_np.float32) ** 2)))

        # Well below the VAD's own start gate: this only decides when the wire
        # opens, so erring open costs a little bandwidth, while erring closed
        # would clip the start of a sentence.
        gate = max(40.0, self.cfg.get("vad.energy_threshold", 150) * 0.4)
        pre: list = []
        speaking = False
        pad = max(1, self.cfg.get("vad.speech_pad_ms", 200)
                  // max(1, self.cfg.get("audio.block_ms", 30)))
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

    def _maybe_speculate(self, partial: str, turn_p) -> None:
        """Start generating a reply BEFORE the person finishes, when the turn
        model is confident they're done and the words already read like a
        complete request. Nothing is spoken from this — commit_speculation()
        decides whether it survives."""
        if not self.cfg.get("llm.speculative_reply", True):
            return
        if turn_p is None or turn_p < self.cfg.get(
                "llm.speculative_reply_threshold", 0.8):
            log.debug("no speculation: turn_p=%s below threshold", turn_p)
            return
        if not worth_speculating(partial):
            log.debug("no speculation: partial not bet-worthy (%r)", partial)
            return
        if self._speculation is not None:
            if self._speculation.matches(partial):
                return  # already betting on exactly this
            self._speculation.abandon()  # they said more — the old bet is void
            self._speculation = None
        try:
            messages = self._attach_vision(
                [*self.convo.messages(), {"role": "user", "content": partial}],
                partial)
            chunks, stop = self._stream_in_background(messages)
            self._speculation = Speculation(partial, chunks, stop)
            log.info("speculating on %r (p=%.2f)", partial, turn_p)
        except Exception as e:  # a failed bet must never break the turn
            log.debug("speculation failed to start: %s", e)
            self._speculation = None

    def _take_speculation(self, final_text: str):
        """The gate. Returns (chunks, stop) only when the finished sentence is
        word-for-word what we bet on; otherwise the bet is killed and None is
        returned so the caller generates normally. Closed by default — a reply
        to a question that changed is worse than any delay."""
        spec, self._speculation = self._speculation, None
        if spec is None:
            return None
        if spec.matches(final_text):
            log.info("speculation HIT — reply already generating")
            return spec.chunks, spec.stop
        log.info("speculation miss (bet on %r, heard %r) — discarded",
                 spec.text, final_text)
        spec.abandon()
        return None

    def _maybe_backchannel(self, audio_i16, turn_p) -> None:
        """Active listening: at a brief pause deep inside a LONG user turn that
        is clearly not finished, murmur a soft ack ("Mm-hmm.") the way a human
        listener signals "I'm with you, keep going". Gated hard: long turn,
        cooldown, and the turn model must say the thought is NOT complete —
        backchanneling into a finished question is talking over the answer.
        Plays with the mic muted so its own echo never enters the capture."""
        if self.text_mode:
            return
        cfg = self.cfg
        if not cfg.get("conversation.backchannel.enabled", True):
            return
        if turn_p is None or turn_p > cfg.get(
                "conversation.backchannel.max_turn_prob", 0.4):
            return
        sr = cfg.get("audio.sample_rate", 16000)
        min_s = cfg.get("conversation.backchannel.min_speech_ms", 4000) / 1000.0
        if audio_i16.size < sr * min_s:
            return
        cooldown = cfg.get("conversation.backchannel.cooldown_ms", 8000) / 1000.0
        if time.monotonic() - self._last_backchannel < cooldown:
            return
        audios = self._fillers.get("ack") or []
        if not audios:
            return
        self._last_backchannel = time.monotonic()
        try:
            self.mic.pause()   # our own murmur must not enter the capture
            clip = random.choice(audios)
            self.speaker.play(clip * 0.6, self.voice.sample_rate,
                              should_stop=lambda: False)
        except Exception as e:  # a failed murmur must never break the capture
            log.debug("backchannel failed: %s", e)
        finally:
            self.mic.resume()

    def _confirm_voice(self, speech, stop):
        """Collect the rest of the interruption and report how much of it was
        actually speech. Returns (frames, voiced_ratio), or None if playback
        ended while we were listening. This runs BEFORE any ducking, so a
        noise burst never touches the reply volume."""
        frames = list(speech.frames)
        block_ms = self.cfg.get("audio.block_ms", 30)
        need_quiet = max(1, int(350 / block_ms))
        cap = max(1, int(4000 / block_ms))
        # Judge only the frames from the trigger onward — the ring's lead-in is
        # room tone and would drag the ratio down.
        judged = voiced = 0
        quiet = 0
        for frame in self.mic.frames(stop=stop):
            if stop.is_set():
                return None
            frames.append(frame)
            is_voice = self.endpointer.is_speech_frame(frame)
            judged += 1
            voiced += 1 if is_voice else 0
            quiet = 0 if is_voice else quiet + 1
            if quiet >= need_quiet or len(frames) >= cap:
                break
        if stop.is_set():   # frames() ended because the monitor was stopped
            return None
        return frames, (voiced / judged if judged else 0.0)

    def _assess_interruption(self, speech, stop) -> bool:
        """Sustained speech over the reply. React like a person, in order:
        DUCK (drop the reply's volume — the audible "I noticed you"), collect
        the rest of the interruption, transcribe it NOW, and classify:

        * correction  -> cut the reply mid-word (they said we're wrong)
        * new thought -> finish the current sentence, then answer it (queued)
        * backchannel -> un-duck and keep talking (they were agreeing)
        * noise       -> un-duck and keep talking

        Returns True when playback is being stopped (the monitor exits) —
        False re-arms the detector and keeps watching.

        VOICE-GATED, not energy-gated. In a quiet room a loudness+VAD trigger
        is fine; in an exhibition hall a cough, a chair or a passing group
        clears it constantly, and every one of those audibly ducked the reply
        (the log showed `interruption '' -> noise`). So before touching the
        volume we hold a short confirmation window and require the frames to
        be genuinely speech-shaped. Non-speech never reaches the duck."""
        confirm = self._confirm_voice(speech, stop)
        if confirm is None:
            return True          # playback ended under us
        frames, voiced_ratio = confirm
        min_ratio = self.cfg.get("conversation.barge_in_voiced_ratio", 0.55)
        if voiced_ratio < min_ratio:
            # INFO, not debug: a venue session where "none of my interruptions
            # worked" had no way to show WHERE they died. Every vetoed trigger
            # must leave a visible trace.
            log.info("barge-in ignored: only %.0f%% voiced (need %.0f%%) — "
                     "noise, not speech", voiced_ratio * 100, min_ratio * 100)
            return False         # the reply carries on untouched
        # NO dip, at all. Tried 0.35 and 0.75; both made the sentence ZERO is
        # finishing trail off quieter at exactly the moment it should sound
        # normal — a queued interruption means "I will answer you after this
        # thought", and people do not fade out when someone starts talking.
        # Volume that MEANS something ("talk quietly") is handled separately
        # and is unaffected.
        import numpy as np

        sr = self.cfg.get("audio.sample_rate", 16000)
        audio = np.concatenate(frames).astype("float32") / 32768.0
        text, stt_failed = "", False
        try:
            with self._stt_lock:
                text = (self.stt.transcribe(audio, sr) or "").strip()
        except Exception as e:
            stt_failed = True
            log.debug("interrupt stt failed: %s", e)
        kind = classify_interrupt(text)
        log.info("interruption %r -> %s", text, kind.value)
        if kind is InterruptKind.BACKCHANNEL:
            # Not an interruption — but not nothing either: record it so the
            # next turn's context knows they were engaged (humans calibrate on
            # exactly this). The reply itself flows on.
            self._overheard.append(text)   # they were agreeing — carry on
            self._restore_level()
            return False
        if kind is InterruptKind.NOISE:
            if stt_failed:
                # STT is down but the detector heard REAL sustained speech —
                # ignoring them would be worse than yielding. Stop politely and
                # let the next turn re-listen with the audio prepended.
                self._bargein_frames = frames
                self._interrupt_note = (
                    "(They interrupted you but you couldn't make out the words "
                    "— ask them to repeat it briefly.)")
                self._soft_stop = True
                return True
            self._restore_level()
            # By here BOTH engines came up empty (FallbackSTT escalates an
            # empty primary on clear speech) — this really is a blip now, not
            # the old single-engine miss that ate real interruptions.
            log.info("barge-in dismissed: %.0f%% voiced but both STT engines "
                     "heard nothing", voiced_ratio * 100)
            return False   # a transcribed nothing — hallucinated blip
        self._queued_turn = {"text": text, "frames": frames, "kind": kind.value}
        if kind is InterruptKind.CORRECTION:
            self._interrupt_note = (
                "(They cut you off with a correction — what you were saying "
                "missed the mark. Don't defend or restart it; address the "
                "correction directly.)")
            self._interrupt = True    # hard: playback cuts mid-word
        else:  # QUEUE — the polite path
            self._interrupt_note = (
                "(They said this while you were finishing your last sentence; "
                "you stopped at the end of it. If it supports what you were "
                "saying, weave it in and continue your thought; if it changes "
                "direction, follow them. Don't mention the overlap.)")
            self._soft_stop = True    # finish the sentence, then yield
        return True

    def _start_bargein(self):
        """Keep the mic live during playback and interrupt on EITHER the wake
        word OR sustained user speech over the reply (echo-aware, no wake word
        needed — natural interruption). Speech interrupts are CLASSIFIED
        (_assess_interruption) instead of blindly cutting playback. Returns
        (stop_event, thread) — both None if barge-in is off / no mic."""
        self._interrupt = False
        self._soft_stop = False
        self._bargein_frames = None
        if self.text_mode or not self.cfg.get("conversation.barge_in", True):
            return None, None
        self.mic.resume()
        self.mic.drain()   # drop the tail of our own audio captured a moment ago
        _bi_room = getattr(self, "room", None)
        speech = None
        if self.cfg.get("conversation.barge_in_on_speech", True):
            from zero.audio.bargein import SpeechBargeIn

            speech = SpeechBargeIn(
                is_speech=self.endpointer.is_speech_frame,
                block_ms=self.cfg.get("audio.block_ms", 30),
                learn_ms=self.cfg.get("conversation.barge_in_learn_ms", 900),
                trigger_ms=self.cfg.get("conversation.barge_in_speech_ms", 300),
                ratio=self.cfg.get("conversation.barge_in_ratio", 1.6),
                # Scaled to the room: in a hall the crowd itself would clear
                # a fixed gate, so the bar rises with the ambient floor.
                min_rms=(_bi_room.gate(
                    self.cfg.get("conversation.barge_in_min_rms", 250))
                    if _bi_room is not None
                    else self.cfg.get("conversation.barge_in_min_rms", 250)),
                # Keep only the interrupting words (trigger window + lead-in),
                # NOT 1.5s of ring — the ring's prefix is ZERO's own reply
                # echo, which garbled the post-interrupt turn.
                keep_ms=self.cfg.get("conversation.barge_in_speech_ms", 300) + 600,
                # The Bluetooth-proof echo test: the mic envelope is correlated
                # against what the speaker ACTUALLY played (at any lag), so the
                # reply's own echo — however loud — can't fire a false trigger.
                played_env=self.speaker.env_snapshot,
                env_corr_max=self.cfg.get("conversation.barge_in_env_corr", 0.65),
                floor_percentile=self.cfg.get(
                    "conversation.barge_in_floor_percentile", 80),
                afterthought_ms=self.cfg.get(
                    "conversation.afterthought_ms", 350),
                gate_ceiling=self.cfg.get(
                    "conversation.barge_in_gate_ceiling", 1200),
            )
        stop = threading.Event()

        def monitor():
            # wake.reset() measured 521ms — openWakeWord rebuilds its state.
            # Called inline it was 31% of end-to-end latency, sitting in front
            # of every reply for no benefit (the wake word is not needed to
            # continue a conversation). Doing it here moves the cost onto this
            # thread, where it overlaps the reply instead of delaying it.
            try:
                self.wake.reset()
            except Exception as e:
                log.debug("wake reset failed: %s", e)
            # ASYMMETRIC wake threshold while a reply is playing. A live
            # session logged wake scores 0.33 and 0.41 against 0.50 — someone
            # said the wake word to cut ZERO off and was ignored. During a
            # reply a false wake merely stops ZERO talking (cheap), while a
            # miss reads as being ignored (expensive) — so the bar drops here
            # and only here. Restored before the monitor exits; the monitor is
            # joined before the idle loop ever runs wake.process again.
            base_thr = getattr(self.wake, "threshold", None)
            if base_thr is not None:
                self.wake.threshold = min(base_thr, self.cfg.get(
                    "conversation.barge_in_wake_threshold", 0.35))
            # stop-aware frames(): the monitor must exit even when the mic is
            # paused (empty queue) — a plain frames() blocked forever there,
            # outlived the join in _stop_bargein, and then fought the next
            # turn's endpointer for the one shared frame queue.
            try:
                for frame in self.mic.frames(stop=stop):
                    if stop.is_set():
                        return
                    try:
                        if self.wake.process(frame):
                            self._interrupt = True  # wake word: always a hard stop
                            return
                        if speech is None:
                            continue
                        if speech.update(frame, active=self.speaker.playing):
                            if self._assess_interruption(speech, stop):
                                return
                            speech.rearm()   # backchannel/noise: keep watching
                            continue
                        # A remark that finished in the think-gap (before any
                        # reply audio): transcribe it now so the merge points
                        # find text. On a WORKER thread — transcription is a
                        # network call that can block for seconds, and a
                        # monitor stuck inside it can't see stop and misses
                        # its exit window ("barge-in monitor did not exit in
                        # time").
                        frames_af = speech.take_afterthought()
                        if frames_af:
                            threading.Thread(
                                target=self._transcribe_afterthought,
                                args=(frames_af,), name="afterthought-stt",
                                daemon=True).start()
                    except Exception as e:  # never let the monitor crash playback
                        log.debug("barge-in monitor error: %s", e)
                        return
            finally:
                if base_thr is not None:
                    self.wake.threshold = base_thr

        thread = threading.Thread(target=monitor, name="bargein", daemon=True)
        thread.start()
        return stop, thread

    def _stop_bargein(self, bargein) -> None:
        stop, thread = bargein
        if stop is None:
            return
        stop.set()
        # Join the monitor WHILE the mic is still live: frames keep arriving, so
        # the monitor pulls one more, sees the stop flag and exits. If we paused
        # first, it would block forever in frames() (no frames while paused) and
        # then steal the NEXT turn's audio off the shared queue — which made ZERO
        # go deaf after the first reply. Only after it's dead do we mute.
        if thread is not None:
            thread.join(timeout=1.5)
            if thread.is_alive():
                log.warning("barge-in monitor did not exit in time — mic contention")
        self._restore_level()   # back to asked-for x room, never a bare 1.0
        self.mic.pause()

    def _speak_one(self, text: str) -> None:
        """Synthesize + play a single fixed phrase (e.g. the goodbye line)."""
        audio = self.voice.synthesize(text)
        if getattr(audio, "size", 0):
            self.speaker.play(audio, self.voice.sample_rate, should_stop=lambda: False)

    # -- guided enrolment ---------------------------------------------------
    _ENROLL_POSES = (
        "Okay, look straight at me.",
        "Great — now turn your head a little to your left.",
        "And a little to your right.",
        "Perfect. Now look back at me.",
    )

    def _guided_enroll(self, name: str, first_audio=None) -> str:
        """Capture several FACE samples across head poses (plus one VOICE sample
        from the triggering utterance), narrating each step. More varied samples
        = far more robust recognition than a single frontal shot. Returns the
        spoken summary line; the caller speaks it and logs the turn."""
        if self.identity is None or self.eyes is None:
            # Voice-only fallback: no camera, just save the voice sample.
            result, kinds = self.identity.enroll(name, audio=first_audio) \
                if self.identity is not None else (None, [])
            if kinds:
                self._person = result
                return f"Got it, {name}. I've saved your voice — I'll know you next time."
            return (f"I'd love to remember you, {name}, but I can't see or hear "
                    "you clearly right now. Let's try again in a moment.")

        faces = 0
        voice_ok = False
        for i, prompt in enumerate(self._ENROLL_POSES):
            self._to(State.SPEAKING)
            self._speak_one(prompt)
            time.sleep(0.7)  # let them move and the camera catch a fresh frame
            frame = self.eyes.current_frame()
            audio = first_audio if i == 0 else None  # voice once, from their ask
            try:
                _result, kinds = self.identity.enroll(name, audio=audio,
                                                      frame_rgb=frame)
            except Exception as e:  # enrolment must never crash the turn
                log.warning("guided enroll step failed: %s", e)
                kinds = []
            faces += 1 if "face" in kinds else 0
            voice_ok = voice_ok or ("voice" in kinds)

        # Refresh who ZERO thinks it's looking at, so the rest of the
        # conversation is attributed to the freshly enrolled person.
        try:
            ident = self.identity.identify(frame_rgb=self.eyes.current_frame())
            if ident.is_known:
                self._person = ident
        except Exception:
            pass

        if faces == 0 and not voice_ok:
            return (f"Hmm, I couldn't get a clear look at you, {name}. Let's try "
                    "again in better light, facing the camera.")
        got = []
        if faces:
            got.append(f"your face from {faces} angle" + ("s" if faces != 1 else ""))
        if voice_ok:
            got.append("your voice")
        return (f"Got it, {name}. I've saved {' and '.join(got)} — "
                "I'll recognise you next time.")

    # -- external control (the AF1 fusion surface — zero/control.py) ---------
    # These run an AF1 push-to-talk turn through the SAME pipeline the native
    # mic loop uses: same Conversation, same memory, same tool registry
    # (web_search, timers, remember/recall), same voice + Pi speaker. AF1 gets
    # everything ZERO has because it IS ZERO answering.

    def _abort_if_already_running(self) -> None:
        """Refuse to start when a live ZERO already answers on the control port.

        Two instances share one mic and one speaker: both answer every wake
        word (you hear the reply twice, interleaved) and their concurrent
        Orpheus streams crash the CUDA backend — the in-process synthesis
        lock can't serialize across processes. So a confirmed duplicate is
        fatal, not a warning. A busy port that ISN'T ZERO still falls through
        to _start_control's existing warn-and-continue path.
        """
        if not self.cfg.get("control.enabled", False):
            return
        import json
        import urllib.request

        port = self.cfg.get("control.port", 8090)
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2) as r:
                j = json.loads(r.read().decode("utf-8", "replace"))
        except Exception:
            return  # nothing (or something non-ZERO) on the port
        if isinstance(j, dict) and j.get("service") == "zero-control":
            raise SystemExit(
                f"Another ZERO instance is already running (control port "
                f"{port} answered /health). Two instances double-speak every "
                "reply and crash the GPU TTS. Stop it first:\n"
                "  sudo systemctl stop zero    # if it's the service unit\n"
                "  pkill -f zero.main          # if it's a stray terminal run")

    def _start_control(self) -> None:
        if not self.cfg.get("control.enabled", False):
            return
        try:
            from zero.control import ControlServer

            self._control = ControlServer(
                self,
                host=self.cfg.get("control.host", "0.0.0.0"),
                port=self.cfg.get("control.port", 8090))
            self._control.start()
        except Exception as e:  # the robot must run even if the surface can't
            log.warning("control server failed to start: %s", e)
            self._control = None

    def external_status(self) -> dict:
        return {
            "ok": True,
            "state": str(self.state.value),
            "ready": bool(self._control is not None and self._control.ready),
            "last": dict(self._last_ext),
            "voice_degraded": (bool(getattr(self.voice, "degraded", False))
                               if not self.text_mode else None),
            "stt_degraded": (bool(getattr(self.stt, "degraded", False))
                             if not self.text_mode else None),
        }

    def _remote_tts_engine(self):
        """Walk the voice wrapper chain (orchestrator → fallback → engine) to
        the RemoteTTS whose `.voice` is the per-request Orpheus speaker name."""
        obj = getattr(self, "voice", None)
        for _ in range(5):
            if obj is None:
                return None
            if type(obj).__name__ == "RemoteTTS":
                return obj
            obj = getattr(obj, "tts", None) or getattr(obj, "primary", None)
        return None

    @contextlib.contextmanager
    def _voice_override(self, voice: str | None):
        """Speak THIS request in a different Orpheus voice, then restore ZERO's
        own. AF1's picker keeps its whole roster without touching the default."""
        eng = self._remote_tts_engine() if voice else None
        if eng is None or not voice or getattr(eng, "voice", None) == voice:
            yield
            return
        old = eng.voice
        eng.voice = str(voice)
        try:
            yield
        finally:
            eng.voice = old

    def _wait_not_busy(self, timeout_s: float = 20.0) -> bool:
        """Hold external requests off the native loop's think/speak phase so
        two replies never talk over each other on the one speaker."""
        deadline = time.monotonic() + timeout_s
        while self.state in (State.THINKING, State.SPEAKING):
            if time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        return True

    def external_say(self, text: str, voice: str | None = None) -> None:
        """Speak one line on the Pi speaker (AF1 announcements/callouts)."""
        if self.text_mode:
            log.info("external say (text mode, not spoken): %r", text)
            return
        with self._ext_lock:
            self._wait_not_busy()
            self.mic.pause()  # don't transcribe our own voice off the speaker
            try:
                with self._voice_override(voice):
                    self._speak_one(text)
            finally:
                self.mic.resume()
                self.mic.drain()

    def external_turn_audio(self, audio, sr: int, voice: str | None = None,
                            person_id: int | None = None) -> dict:
        """Full turn from an AF1-recorded utterance: STT → brain → Pi speaker."""
        try:
            with self._stt_lock:
                text = self.stt.transcribe(audio, sr).strip()
        except Exception as e:
            log.warning("external stt failed: %s", e)
            return {"ok": False, "error": f"stt: {str(e)[:120]}"}
        if not text:
            return {"ok": False, "error": "empty-transcript"}
        out = self.external_turn_text(text, voice=voice, person_id=person_id)
        out["heard"] = text
        return out

    def external_turn_text(self, text: str, voice: str | None = None,
                           speak: bool = True,
                           person_id: int | None = None) -> dict:
        """One brain turn — identical semantics to a native mic turn: shared
        history, vision note, memory recall, tools/web_search via llm.stream,
        streamed TTS on the Pi speaker. Returns {ok, heard, reply}."""
        with self._ext_lock:
            if not self._wait_not_busy():
                return {"ok": False, "error": "busy-speaking"}
            # First external turn of a session: load durable memory the same
            # way _converse does, so recall works even before any native chat.
            if self.memory is not None and not getattr(self.convo,
                                                       "_memory_block", ""):
                self.convo.set_memory(self.memory.as_block())
            self.convo.add_user(text)
            self._t_reply_start = time.monotonic()
            messages = self._attach_vision(self.convo.messages(), text)
            do_speak = speak and not self.text_mode
            if do_speak:
                self.mic.pause()
            try:
                chunks, llm_stop = self._stream_in_background(messages)
                if do_speak:
                    self._interrupt = False
                    self._soft_stop = False  # a stale soft stop would mute this
                    with self._voice_override(voice):
                        reply = self._speak_streaming(chunks, llm_stop)
                else:
                    reply = "".join(chunks).strip()
            except Exception as e:
                log.exception("external turn failed")
                return {"ok": False, "error": str(e)[:200], "heard": text}
            finally:
                if do_speak:
                    self.mic.resume()
                    self.mic.drain()
            reply = (reply or "").strip()
            if not reply:
                return {"ok": False, "error": "empty-reply", "heard": text}
            self.convo.add_assistant(reply)
            pid = (person_id if person_id is not None
                   else self.cfg.get("control.person_id", 1))
            if self.memory is not None:
                try:  # remembered like an AF1 episode — one mind, both surfaces
                    self.memory.add_episode(
                        f"(via AF1) they said: {text} — I replied: {reply}",
                        person_id=pid)
                except Exception as e:
                    log.debug("external episode save failed: %s", e)
            self._last_ext = {"heard": text, "reply": reply, "t": time.time()}
            return {"ok": True, "heard": text, "reply": reply}

    def external_sleep(self) -> None:
        """Reset the external conversation thread (AF1's 'sleep' control). The
        native loop owns its own conversation lifecycle — never touch it mid-chat."""
        with self._ext_lock:
            if self.state == State.IDLE:
                self.convo.reset()
                self._last_ext = {}


def _acquire_instance_lock():
    """Refuse to start a second voice instance. Two ZEROs on one box both
    answer the mic (double voice) and their concurrent TTS requests crash the
    Orpheus CUDA backend. flock releases automatically however the process
    dies, so a stale lock file can never block a fresh start."""
    import fcntl

    lock = open("/tmp/zero-main.lock", "w")  # noqa: SIM115 — held for process lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another ZERO instance is already running (voice would double up).\n"
              "find it with:  pgrep -af zero.main", file=sys.stderr)
        raise SystemExit(1)
    lock.write(str(os.getpid()))
    lock.flush()
    return lock  # keep the handle alive; closing it would drop the lock


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="zero",
        description="ZERO — offline conversational voice robot.",
    )
    parser.add_argument("config", nargs="?", default=None,
                        help="path to a config YAML (default: config.yaml)")
    parser.add_argument("--text", action="store_true",
                        help="type-to-chat mode: LLM + memory only, no mic/speaker")
    args = parser.parse_args()
    setup_logging()
    if not args.text:
        globals()["_instance_lock"] = _acquire_instance_lock()
    zero = Zero(args.config, text_mode=args.text)
    if args.text:
        zero.run_text()
    else:
        zero.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
