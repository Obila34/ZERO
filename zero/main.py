"""ZERO entry point — the IDLE -> LISTENING -> THINKING -> SPEAKING loop.

Run on the Pi (after scripts/setup_pi.sh) with:
    python -m zero.main

The loop streams the LLM reply sentence-by-sentence and speaks each sentence as
soon as it's ready, so the first words come out while the rest is still being
generated — the main trick for keeping a fully-local pipeline feeling responsive.
"""
from __future__ import annotations

import argparse
import itertools
import queue
import random
import re
import threading
import time

from zero.config import load_config
from zero.conversation import Conversation
from zero.events import EventBus
from zero.factory import (
    build_endpointer, build_identity, build_llm, build_memory, build_perception,
    build_privacy, build_proactive, build_stt, build_tools, build_vision,
    build_voice, build_voiceid, build_wake,
)
from zero.privacy.guard import parse_forget_command
from zero.identity.service import parse_enroll_command, parse_enrollment
from zero.memory.preferences import apply_rate_delta, parse_preference
from zero.perception.affect import MoodTracker
from zero.tools.base import ToolContext
from zero.vision.learned import parse_object_teach
from zero.llm.persona import build_system_prompt
from zero.state import State, can_transition
from zero.tts.orchestrator import split_sentences, strip_asides
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
    if t.endswith((",", "-", "—", ":")):
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
        sr = self.cfg.get("audio.sample_rate", 16000)

        # The LLM is all text mode needs. Audio capture, wake word, STT and the
        # voice are only built for the full voice pipeline — so text mode runs on
        # a box with no mic/Piper/whisper installed (just to test brain + memory).
        log.info("loading engines...")
        self.events = EventBus()
        self.memory = build_memory(self.cfg)  # long-term SQLite store (or None)
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
            self.speaker = Speaker(
                device=self.cfg.get("audio.output_device"),
                echo_ref=echo_ref,
                prebuffer_ms=self.cfg.get("tts.orpheus.prebuffer_ms", 0))
            self.wake = build_wake(self.cfg)
            self.endpointer = build_endpointer(self.cfg)
            self.stt = build_stt(self.cfg)
            self.voice = build_voice(self.cfg)
            # Eyes: always-on camera + detection (or None if vision disabled /
            # the camera stack isn't installed). Started in run().
            self.eyes = build_vision(self.cfg)
        else:
            self.eyes = None

        tool_block = (self.tool_registry.spec_block()
                      if self.tool_registry is not None else "")
        lang_block = ""
        if self.cfg.get("conversation.multilingual", True):
            lang_block = (
                "Language: if the person speaks Swahili, or mixes Swahili and "
                "English (Sheng-style), reply in the same language or mix, "
                "naturally. Never point out the switch — just flow with it.")
        system_prompt = build_system_prompt(tool_block, lang_block)
        self.convo = Conversation(
            system_prompt=system_prompt,
            history_turns=self.cfg.get("llm.history_turns", 3),
            trim_at_turns=self.cfg.get("llm.history_trim_at", 8),
        )
        self.state = State.IDLE
        self._interrupt = False  # set by the barge-in monitor to stop playback
        self._bargein_frames = None   # the interrupting words, fed to the next STT
        self._was_interrupted = False  # tell the LLM it got cut off, once
        self._mood = MoodTracker()     # cross-turn emotional state
        self._stt_lock = threading.Lock()  # serialize speculative vs final STT
        self._memory_thread: threading.Thread | None = None  # background fact save
        self._summary_thread: threading.Thread | None = None  # rolling compaction
        self._face_name = None         # who the camera recognises (log/perception only)
        self._last_id_key = None       # last identity note attached (dedup across turns)
        self._turn_durable_pid = None  # speaker to CREDIT durable memory this turn (or None)
        self._session_log: list[tuple] = []  # (durable_pid, role, text) for per-person save
        self._welcomed: set[int] = set()  # people greeted "welcome back" this session

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
            self.privacy, self.indicator = build_privacy(self.cfg)
            self.affect, self.speaker_tracker = build_perception(
                self.cfg, self.identity)
            self.proactive, self.curiosity, self.policy = build_proactive(
                self.cfg, events=self.events, eyes=self.eyes,
                identity=self.identity, memory=self.memory,
                is_idle=lambda: self.state == State.IDLE,
            )
            self._fillers = self._presynth_fillers()
        else:
            self.voiceid, self._voiceprint = None, None
            self.identity = None
            self.privacy, self.indicator = None, None
            self.affect, self.speaker_tracker = None, None
            self.proactive, self.curiosity, self.policy = None, None, None

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
                if event.meta.get("open_conversation"):
                    open_conversation = True
                    # The opener is real dialogue — _converse() folds it into
                    # the fresh history so the LLM knows it said it.
                    self._pending_opener = event.text
            except Exception as e:  # an announcement must never kill the loop
                log.warning("announcement failed: %s", e)
            self._to(prev if prev != State.SPEAKING else State.IDLE)
        return open_conversation

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
        warmup = getattr(self.llm, "warmup", None)
        if callable(warmup):
            warmup(self._warmup_messages())
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
                else:
                    print("zero> (no reply — see log; check the tunnel/model)\n")
                self._maybe_compact()  # rolling summary keeps context bounded
        except KeyboardInterrupt:
            print()
        self._end_conversation()
        self._join_memory_thread()

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        # Pin the LLM in RAM now — with the real prefix, so the first reply
        # pays neither the cold load nor the persona+memory prefill.
        warmup = getattr(self.llm, "warmup", None)
        if callable(warmup):
            warmup(self._warmup_messages())
        self.mic.start()
        # Open the eyes BEFORE the wake loop so perception is already running when
        # the user speaks — the scene is pre-computed, never on the critical path.
        if self.eyes is not None:
            try:
                self.eyes.start()
            except Exception as e:  # a missing camera must not stop voice working
                log.warning("could not start eyes — running voice-only: %s", e)
                self.eyes = None
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
        # A proactive opener ("Hey David, welcome back.") started this
        # conversation — seed it as the first assistant turn.
        opener = getattr(self, "_pending_opener", None)
        if opener:
            self.convo.add_assistant(opener)
            self._pending_opener = None
        if self.speaker_tracker is not None:
            self.speaker_tracker.reset()   # a new conversation, a clean slate
        self._mood.reset()
        self._was_interrupted = False  # a stale note must not leak across chats
        self._face_name = None
        self._last_id_key = None       # fresh conversation: re-introduce who's here
        self._session_log = []         # per-turn (speaker, role, text) for per-person memory
        self._welcomed = set()         # welcome-back fires once per person
        sr = self.cfg.get("audio.sample_rate", 16000)
        idle_s = self.cfg.get("conversation.sleep_timeout_ms", 30000) / 1000.0
        log.info("conversation open — just talk (say 'goodbye' to stop)")

        while True:
            self._drain_events()  # timers/reminders land at turn boundaries
            self._maybe_compact()  # fold trimmed turns into a rolling summary
            self._to(State.LISTENING)
            self.mic.resume()  # re-open the mic for the user's turn
            # On a speech barge-in the interrupting words were already captured —
            # feed them into this capture so the user never repeats themselves.
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
            spec: dict = {}
            # Speculation only pays off when STT is the FAST remote. While it's
            # degraded to the slow local CPU engine, speculating just runs that
            # ~5s job twice (pause + endpoint) — so skip it and transcribe once.
            allow_spec = (self.voiceid is None
                          and not getattr(self.stt, "degraded", False)
                          and not (self.privacy is not None
                                   and getattr(self.privacy, "mode", "") == "strict"))

            def _on_pause(audio_i16):
                res: dict = {}

                def run():
                    with self._stt_lock:
                        try:
                            res["text"] = self.stt.transcribe(
                                audio_i16.astype("float32") / 32768.0, sr)
                        except Exception as e:
                            log.debug("speculative stt failed: %s", e)
                spec["audio"], spec["res"] = audio_i16, res
                spec["thread"] = t = threading.Thread(
                    target=run, name="stt-spec", daemon=True)
                t.start()

            def _hold() -> bool:
                # Semantic endpointing: if the speculative transcript for THIS
                # pause is back and reads mid-thought, hold the endpoint open.
                res = spec.get("res") or {}
                return ends_mid_thought(res.get("text", ""))

            utterance = self.endpointer.capture(
                frames_src, idle_timeout_s=idle_s,
                on_speech_pause=_on_pause if allow_spec else None,
                should_hold=_hold if (allow_spec and self.cfg.get(
                    "vad.semantic_hold", True)) else None)

            if utterance is None or getattr(utterance, "size", 0) == 0:
                log.info("Sleeping… (no speech for %.0fs — say the wake word to talk again)",
                         idle_s)
                self._end_conversation()
                return

            # Mute the mic for the whole think+speak phase so ZERO can't transcribe
            # its own voice off the BT speaker (echo).
            self.mic.pause()
            self._to(State.THINKING)

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
            spec_a = spec.get("audio")
            # Tail bound: more than ~2x the silence window after the spec pause
            # means speech resumed / max-cap fired — the spec text is a prefix.
            slack = int(sr * (2 * self.cfg.get("vad.silence_ms", 450)
                              + self.cfg.get("vad.speech_pad_ms", 200) + 400) / 1000)
            if (spec_a is not None
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
                self._turn_notes.append(
                    "(You were interrupted mid-sentence a moment ago — don't "
                    "restart the old answer; just respond to what they say now.)")
            self._turn_durable_pid = None  # only a CONFIDENT voice credits memory
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
                if ident.is_known:
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
                # Diarization: notice when a DIFFERENT person takes over.
                if self.speaker_tracker is not None:
                    change_note = self.speaker_tracker.update(
                        person_id=ident.person_id if ident.is_known else None,
                        name=ident.name if ident.is_known else None,
                        voice_emb=self.identity.voice_embedding(utterance),
                    )
                    if change_note:
                        self._turn_notes.append(change_note)

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
            else:
                text = self.stt.transcribe(utterance, sr).strip()
            if not text:
                continue  # misfire / noise — keep listening, stay in conversation

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
            if self.affect is not None:
                read = self.affect.estimate(utterance, sr, text, frame_rgb=frame)
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
            # be split per person instead of lumped onto one owner.
            self._session_log.append((self._turn_durable_pid, "user", text))
            self._t_reply_start = time.monotonic()  # end-of-STT marker for timing
            # Fold in what ZERO currently sees as an EPHEMERAL note on THIS turn
            # only — the note + keyframes are attached to the outgoing copy of the
            # messages, never saved to history, so the cached prefix and future
            # turns stay clean and image-free.
            messages = self._attach_vision(self.convo.messages(), text)
            # Kick the LLM off in the BACKGROUND so its prefill overlaps the spoken
            # filler — the model is already generating while the filler plays, so the
            # real answer flows in with little or no added delay.
            chunks, llm_stop = self._stream_in_background(messages)
            self._to(State.SPEAKING)
            # Barge-in covers the filler AND the reply, so speech or the wake
            # word can cut ZERO off at any point while it's making sound.
            monitor = self._start_bargein()
            try:
                # The filler RACES the real reply: it only plays if no reply
                # audio arrived within the grace window, so a fast answer is
                # never delayed by a canned "let me think".
                reply = self._speak_streaming(chunks, llm_stop,
                                              filler_audio=self._pick_filler(text))
            finally:
                self._stop_bargein(monitor)
            if self._interrupt:
                llm_stop.set()  # stop generating a reply nobody is listening to
                self._was_interrupted = True
                log.info("barge-in: stopped speaking to listen")
            # Store only what was actually SPOKEN — after a barge-in the model must
            # not "remember" saying sentences the user never heard.
            if reply:
                self.convo.add_assistant(reply)
                self._session_log.append(
                    (self._turn_durable_pid, "assistant", reply))
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
            if self._is_visual(text):
                ctx = self.eyes.visual_context(question=text)
                return ctx.text, ctx.images
            return self.eyes.local_context(), []
        except Exception as e:  # vision must never break a conversation turn
            log.debug("vision look failed: %s", e)
            return "", []

    def _attach_vision(self, messages: list, text: str) -> list:
        """Attach the live-sight note + keyframes to the FINAL user message of a
        throwaway copy of ``messages``. The persona prompt ("Your eyes") tells the
        model to treat the parenthetical as its own perception, not user text.
        """
        note, images = self._look(text)
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
        recall_note = ""
        if self.memory is not None:
            try:
                pid = ident.person_id if ident is not None else None
                recalled = self.memory.relevant_block(text, person_id=pid)
                if recalled:
                    recall_note = f"(This reminds you of things you know: {recalled}.)"
            except Exception as e:  # recall must never break a turn
                log.debug("recall failed: %s", e)
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
        if (self.memory is not None and self._session_log
                and getattr(self, "_memory_allowed", True)):
            # Sessions are voice-owned: hand over the per-turn (speaker, role, text)
            # log so memory is written PER PERSON, never lumped onto one owner.
            session_log = list(self._session_log)
            self._memory_thread = threading.Thread(
                target=self._save_memories, args=(session_log,),
                name="memory-save", daemon=True,
            )
            self._memory_thread.start()
        self._person = None  # identity does not persist across conversations
        self._to(State.IDLE)

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
                    # Only rescan for sentences when this chunk could complete one.
                    if not any(c in chunk for c in ".!?…\n"):
                        continue
                    sentences = split_sentences(buffer)
                    complete, buffer = sentences[:-1], (sentences[-1] if sentences else "")
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
                if not spoke_any:
                    log.info("first audio out: %.2fs after STT",
                             time.monotonic() - self._t_reply_start)
                    spoke_any = True
                played = idx
                yield piece

        # Barge-in: the monitor (started by the caller) keeps the mic live and
        # listens for the wake word; saying it cuts ZERO off mid-reply (echo-safe:
        # ZERO never says its own wake word, so its voice won't false-trigger).
        self.speaker.play_stream(audio_gen(), self.voice.sample_rate,
                                 should_stop=self._should_interrupt)

        if self._interrupt:
            # Shut the whole pipeline down and unblock the producer, then return
            # only the sentences the user actually heard.
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

    def _start_bargein(self):
        """Keep the mic live during playback and interrupt on EITHER the wake
        word OR sustained user speech over the reply (echo-aware, no wake word
        needed — natural interruption). Returns (stop_event, thread) — both
        None if barge-in is off / no mic."""
        self._interrupt = False
        self._bargein_frames = None
        if self.text_mode or not self.cfg.get("conversation.barge_in", True):
            return None, None
        self.mic.resume()
        self.mic.drain()   # drop the tail of our own audio captured a moment ago
        self.wake.reset()
        speech = None
        if self.cfg.get("conversation.barge_in_on_speech", True):
            from zero.audio.bargein import SpeechBargeIn

            speech = SpeechBargeIn(
                is_speech=self.endpointer.is_speech_frame,
                block_ms=self.cfg.get("audio.block_ms", 30),
                learn_ms=self.cfg.get("conversation.barge_in_learn_ms", 900),
                trigger_ms=self.cfg.get("conversation.barge_in_speech_ms", 300),
                ratio=self.cfg.get("conversation.barge_in_ratio", 1.6),
                min_rms=self.cfg.get("conversation.barge_in_min_rms", 250),
                # Keep only the interrupting words (trigger window + lead-in),
                # NOT 1.5s of ring — the ring's prefix is ZERO's own reply
                # echo, which garbled the post-interrupt turn.
                keep_ms=self.cfg.get("conversation.barge_in_speech_ms", 300) + 600,
            )
        stop = threading.Event()

        def monitor():
            for frame in self.mic.frames():
                if stop.is_set():
                    return
                try:
                    if self.wake.process(frame):
                        self._interrupt = True
                        return
                    if (speech is not None
                            and speech.update(frame,
                                              active=self.speaker.playing)):
                        self._bargein_frames = speech.frames
                        self._interrupt = True
                        log.info("barge-in: user spoke over the reply")
                        return
                except Exception as e:  # never let the monitor crash playback
                    log.debug("barge-in monitor error: %s", e)
                    return

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
    zero = Zero(args.config, text_mode=args.text)
    if args.text:
        zero.run_text()
    else:
        zero.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
