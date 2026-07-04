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
from zero.identity.service import parse_enrollment
from zero.memory.preferences import apply_rate_delta, parse_preference
from zero.tools.base import ToolContext
from zero.vision.learned import parse_object_teach
from zero.llm.persona import build_system_prompt
from zero.state import State, can_transition
from zero.tts.orchestrator import split_sentences
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
            self.speaker = Speaker(device=self.cfg.get("audio.output_device"))
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
        self._memory_thread: threading.Thread | None = None  # background fact save

        # Voice-only extras (need the voice/mic): owner verification + spoken fillers.
        self._filler_prob = self.cfg.get("conversation.filler_probability", 0.5)
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
        for category, phrases in sets.items():
            audios = []
            for phrase in phrases:
                try:
                    audio = self.voice.synthesize(phrase)
                    if getattr(audio, "size", 0):
                        audios.append(audio)
                        total += 1
                except Exception as e:  # never block startup on a filler
                    log.debug("filler synth failed for %r: %s", phrase, e)
            out[category] = audios
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

    # -- text mode ----------------------------------------------------------
    def run_text(self) -> None:
        """Type-to-chat: tests the brain (LLM + memory + conversation) with no mic,
        Piper or Whisper needed. Useful for validating the GPU LLM offload."""
        warmup = getattr(self.llm, "warmup", None)
        if callable(warmup):
            warmup()
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
        except KeyboardInterrupt:
            print()
        self._end_conversation()
        self._join_memory_thread()

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        # Pin the LLM in RAM now so the first real reply isn't a ~28s cold load.
        warmup = getattr(self.llm, "warmup", None)
        if callable(warmup):
            warmup()
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
        sr = self.cfg.get("audio.sample_rate", 16000)
        idle_s = self.cfg.get("conversation.sleep_timeout_ms", 30000) / 1000.0
        log.info("conversation open — just talk (say 'goodbye' to stop)")

        while True:
            self._drain_events()  # timers/reminders land at turn boundaries
            self._to(State.LISTENING)
            self.mic.resume()  # re-open the mic for the user's turn
            self.mic.drain()   # drop audio captured while we were speaking/thinking
            utterance = self.endpointer.capture(self.mic.frames(), idle_timeout_s=idle_s)

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

            # Who is speaking? Face (current frame) + voice (this utterance),
            # fused. Local + fast; never on the critical path of a failure.
            frame = self.eyes.current_frame() if self.eyes is not None else None
            self._turn_notes = []
            if self.identity is not None:
                ident = self.identity.identify(audio=utterance, frame_rgb=frame)
                self._person = ident if ident.is_known else None
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

            # Affect: a coarse read of tone (voice prosody + words + face).
            # Only notable, confident reads reach the LLM — never a guess.
            if self.affect is not None:
                mood = self.affect.estimate(utterance, sr, text, frame_rgb=frame)
                if mood.notable:
                    self._turn_notes.append(
                        f"(The speaker sounds {mood.label} right now — "
                        "let that shape your tone, don't comment on it.)")

            # Self-state narration: when a stage is running on its fallback,
            # ZERO knows — and can be upfront instead of sounding "off".
            self._turn_notes.extend(self._self_state_notes())

            # Conversational enrolment: "I'm David" / "my name is David" —
            # capture this moment's face + voice and remember the person.
            if self.identity is not None:
                new_name = parse_enrollment(text)
                if new_name:
                    result, kinds = self.identity.enroll(
                        new_name, audio=utterance, frame_rgb=frame)
                    if kinds:
                        self._person = result
                        senses = " and ".join(kinds)
                        line = (f"Nice to meet you, {new_name}. "
                                f"I'll remember your {senses}.")
                    else:
                        line = (f"Nice to meet you, {new_name} — but I couldn't "
                                "get a good look or listen, so tell me again "
                                "in a moment.")
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
            # Barge-in covers the filler AND the reply, so the wake word can cut
            # ZERO off at any point while it's making sound.
            monitor = self._start_bargein()
            try:
                self._play_filler(text)
                reply = "" if self._interrupt else self._speak_streaming(chunks, llm_stop)
            finally:
                self._stop_bargein(monitor)
            if self._interrupt:
                llm_stop.set()  # stop generating a reply nobody is listening to
                log.info("barge-in: stopped speaking to listen")
            # Store only what was actually SPOKEN — after a barge-in the model must
            # not "remember" saying sentences the user never heard.
            if reply:
                self.convo.add_assistant(reply)
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
        id_note = (
            f"(You recognise the speaker — it's {ident.name}.)"
            if ident is not None and ident.is_known else ""
        )
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
        if not (note or images or id_note or recall_note or turn_notes):
            return messages
        last = dict(messages[-1])
        extras = []
        if id_note:
            extras.append(id_note)
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

    def _end_conversation(self) -> None:
        """On sleep/stop: extract durable facts from the chat into long-term memory
        (two extra LLM passes) on a BACKGROUND thread, so ZERO goes back to
        listening for the wake word immediately instead of being deaf for seconds.
        The transcript is snapshotted first — the next conversation may reset the
        history while the save is still running."""
        if (self.memory is not None and self.convo._history
                and getattr(self, "_memory_allowed", True)):
            transcript = self.convo.transcript()
            pid = self._person.person_id if self._person is not None else None
            self._memory_thread = threading.Thread(
                target=self._save_memories, args=(transcript, pid),
                name="memory-save", daemon=True,
            )
            self._memory_thread.start()
        self._person = None  # identity does not persist across conversations
        self._to(State.IDLE)

    def _save_memories(self, transcript: str, person_id: int | None = None) -> None:
        try:
            self._extract_facts(transcript, person_id)
        except Exception as e:  # never let memory break the loop
            log.warning("memory extraction failed: %s", e)
        try:
            self._summarize_session(transcript, person_id)  # episodic memory
        except Exception as e:
            log.warning("session summary failed: %s", e)
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
                           person_id: int | None = None) -> None:
        """Write a one-line summary of the conversation into episodic memory, so
        next time ZERO can recall the ARC of past chats, not just isolated facts."""
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

    def _play_filler(self, user_text: str) -> None:
        """Play one pre-synthesized filler chosen to match what the user said.
        Interruptible: the barge-in monitor is already running, so the wake word
        cuts a filler off just like it cuts off a reply."""
        if random.random() > self._filler_prob:
            return
        category = self._filler_category(user_text)
        audios = self._fillers.get(category) or self._fillers.get("default") or []
        if not audios:
            return
        log.debug("filler category: %s", category)
        self.speaker.play(random.choice(audios), self.voice.sample_rate,
                          should_stop=self._should_interrupt)

    def _speak_streaming(self, chunks, llm_stop: threading.Event) -> str:
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
            while True:
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
        """Run the wake detector on the live mic during playback. Returns
        (stop_event, thread) — both None if barge-in is off / no mic."""
        self._interrupt = False
        if self.text_mode or not self.cfg.get("conversation.barge_in", True):
            return None, None
        self.mic.resume()
        self.mic.drain()   # drop the tail of our own audio captured a moment ago
        self.wake.reset()
        stop = threading.Event()

        def monitor():
            for frame in self.mic.frames():
                if stop.is_set():
                    return
                try:
                    if self.wake.process(frame):
                        self._interrupt = True
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
