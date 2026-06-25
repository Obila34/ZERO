"""ZERO entry point — the IDLE -> LISTENING -> THINKING -> SPEAKING loop.

Run on the Pi (after scripts/setup_pi.sh) with:
    python -m zero.main

The loop streams the LLM reply sentence-by-sentence and speaks each sentence as
soon as it's ready, so the first words come out while the rest is still being
generated — the main trick for keeping a fully-local pipeline feeling responsive.
"""
from __future__ import annotations

import itertools
import queue
import random
import sys
import threading
import time

from zero.config import load_config
from zero.conversation import Conversation
from zero.factory import (
    build_endpointer, build_llm, build_memory, build_stt, build_voice, build_wake,
)
from zero.llm.persona import build_system_prompt
from zero.audio.capture import MicCapture
from zero.audio.playback import Speaker
from zero.state import State, can_transition
from zero.tts.orchestrator import split_sentences
from zero.utils.logging import get_logger, setup_logging

log = get_logger("main")


class Zero:
    def __init__(self, config_path: str | None = None):
        self.cfg = load_config(config_path)
        sr = self.cfg.get("audio.sample_rate", 16000)

        # Audio I/O (one shared mic for the whole pipeline).
        self.mic = MicCapture(
            sample_rate=sr,
            block_ms=self.cfg.get("audio.block_ms", 30),
            device=self.cfg.get("audio.input_device"),
        )
        self.speaker = Speaker(device=self.cfg.get("audio.output_device"))

        # Pipeline stages (engines chosen in config.yaml).
        log.info("loading engines...")
        self.wake = build_wake(self.cfg)
        self.endpointer = build_endpointer(self.cfg)
        self.stt = build_stt(self.cfg)
        self.llm = build_llm(self.cfg)
        self.voice = build_voice(self.cfg)

        system_prompt = build_system_prompt(
            name=self.cfg.get("persona.name", "Zero"),
            description=self.cfg.get("persona.description", "a warm companion."),
        )
        self.convo = Conversation(
            system_prompt=system_prompt,
            history_turns=self.cfg.get("llm.history_turns", 3),
            trim_at_turns=self.cfg.get("llm.history_trim_at", 8),
        )
        self.memory = build_memory(self.cfg)  # long-term SQLite store (or None)
        self.state = State.IDLE

        # Context-aware "thinking" fillers, pre-synthesized per category so we can
        # play the RIGHT one instantly (no synth cost) while the reply generates in
        # the background. Category is chosen from what the user just said.
        self._filler_prob = self.cfg.get("conversation.filler_probability", 0.9)
        self._fillers = self._presynth_fillers()

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

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        # Pin the LLM in RAM now so the first real reply isn't a ~28s cold load.
        warmup = getattr(self.llm, "warmup", None)
        if callable(warmup):
            warmup()
        self.mic.start()
        log.info("ZERO ready. Say the wake word to start talking. (Ctrl-C to quit)")
        try:
            while True:
                self._wait_for_wake()   # IDLE: wait for "Hey Jarvis" to begin
                self._converse()        # free-flowing multi-turn until sleep/stop
        except KeyboardInterrupt:
            print()
            log.info("shutting down")
        finally:
            self.mic.stop()

    def _wait_for_wake(self) -> None:
        self._to(State.IDLE)
        self.wake.reset()
        for frame in self.mic.frames():
            if self.wake.process(frame):
                log.info("wake word! let's talk.")
                return

    # -- conversation -------------------------------------------------------
    # Stop phrases that end the conversation and return to wake-word mode.
    _STOP_PHRASES = (
        "goodbye", "good bye", "go to sleep", "stop listening", "stop talking",
        "that's all", "thats all", "see you later", "bye zero", "shut down",
    )

    def _is_stop(self, text: str) -> bool:
        t = text.lower().strip(" .!?,")
        return any(p in t for p in self._STOP_PHRASES)

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
        sr = self.cfg.get("audio.sample_rate", 16000)
        idle_s = self.cfg.get("conversation.sleep_timeout_ms", 30000) / 1000.0
        log.info("conversation open — just talk (say 'goodbye' to stop)")

        while True:
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
            text = self.stt.transcribe(utterance, sr).strip()
            if not text:
                continue  # misfire / noise — keep listening, stay in conversation

            if self._is_stop(text):
                self._to(State.SPEAKING)
                self._speak_one("Okay, talk to you later.")
                log.info("Sleeping… (stop phrase)")
                self._end_conversation()
                return

            self._maybe_remember(text)  # explicit "remember that ..."

            self.convo.add_user(text)
            self._t_reply_start = time.monotonic()  # end-of-STT marker for timing
            # Kick the LLM off in the BACKGROUND so its prefill overlaps the spoken
            # filler — the model is already generating while the filler plays, so the
            # real answer flows in with little or no added delay.
            chunks = self._stream_in_background(self.convo.messages())
            self._to(State.SPEAKING)
            self._play_filler(text)
            reply = self._speak_streaming(chunks)
            self.convo.add_assistant(reply)
            log.info("reply: %r", reply)

    # -- long-term memory ---------------------------------------------------
    def _maybe_remember(self, text: str) -> None:
        """Explicit 'remember that ...' — store immediately so it survives even if
        the conversation never ends cleanly."""
        if self.memory is None:
            return
        low = text.lower()
        for trigger in ("remember that ", "remember "):
            if low.startswith(trigger):
                fact = text[len(trigger):].strip()
                if fact:
                    self.memory.remember(f"note ({int(time.time())})", fact)
                return

    def _end_conversation(self) -> None:
        """On sleep/stop: extract durable facts from the chat into long-term memory,
        then go idle. Runs an extra (offline) LLM pass — fine, the user is done."""
        if self.memory is not None and self.convo._history:
            try:
                self._extract_facts()
            except Exception as e:  # never let memory break the loop
                log.warning("memory extraction failed: %s", e)
        self._to(State.IDLE)

    def _extract_facts(self) -> None:
        prompt = [
            {"role": "system", "content": (
                "Extract durable facts about the USER from the conversation. Output "
                "only short 'key: value' lines for lasting facts (name, location, "
                "job, preferences, ongoing projects, important personal details). "
                "Use short lowercase keys. If there are no durable facts, output "
                "exactly NONE and nothing else.")},
            {"role": "user", "content": self.convo.transcript()},
        ]
        result = "".join(self.llm.stream(prompt)).strip()
        if not result or result.upper().startswith("NONE"):
            return
        for line in result.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                self.memory.remember(key.strip("-• ").strip(), value)

    def _stream_in_background(self, messages):
        """Start the LLM streaming on a worker thread feeding a queue, and return
        a generator over it. Calling this begins prefill immediately (a bare
        generator would wait until first consumed), so it overlaps the filler.
        """
        q: "queue.Queue" = queue.Queue()

        def worker():
            try:
                for chunk in self.llm.stream(messages):
                    q.put(chunk)
            except Exception as e:  # never let the worker die silently
                log.error("LLM stream error: %s", e)
            finally:
                q.put(None)  # sentinel = done

        threading.Thread(target=worker, daemon=True).start()

        def gen():
            while True:
                item = q.get()
                if item is None:
                    return
                yield item

        return gen()

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
        """Play one pre-synthesized filler chosen to match what the user said."""
        if random.random() > self._filler_prob:
            return
        category = self._filler_category(user_text)
        audios = self._fillers.get(category) or self._fillers.get("default") or []
        if not audios:
            return
        log.debug("filler category: %s", category)
        self.speaker.play(random.choice(audios), self.voice.sample_rate,
                          should_stop=lambda: False)

    def _speak_streaming(self, chunks) -> str:
        """Speak sentences as they complete; return the full reply text."""
        full: list[str] = []
        buffer = ""
        spoke_any = False
        first_token = True

        for chunk in itertools.chain(chunks, ["\n"]):  # sentinel flush
            if first_token and chunk.strip():
                log.info("LLM first token: %.2fs", time.monotonic() - self._t_reply_start)
                first_token = False
            buffer += chunk
            sentences = split_sentences(buffer)
            # Keep the last (possibly incomplete) fragment buffered.
            complete, buffer = sentences[:-1], (sentences[-1] if sentences else "")
            for sentence in complete:
                if not spoke_any:
                    log.info("first audio out: %.2fs after STT",
                             time.monotonic() - self._t_reply_start)
                    self._to(State.SPEAKING)
                    spoke_any = True
                self._speak_one(sentence)
                full.append(sentence)

        if buffer.strip():  # flush trailing fragment
            if not spoke_any:
                self._to(State.SPEAKING)
            self._speak_one(buffer)
            full.append(buffer.strip())

        return " ".join(full).strip()

    def _speak_one(self, sentence: str) -> None:
        audio = self.voice.synthesize(sentence)
        if audio.size:
            # should_stop hook reserved for barge-in (Phase 7); always False now.
            self.speaker.play(audio, self.voice.sample_rate, should_stop=lambda: False)


def main() -> int:
    setup_logging()
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    Zero(config_path).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
