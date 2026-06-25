"""ZERO entry point — the IDLE -> LISTENING -> THINKING -> SPEAKING loop.

Run on the Pi (after scripts/setup_pi.sh) with:
    python -m zero.main

The loop streams the LLM reply sentence-by-sentence and speaks each sentence as
soon as it's ready, so the first words come out while the rest is still being
generated — the main trick for keeping a fully-local pipeline feeling responsive.
"""
from __future__ import annotations

import itertools
import sys
import time

from zero.config import load_config
from zero.conversation import Conversation
from zero.factory import build_endpointer, build_llm, build_stt, build_voice, build_wake
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
            history_turns=self.cfg.get("llm.history_turns", 6),
        )
        self.state = State.IDLE

    # -- state transition helper -------------------------------------------
    def _to(self, dst: State) -> None:
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
                self._to(State.IDLE)
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
                self._to(State.IDLE)
                return

            self.convo.add_user(text)
            self._t_reply_start = time.monotonic()  # end-of-STT marker for timing
            reply = self._speak_streaming(self.llm.stream(self.convo.messages()))
            self.convo.add_assistant(reply)
            log.info("reply: %r", reply)

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
