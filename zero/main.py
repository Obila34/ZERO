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
        self.mic.start()
        log.info("ZERO ready. Say the wake word to talk. (Ctrl-C to quit)")
        try:
            while True:
                self._wait_for_wake()        # IDLE
                utterance = self._listen()   # LISTENING
                self._respond(utterance)     # THINKING + SPEAKING
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
                log.info("wake word!")
                return

    def _listen(self) -> "None | object":
        self._to(State.LISTENING)
        self.mic.drain()  # drop any audio buffered during wake detection
        return self.endpointer.capture(self.mic.frames())

    def _respond(self, utterance) -> None:
        self._to(State.THINKING)
        if utterance is None or utterance.size == 0:
            log.info("nothing heard")
            self._to(State.IDLE)
            return

        sr = self.cfg.get("audio.sample_rate", 16000)
        text = self.stt.transcribe(utterance, sr).strip()
        if not text:
            self._to(State.IDLE)
            return

        self.convo.add_user(text)
        reply = self._speak_streaming(self.llm.stream(self.convo.messages()))
        self.convo.add_assistant(reply)
        self._to(State.IDLE)

    def _speak_streaming(self, chunks) -> str:
        """Speak sentences as they complete; return the full reply text."""
        full: list[str] = []
        buffer = ""
        spoke_any = False

        for chunk in itertools.chain(chunks, ["\n"]):  # sentinel flush
            buffer += chunk
            sentences = split_sentences(buffer)
            # Keep the last (possibly incomplete) fragment buffered.
            complete, buffer = sentences[:-1], (sentences[-1] if sentences else "")
            for sentence in complete:
                if not spoke_any:
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
