"""Remote TTS — send text to an Orpheus TTS server (on a GPU node) and get audio
back. Orpheus is far more natural and expressive than Piper, but it's a 3B model,
so it runs on the GPU like Gemma and Whisper do. The Pi just sends text and plays
the returned audio.

Orpheus performs emotion cues natively, so we translate ZERO's shared cue tags
(e.g. [laughs]) into Orpheus's syntax (<laugh>) and strip the rest. Output is
float32 mono at the server's rate (Orpheus/SNAC = 24 kHz).
"""
from __future__ import annotations

import io
import re
import wave
from typing import Iterator

import numpy as np
import requests

from zero.tts.base import TTS
from zero.utils.logging import get_logger

log = get_logger("tts.remote")

# ZERO's shared cue tags -> Orpheus native tags. Unmapped cues are stripped.
_CUE_MAP = {
    "[laughs]": " <laugh> ",
    "[chuckles]": " <chuckle> ",
    "[sighs]": " <sigh> ",
    "[hmm]": " ",
    "[pause]": " ",
}
_LEFTOVER_CUE = re.compile(r"\[[a-z]+\]")
_EMPHASIS = re.compile(r"\*(\w+)\*")


def _to_orpheus(text: str) -> str:
    for tag, repl in _CUE_MAP.items():
        text = text.replace(tag, repl)
    text = _LEFTOVER_CUE.sub("", text)     # drop any cue we don't map
    text = _EMPHASIS.sub(r"\1", text)       # drop *emphasis* asterisks
    return " ".join(text.split())           # tidy whitespace


class RemoteTTS(TTS):
    def __init__(self, url: str, voice: str = "tara", sample_rate: int = 24000,
                 timeout: float = 30.0):
        self.url = url
        self.stream_url = url.replace("/tts", "/tts_stream")
        self.voice = voice
        self.sample_rate = sample_rate
        self.timeout = timeout
        log.info("remote TTS (orpheus) -> %s (voice=%s)", url, voice)

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 audio chunks as the server generates them — first audio
        comes back in ~200ms instead of after the whole sentence."""
        text = _to_orpheus(text)
        if not text:
            return
        try:
            resp = requests.post(
                self.stream_url, json={"text": text, "voice": self.voice},
                stream=True, timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error("remote TTS stream failed (server/tunnel up?): %s", e)
            return
        leftover = b""
        for raw in resp.iter_content(chunk_size=8192):
            if not raw:
                continue
            buf = leftover + raw
            n = len(buf) - (len(buf) % 2)  # whole int16 samples only
            leftover = buf[n:]
            if n:
                yield np.frombuffer(buf[:n], dtype=np.int16).astype(np.float32) / 32768.0

    def synthesize(self, text: str) -> np.ndarray:
        text = _to_orpheus(text)
        if not text:
            return np.zeros(0, dtype=np.float32)
        try:
            resp = requests.post(
                self.url, json={"text": text, "voice": self.voice},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error("remote TTS failed (server/tunnel up?): %s", e)
            return np.zeros(0, dtype=np.float32)
        try:
            with wave.open(io.BytesIO(resp.content), "rb") as w:
                self.sample_rate = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        except (wave.Error, EOFError) as e:
            log.error("remote TTS returned bad audio: %s", e)
            return np.zeros(0, dtype=np.float32)
        return pcm.astype(np.float32) / 32768.0
