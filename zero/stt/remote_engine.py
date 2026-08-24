"""Remote STT — send the captured utterance to a Whisper server (on the GPU) and
get text back. This is how we run large-v3-turbo: too slow on the Pi CPU (29s),
fast on the GPU. The audio goes up the SSH tunnel as a small WAV, text comes back.

Only needs requests + numpy on the Pi (no pywhispercpp, no local model).
"""
from __future__ import annotations

import io
import wave

import numpy as np
import requests

from zero.stt.base import STT
from zero.utils.logging import get_logger

log = get_logger("stt.remote")


class RemoteSTT(STT):
    def __init__(self, url: str, timeout: float = 30.0,
                 language: str | None = None):
        # Optional language pass-through: 'sw', 'auto' (server lets Whisper
        # detect — best for code-switched speech), or None = server default.
        if language:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}language={language}"
        self.url = url
        self.timeout = timeout
        log.info("remote STT -> %s", url)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # float32 [-1,1] (or int16) -> 16-bit PCM WAV bytes (self-describing rate).
        if audio.dtype == np.int16:
            pcm = audio
        else:
            pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())

        try:
            resp = requests.post(
                self.url,
                data=buf.getvalue(),
                headers={"Content-Type": "audio/wav"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
            # Suppress common Whisper silence / low-noise hallucinations
            low = text.lower().strip(" .!?,")
            if low in ("thank you", "thanks", "thank you very much", "subtitles by", "you", "thanks for watching", "bye"):
                log.info("suppressed Whisper silence hallucination: %r", text)
                text = ""
        except (requests.RequestException, ValueError) as e:
            # Raise instead of returning "" — the caller must be able to tell a
            # dead tunnel from silence, so it can fail over to a local engine.
            raise RuntimeError(
                f"remote STT failed (is the tunnel/server up?): {e}"
            ) from e
        log.info("heard: %r", text)
        return text
