"""Google Free Speech-to-Text Engine.

Uses Google's free Speech Recognition service (Web Speech API) via speech_recognition.
Ultra-accurate on English (names, proper nouns, accents, commands), low latency (~200ms),
and requires no authentication or API keys.
"""
from __future__ import annotations

import io
import wave
import numpy as np
import speech_recognition as sr

from zero.stt.base import STT
from zero.utils.logging import get_logger

log = get_logger("stt.google")


class GoogleSTT(STT):
    def __init__(self, language: str = "en-US", timeout: float = 10.0, fallback: STT | None = None):
        self.language = language
        self.timeout = timeout
        self._recognizer = sr.Recognizer()
        self._fallback = fallback
        log.info("Google Speech Recognition (free STT) initialized [lang=%s]", language)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        # Convert float32 or int16 numpy audio into 16-bit PCM WAV bytes
        if audio.dtype == np.int16:
            pcm = audio
        else:
            pcm = np.clip(audio * 32768.0, -32768.0, 32767.0).astype(np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        wav_buf.seek(0)

        try:
            with sr.AudioFile(wav_buf) as source:
                audio_data = self._recognizer.record(source)
            text = self._recognizer.recognize_google(audio_data, language=self.language)
            text = str(text or "").strip()
            log.info("Google STT heard: %r", text)
            return text
        except sr.UnknownValueError:
            log.debug("Google STT: audio was unintelligible")
            return ""
        except sr.RequestError as e:
            log.warning("Google STT request failed (%s)", e)
            if self._fallback is not None:
                log.info("falling back to secondary STT engine...")
                return self._fallback.transcribe(audio, sample_rate)
            raise RuntimeError(f"Google STT service unreachable: {e}") from e
        except Exception as e:
            log.warning("Google STT exception: %s", e)
            if self._fallback is not None:
                return self._fallback.transcribe(audio, sample_rate)
            return ""
