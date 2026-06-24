# ZERO — Progress Report
**Date:** 24 June 2026  ·  **Platform:** Raspberry Pi 5 (8 GB), fully offline  ·  **Author:** Gregory Kimani

## 1. Objective
Build the first step toward a conversational humanoid: a device you talk to in English that
replies in a natural human voice — running **entirely on-device, no internet**.

## 2. What we built today
A complete, working offline voice assistant with an interface-first, config-driven architecture
(every stage swappable via `config.yaml`, no code changes):

| Stage | Engine | Status |
|-------|--------|--------|
| Wake word | openWakeWord ("Hey Jarvis", ONNX runtime) | ✅ working |
| Voice activity / endpointing | silero-VAD | ✅ working |
| Speech-to-text | whisper.cpp (`tiny.en`) | ✅ ~2 s/utterance |
| Conversation brain | Ollama + Llama 3.2 3B | ✅ working |
| Text-to-speech | Piper (Fish S1-mini planned for emotion) | ✅ speaking |
| Output | Bluetooth speaker (A2DP) · Input: Logitech BRIO USB mic | ✅ working |

**Key features delivered**
- **Multi-turn conversation mode:** say the wake word once, then converse freely; after 30 s of
  silence it returns to sleep and waits for the wake word again.
- **Streaming replies:** TTS speaks each sentence while the LLM is still generating the rest.
- **Diagnostic tooling:** scripts to test audio devices, microphone level, wake-word score, and TTS.

**Problems solved during the day**
- openWakeWord install failure (no `tflite-runtime` for Python 3.12/ARM) → switched to ONNX runtime.
- whisper model path not loading (pywhispercpp expects name + dir) → robust loader.
- silero VAD crash ("chunk too short") → buffer audio into exact 512-sample windows.
- Audio queue overflow during processing → ring-buffer + drain-before-listen.
- **Whisper hallucinations** ("Thank you", "I hope you enjoyed this video" on silence) → VAD
  threshold + minimum-speech-duration gate, so noise blips are rejected before transcription.

## 3. Result
A genuinely usable offline assistant: **wake → natural back-and-forth → sleep**, ~2 s transcription,
spoken replies through the Bluetooth speaker. Proves the full pipeline end-to-end on the Pi 5.

## 4. Why a GPU is the better long-term host (vs. Pi 5)
The Pi 5 has **no ML-usable GPU** — every model runs on its 4 CPU cores. This is the root cause of
our main limitation: **latency**.

| Factor | Pi 5 (CPU only) | GPU host |
|--------|-----------------|----------|
| LLM reply (3B model) | ~9–12 s/turn (a few tokens/sec) | < 1–2 s (50–100+ tokens/sec) |
| Speech-to-text | ~2–6 s | ~0.2–0.5 s (5–20× faster) |
| Expressive TTS (Fish S1-mini) | impractical on CPU | real-time — unlocks laughs/sighs/emotion |
| Memory | 8 GB shared; LLM + STT + Fish can't all fit | VRAM + system RAM hold all models at once |
| Sustained load | thermal throttling | designed for it |
| Model headroom | small models only | 7B+ LLMs, vision, future humanoid features |

**Recommendation.** The Pi 5 is excellent for *proving the system* and for low-power, portable
deployment, and it has succeeded at that. But the project's defining goals — **near-instant,
fluid conversation** and a **genuinely human, expressive voice (the Fish/emotion engine)** — are
gated by CPU compute the Pi cannot provide. A GPU host (or a hybrid: Pi as the edge device talking
to a GPU server) removes the latency wall, makes expressive TTS viable, and leaves room for the
vision and larger-model work the humanoid roadmap will need.

## 5. Next steps
- Switch the expressive voice (Fish S1-mini) on once a GPU is available; benchmark real-time factor.
- Train a custom **"Hey Zero"** wake word to replace the placeholder "Hey Jarvis".
- Optional: add a short mic-mute window after speaking to prevent any speaker-into-mic echo.
