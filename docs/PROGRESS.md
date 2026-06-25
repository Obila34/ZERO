# ZERO — Offline Voice Assistant on Raspberry Pi 5

**Status:** Working end-to-end, conversational. Tuning latency.
**Last updated:** 2026-06-25
**Branch:** `offline_v5`

---

## 1. What it is

A **fully offline** voice assistant running entirely on a Raspberry Pi 5 (16 GB) —
no cloud, no internet at runtime. You say a wake word once, then hold a natural
back-and-forth conversation. Everything (speech-to-text, the language model, and
text-to-speech) runs on the Pi's CPU.

---

## 2. The pipeline

```
You speak ─► Mic (USB) ─► Wake word ─► Voice activity detection ─► Speech-to-text
                                                                        │
                                                                        ▼
            BT speaker ◄─ Text-to-speech ◄─ Language model (streamed) ◄─┘
```

| Stage | Engine | Model | Why |
|-------|--------|-------|-----|
| Wake word | openWakeWord (ONNX) | `hey_jarvis` | Light, always-on; ONNX (no tflite needed on Py 3.13) |
| Voice activity | webrtcvad | — | Ultra-light, fully offline, no PyTorch |
| Speech-to-text | whisper.cpp | `base.en` (Q5) | Best speed/accuracy balance on Pi CPU |
| Language model | Ollama | `qwen2.5:3b-instruct` (Q4) | Smartest 3B that stays low-latency on Pi |
| Text-to-speech | Piper | `en_US-amy-medium` | Fast, natural, runs as a native binary |

---

## 3. Key decisions (and why)

- **No giant quantized models.** The Pi is limited by CPU speed, not storage. A
  quantized 8B model *fits* but runs at ~2 tok/s — unusable. The 3B class is the
  sweet spot. Bigger ≠ better here.
- **Qwen2.5-3B over Llama-3.2-3B / TinyLlama.** TinyLlama hallucinated badly; the
  3B Qwen is concise and reliable with a tight system prompt.
- **Ollama instead of loading the model directly.** Ollama keeps the model warm in
  RAM between turns, which is a real latency win.
- **Linear pipeline, not "streaming/speculative."** On a CPU-bound Pi, overlapping
  speech-to-text + LLM + TTS concurrently starves the CPU and glitches audio. We
  overlap only the safe, high-impact part (LLM → TTS, sentence by sentence).

---

## 4. The engineering journey (what we hit, what we did)

| # | Problem | Fix | Result |
|---|---------|-----|--------|
| 1 | Py 3.13 has no `tflite-runtime` wheel; openWakeWord wouldn't install | Force the **ONNX backend** + install with `--no-deps` | Wake word loads |
| 2 | `webrtcvad` needs `pkg_resources` | Pin `setuptools<81` | VAD loads |
| 3 | Default ALSA device timed out (`paTimedOut`) | Capture the **USB mic directly**; retry + high latency on open | Stable mic |
| 4 | Mic captured silence → Whisper hallucinated | Pin PipeWire default source to the USB mic | Real transcription |
| 5 | **Speech-to-text took 14.7s per phrase** | Whisper always encodes a 30s window — capped it with **`audio_ctx=768`** + 4 threads | **14.7s → ~1.5s** |
| 6 | TTS silent | Installed the **Piper binary** (arm64) | It speaks |
| 7 | **First reply froze for ~28s** (cold model load) | **Warm up** the LLM at boot + `keep_alive=-1` (pin in RAM) | First reply fast |
| 8 | It only answered ONE question per wake word | Rewrote the loop into a **continuous conversation** (no wake word between turns; ends on "goodbye" or 30s silence) | Real multi-turn |
| 9 | **~40s freezes mid-conversation** | The rolling history was forcing the model to re-read the whole conversation each turn. Shrank the system prompt + history (6→3 turns) + shorter replies | **40s → ~10s** on bad turns |
| 10 | It transcribed background people & its own voice | **Mute mic while thinking/speaking**, drop non-speech like `(applause)`, VAD aggressiveness 3 + **loudness gate** | Stops eating the room |
| 11 | Long silences while it "thought" | **Spoken fillers** ("Let me think.") play *while* the model generates in the background | Masks the wait |

---

## 5. A few key code bits

**The 30-second-window fix (the biggest latency win):**
```python
# zero/stt/whispercpp_engine.py — cap the encoder window for short turns
kwargs = {"language": self.language}
if self.audio_ctx:                 # 768 in config
    kwargs["audio_ctx"] = self.audio_ctx
segments = self._model.transcribe(audio, **kwargs)
```

**Pin the model in RAM + warm it at boot:**
```python
# zero/llm/ollama_engine.py
self.keep_alive = -1               # never unload from RAM
def warmup(self):                  # called once at startup
    requests.post(f"{self.host}/api/chat", json={
        "model": self.model, "messages": [{"role": "user", "content": "hi"}],
        "keep_alive": self.keep_alive, "options": {"num_predict": 1}})
```

**Drop Whisper's non-speech hallucinations (background noise):**
```python
# "(applause)", "[BLANK_AUDIO]" -> "" so they never reach the LLM
_BRACKETS = re.compile(r"[\(\[][^\)\]]*[\)\]]")
def _strip_hallucinations(text):
    cleaned = _BRACKETS.sub("", text).strip(" .,!?-…")
    return cleaned if re.search(r"[A-Za-z]{2,}", cleaned) else ""
```

**Overlapped fillers — generate while a filler plays (no added delay):**
```python
# zero/main.py
chunks = self._stream_in_background(self.convo.messages())  # LLM starts NOW
self._play_filler()              # "Let me think." plays over the model's prefill
reply = self._speak_streaming(chunks)   # real answer flows straight in
```

---

## 6. Current performance (on the Pi 5 CPU)

| Metric | Number |
|--------|--------|
| Speech-to-text | **~1.5–2s** per turn (was 14.7s) |
| LLM first token (warm, short context) | **~2s** |
| LLM first token (when history is full) | **~10–17s** ← the remaining issue |
| End-to-end feel | ~3–5s on good turns; filler masks the slow ones |
| RAM in use | ~4.5 GB of 16 GB (lots of headroom) |

---

## 7. Tunable knobs (all in `config.yaml`)

```yaml
stt:   { model: base.en, threads: 4, audio_ctx: 768 }
vad:   { silence_ms: 500, aggressiveness: 3, energy_threshold: 350 }
llm:   { model: qwen2.5:3b-instruct, temperature: 0.3, max_tokens: 140, history_turns: 3 }
conversation: { sleep_timeout_ms: 30000, filler_probability: 0.9 }
```
- `energy_threshold` — raise it in noisier rooms so background voices are ignored.
- `audio_ctx` — the speech-to-text speed lever.
- `history_turns` — fewer = faster, less memory of the conversation.

---

## 8. Known limitations & next steps

| Item | Status |
|------|--------|
| **LLM still spikes to ~15s when history fills** | **Next:** test `qwen2.5:1.5b-instruct` — prefills ~2× faster. The real fix for raw speed. |
| Open USB mic still catches loud background talk | Software-mitigated (gate + VAD). A **headset/directional mic** is the proper fix in noisy rooms. |
| Audio defaults reset on reboot | Need to make the PipeWire mic/speaker defaults persistent. |
| Wake word is "Hey Jarvis", not "Hey Zero" | Built-in model for now; a custom "Hey Zero" needs training. |

---

## 9. Run it

```bash
# one-time per boot: connect speaker + set audio defaults
bluetoothctl connect <speaker-mac>
pactl set-default-sink   <bt-speaker>
pactl set-default-source <usb-mic>

# start
cd ~/Mzee/offline_v5 && source .venv/bin/activate
python -m zero.main
# say "Hey Jarvis", then just talk. Say "goodbye" to end.
```
