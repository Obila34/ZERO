# ZERO

An **offline** conversational voice robot for the **Raspberry Pi 5** — talk to it
in English, it replies in a natural human-sounding voice (with chuckles, sighs and
emphasis). First step toward a conversational humanoid.

Everything runs **on the Pi, no internet**: wake word → speech-to-text → a local
LLM → text-to-speech. Every stage sits behind a small interface, so any engine
(and later, a cloud engine) can be swapped in `config.yaml` without code changes.

## Pipeline

```
IDLE ──(wake word)──> LISTENING ──(end of speech)──> THINKING ──> SPEAKING ──> IDLE
```

| Stage      | Engine (default)            | Module                              |
|------------|-----------------------------|-------------------------------------|
| Wake word  | openWakeWord                | `zero/wake/openwakeword_engine.py`  |
| Endpointing| silero-vad (webrtc fallback)| `zero/vad/endpointer.py`            |
| STT        | whisper.cpp (`base.en`)     | `zero/stt/whispercpp_engine.py`     |
| LLM        | Ollama + Llama 3.2 3B       | `zero/llm/ollama_engine.py`         |
| TTS (fast) | Piper + orchestrator        | `zero/tts/piper_engine.py`          |
| TTS (expr.)| Fish OpenAudio S1-mini      | `zero/tts/fish_engine.py`           |
| Vision     | YOLO11n (ONNX) + Depth Anything V2 | `zero/vision/` · `server/vision/` |

### Eyes (vision)

ZERO can also **see**. A camera + YOLO11n + color loop (`zero/vision/eyes.py`)
runs continuously from startup — perception is never on the critical path, so the
moment you ask "what's this?" the scene is already perceived. The wake word that
opens a conversation makes ZERO *attend* to what it's been seeing all along.

- Every turn gets a quick, GPU-free ambient line ("in view: a person, a red cup").
- **Visual** questions also fetch grounded distance + bearing from the GPU vision
  server (Depth Anything V2 over the same SSH tunnel) and, if `vision.multimodal`
  is on, hand the keyframe straight to a vision-capable LLM.
- One brain, one memory, one voice: the scene is folded into the **same** Gemma
  prompt and SQLite memory the conversation already uses.

Off by default. Turn on with `vision.enabled: true` in `config.yaml` after
`pip install -r requirements-vision.txt` (light: OpenCV + onnxruntime, **no
torch** — YOLO11n runs as the bundled `yolo11n.onnx`). Falls back to local
detections (no distances) if the GPU is unreachable. See
[`docs/VISION.md`](docs/VISION.md).

### Two-tier voice

- **Piper** (default): fast and reliable. Expressiveness comes from the
  orchestrator — the LLM emits cue tags (`[laughs]`, `[sighs]`, …) and short
  pre-recorded clips are spliced in (`zero/tts/nonverbals/`).
- **Fish S1-mini** (expressive mode): performs `(laughing)`, `(chuckling)`,
  `(whispering)` etc. natively. Heavier — benchmark on the Pi before making it
  default. Use the **PyTorch/CPU** build, **never** the MLX build (Mac-only).

Switch with `tts.engine: piper | fish` in `config.yaml`. The LLM prompt never
changes — the orchestrator translates the shared cue vocabulary per engine.

## Setup (on the Pi 5)

```bash
bash scripts/setup_pi.sh                       # deps, models, Ollama, Piper
source .venv/bin/activate
python scripts/list_audio_devices.py --loopback # verify mic + speaker
python -m zero.main                             # start talking
```

Set the mic/speaker indices it prints into `audio.input_device` /
`audio.output_device` in `config.yaml` (or a `config.local.yaml` override).

## Configuration

All knobs live in [`config.yaml`](config.yaml): audio devices, wake word +
threshold, VAD timings, STT/LLM models, and TTS engine + voice. A
`config.local.yaml` (gitignored) deep-merges on top for per-device overrides.

## Notes / roadmap

- **Latency** is the cost of fully-local: replies are streamed sentence-by-sentence
  so speech starts before the whole reply is generated. Drop the LLM to
  `llama3.2:1b` and STT to `tiny.en` if it's slow.
- **RAM (8 GB Pi):** the LLM + whisper + Fish don't all fit resident at once —
  quantize Fish's transformer (int8/int4) and/or lazy-load. See the plan.
- **Phase 7 polish:** barge-in (interrupt playback on a new wake word) and a
  spoken "thinking" filler are stubbed (`should_stop` hook in playback).
