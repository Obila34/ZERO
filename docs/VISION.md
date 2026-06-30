# ZERO Vision — eyes on the voice robot

This folds the old standalone *Zero Vision* project into ZERO as one more sense.
There is no separate vision app, no second conversation loop, and no second
brain: the camera feeds the **same** Gemma prompt, the **same** SQLite memory,
and the **same** Orpheus voice the conversation already uses.

## The one idea: the eyes are always open

A person doesn't boot up their eyes when you say their name. So the camera →
YOLO11n → color loop (`zero/vision/eyes.py`) runs continuously on a background
thread from the moment ZERO starts, keeping a rolling "current scene" (latest
keyframe + named detections). The wake word doesn't *start* vision — it makes
ZERO *attend* to what it has already been seeing. That's why a visual question
comes back fast: the scene is pre-computed, never on the critical path.

## What runs where

```
+----------------------- Raspberry Pi 5 (edge, always on) ----------------------+
|  mic -> wake word -> VAD -> record clip                                        |
|  camera -> YOLO11n -> HSV color  ==> rolling SceneState (keyframe + detections)|
|  orchestrator (zero/main.py): on each turn, fold the scene into the LLM prompt |
|  memory (SQLite)   Bluetooth playback   barge-in                              |
+----------------------------------------|--------------------------------------+
                       one SSH tunnel (LLM / STT / TTS / Vision)
                                         v
+------------------------------- GPU node --------------------------------------+
|  Whisper large-v3-turbo  (STT)  :9000                                         |
|  Gemma (8B, multimodal-capable) via Ollama  :11434  <- the single brain       |
|  Orpheus (quantized)     (TTS)  :9100                                         |
|  Vision server: Depth Anything V2 -> distance + bearing  :8000  (/facts)      |
+-------------------------------------------------------------------------------+
```

The Pi does the cheap, always-on, latency-critical work; the GPU does the heavy,
per-turn work. If the GPU/tunnel is down, the eyes still work locally (objects +
colors, just no distances) and the voice pipeline falls back as before.

## How a turn uses vision

1. Every turn attaches a quick, GPU-free ambient line to the user message:
   `(You can currently see: a person, a red cup, a gray laptop.)`
2. If the utterance looks **visual** (`see`, `this`, `color`, `where`, `who`,
   `holding`, …), ZERO also:
   - POSTs the current keyframe + local detections to the GPU `/facts` endpoint
     and gets grounded `distance_m` + `bearing` per object, and
   - if `vision.multimodal: true`, attaches the keyframe JPEG to the LLM turn so
     a vision-capable model can actually look at it.
3. The augmented turn goes through the **existing** streaming LLM → Orpheus path.
   The scene text lives only on the user turn (never the cached system prefix or
   stored history), so the prompt cache stays warm and pure-chat turns are as
   fast as before.

### One brain (Gemma) or a separate VLM?

The spatial math (distance, bearing) is computed **deterministically** on the GPU
(YOLO boxes + Depth Anything V2 + intrinsics), so the LLM never invents numbers —
it just narrates grounded facts. That means you do **not** need a dedicated VLM:
a vision-capable Gemma is plenty, and using it as the single brain saves VRAM.

- Check whether your model can see: `ollama show <your-model>` (look for a vision
  capability). If yes → set `vision.multimodal: true`.
- If your text model can't see, leave `multimodal: false`: ZERO still "sees" via
  detections + GPU depth facts as **text**. (A standalone Qwen2-VL `/analyze`
  path is still present in `server/vision/` if you ever want it.)

## Turn it on

### Pi
```bash
pip install -r requirements-vision.txt      # opencv + ultralytics + pydantic
# edit config.yaml:  vision.enabled: true   (and multimodal: true if the LLM sees)
```
`yolo11n.pt` is bundled at the repo root, so detection works fully offline.

### GPU
```bash
cd server/vision
pip install -r requirements.txt             # torch + transformers (depth)
uvicorn app:app --host 0.0.0.0 --port 8000  # or use scripts/run_gpu_servers.sh
curl localhost:8000/health                  # {"status":"ok","gpu":true,...}
```

### Tunnel (Pi → GPU), now carrying four services
```bash
GPU_HOST=user@gpu-box bash scripts/pi_tunnel.sh
# local 11435->11434 (LLM), 9000 (STT), 9100 (TTS), 8000 (Vision)
```

Then `python -m zero.main` and say "Hey Jarvis". For always-on across reboots,
use the units in [`scripts/systemd/`](../scripts/systemd/README.md).

## Config

Everything is under `vision:` in [`config.yaml`](../config.yaml): enable flag,
`multimodal`, camera geometry, the YOLO class list, color thresholds, and the GPU
endpoint. Accuracy of distances depends on camera intrinsics — the server uses a
FOV fallback until you copy a real `intrinsics.json` next to `server/vision/`
(see the calibration script in the original Zero_Vision project).

## Tuning latency

- Keep `vision.multimodal` off unless you need raw-image understanding — sending
  a JPEG up the tunnel + image prefill is the most expensive part.
- `vision.detect_interval_s` caps how often YOLO runs (raise it to lower Pi CPU
  load/heat; 0 = as fast as the CPU allows).
- A Hailo-8L AI HAT can offload detection from the Pi CPU (`detect.device`).
