# ZERO — Current State & Roadmap

A real-time, (mostly) offline conversational robot. You talk, it sees you, it
talks back in a natural voice with emotion — duplex, interruptible, proactive.

## What ZERO can do today

- **Wake + flowing conversation.** `hey_jarvis` opens a session; after that it's
  free-flowing (no wake word between turns) until a stop phrase or silence.
- **Hears** — remote Whisper `large-v3-turbo` on the GPU (~1s), local
  `whisper.cpp` fallback if the tunnel drops.
- **Thinks** — Gemma (8B) via Ollama on the GPU, streamed sentence-by-sentence.
  Warm-started with the real prompt so first token is ~0.6s.
- **Speaks** — Orpheus TTS (GPU, expressive) streaming; Piper local fallback.
  Emits cue tags `[laughs] [chuckles] [sighs] [gasps] [hmm] [pause]`.
- **Sees** — always-on camera + YOLO11 detection (COCO-80). Perception runs
  continuously off the critical path, so the scene is already known when asked.
  Multimodal turns hand keyframes straight to Gemma.
- **Duplex** — speech barge-in (talk over it and it stops, echo-aware),
  interruption self-awareness, backchannel fillers that race the real reply.
- **Emotion** — reads voice prosody + text + (optional) face into a cross-turn
  mood that shapes both Gemma's tone and the voice's delivery.
- **Remembers** — SQLite long-term memory (facts + episodes), activation-ranked
  recall, sleep-phase consolidation and reflection. Each person also gets a
  durable "last time we spoke" record that never fades, so a returning voice is
  greeted with what you last discussed — even years later. Long sessions are
  compacted on the fly (trimmed turns fold into a rolling summary) so context
  stays small and the shared GPU doesn't fill up.
- **Knows people** — sessions are owned by **voice**: whoever's talking owns the
  turn and its memories (anonymous when the voice is unsure); a different enrolled
  voice taking over hands the session off live. Face recognition still runs — for
  "I can see you" and the log — but never decides whose session it is.
  Conversational enrolment ("I'm David"), diarization, privacy modes.
- **Proactive** — greets people it sees, asks queued curiosity questions, makes
  ambient + scene-change remarks; hard-gated by cooldowns and quiet hours.
- **Learns objects** — "this is a french press" binds a crop to a name.
- **Acts** — tool router: time, timers, reminders, remember/recall.

## Architecture (two machines)

```
        Pi 5 (frontend)                    GPU node (heavy senses)
  mic → VAD → [barge-in]                 Whisper large-v3-turbo  :9000
   │                                     Ollama + Gemma 8B       :11435
   └─ wake → STT ─────── SSH tunnel ───▶ Orpheus TTS            :9100
   camera → YOLO11 ────── tunnel ──────▶ YOLO11x / faces / CLIP  :8000
   speaker ◀── audio                    (all lazy local fallbacks on the Pi)
  SQLite: memory · identity · objects · curiosity
```

Every stage sits behind an interface and is chosen in `config.yaml`; each remote
stage degrades to a smaller local engine if the tunnel drops, then recovers.

- Entry point: `zero/main.py` — the IDLE→LISTENING→THINKING→SPEAKING loop.
- Persona / prompt: `zero/llm/persona.py`.
- Config: `config.yaml` (+ gitignored `config.local.yaml` for per-device knobs).
- Ops / start commands / troubleshooting: `docs/OPERATIONS.md`.

## Roadmap

### 1. "Hey Zero" wake word
Currently `hey_jarvis` (a built-in). Train a custom openWakeWord model:
- Generate samples with Piper TTS (many voices/speeds) + augment with noise/RIR,
  or crowd-record real "Hey Zero" clips.
- Train per openWakeWord's notebook → `hey_zero.onnx`; drop it in and set
  `wake.model: hey_zero`. Tune `wake.threshold` against false-accept rate.

### 2. Better object detection (feeding datasets)
Today: YOLO11 on COCO-80 + few-shot learned objects. To go further:
- **Fine-tune** YOLO11 on your own labelled images (the objects in your space it
  gets wrong), export to ONNX, point `vision.detect.model_path` at it.
- **Open-vocabulary**: export a YOLO-World model with a custom prompt vocabulary
  (`scripts/export_yolo_onnx.py --vocab ...`) to name things beyond COCO with no
  retraining.
- Scale the few-shot learner with a real CLIP image encoder for tighter matches.

## Worth doing beyond that

- **Semantic turn-taking** — replace the fixed silence timer with a
  turn-completion model so it responds the instant you're done, not after a gap.
- **Streaming ASR** — feed Whisper rolling partials so Gemma prefills *while* you
  speak (cuts perceived latency toward the sub-300ms target).
- **Acoustic echo cancellation** — a proper AEC (speex/WebRTC APM) would let the
  mic stay fully open during speech and make barge-in bulletproof on a speaker.
- **Auto-gain (AGC)** — end the manual `input_gain` tuning; normalise level so
  any mic just works.
- **Voice/wake as identity** — only wake for enrolled voices; personalise the
  greeting and persona per person.
- **Embodiment** — drive the LED/servos from the mood + attention already
  computed, so it visibly reacts.
- **On-device fallback brain** — a small quantized LLM on the Pi so a dead
  tunnel degrades gracefully instead of going quiet.
- **Config hygiene** — move all per-device hardware knobs to `config.local.yaml`
  so machines stop colliding on `git pull`.
