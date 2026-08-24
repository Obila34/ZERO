# ZERO (AF-1) Conversational Humanoid Brain

ZERO is the embodied AI conversational brain for the **AF-1 Humanoid Robot**, developed by **Zerobionic Africa** in Nairobi, Kenya. It integrates continuous voice perception, Google Speech Recognition (free STT), Gemma 4 (12B) conversational intelligence via vLLM, Kyutai streaming TTS, and real-time bilateral Kenyan Sign Language (KSL) fingerspelling and 10-finger hand dexterity.

---

## 🚀 How to Fire Up the LLM & Services

### 1. Launch the GPU vLLM Model Server
On your GPU server (e.g. `100.95.210.94:8001`):
```bash
vllm serve google/gemma-2-12b-it \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --tensor-parallel-size 1
```

### 2. Start the ZERO Brain Service on Head Pi
```bash
# Start the background conversational service
systemctl --user start zero

# Check real-time service status
systemctl --user status zero

# Follow live logs
tail -f /home/head/zero.service.log

# Restart service after config changes
systemctl --user restart zero

# Stop service
systemctl --user stop zero
```

### 3. Direct HTTP API Text / Voice Turns
```bash
# Fingerspell a word in Sign Language
curl -X POST http://127.0.0.1:8090/zero/turn_text \
  -H "Content-Type: application/json" \
  -d '{"text": "spell PETER in sign language", "speak": true}'

# Execute a hand gesture
curl -X POST http://127.0.0.1:8090/zero/turn_text \
  -H "Content-Type: application/json" \
  -d '{"text": "sign I love you in sign language", "speak": true}'
```

---

## 🧠 Architecture & How the LLM Works

```
IDLE ──(wake word: "hey jarvis")──> LISTENING ──(VAD / silence 750ms)──> THINKING / TOOLS ──> SPEAKING / SIGNING ──> IDLE
```

| Pipeline Stage | Active Engine | Fallback / Alternative | Module |
| :--- | :--- | :--- | :--- |
| **Wake Word** | `openWakeWord` (`hey_jarvis`, thr=0.15) | Alexa, Mycroft | `zero/wake/openwakeword_engine.py` |
| **VAD Endpointer** | `TEN VAD` (wasm, thr=0.35) + `Smart Turn v3` | Silero ONNX | `zero/vad/endpointer.py` |
| **Speech-to-Text (STT)** | **Google Free Speech Recognition** | GPU Whisper (`large-v3-turbo`) | `zero/stt/google_engine.py` |
| **LLM Intelligence** | **Gemma 4 (12B)** via vLLM OpenAI API | Ollama / Llama 3.2 | `zero/llm/openai_engine.py` |
| **TTS Voice** | **Kyutai Streaming TTS** | Piper / Fish Audio | `zero/tts/kyutai_engine.py` |
| **Bilateral KSL & Fingers** | **ArmTool (12 Calibrated Servos)** | Fast Direct Yield | `zero/tools/arms.py` |
| **Web Search** | **SearXNG + LLM Direct Synthesizer** | — | `zero/tools/websearch.py` |

---

## ✨ Features & Capabilities

### 1. Bilateral Kenyan Sign Language (KSL) Fingerspelling
* **Synchronized Dual-Hand Signing**: Automatically mirrors all 26 manual alphabet letters ($A-Z$) across both left and right hands simultaneously.
* **Calibrated Timing**: Exact speed of **1 letter / second** (`1.0s` hold per letter).
* **Spoken Letter Readout**: Synthesizes clear spoken letters (*"Spelling COW: C - O - W."*) while physical hands actuate.
* **Arm Stance Retention**: Steppers remain locked (`allow_steppers: false`) so only fingers and wrists move without drooping the arm.

### 2. Full 10-Finger Hand Dexterity & Gestures
Independent multi-joint control across Thumb, Index, Middle, Ring, and Pinky on both hands:
* ✌️ **Peace / Victory Sign**: Index & Middle extended ($90^\circ$), others curled ($0^\circ$).
* 🤟 **I Love You Sign (ILY)**: Thumb ($5^\circ$), Index ($90^\circ$), and Pinky ($99^\circ$) extended; Middle & Ring curled ($0^\circ$).
* 👉 **Point**: Index extended forward, others curled.
* 👍 **Thumbs Up**: Thumb extended upward ($5^\circ$), fingers in a fist.
* ✊ **Fist / 🖐️ Open Hands**: Full closure or extension.
* 〰️ **Wiggle Fingers**: Dynamic sequential wave ripple across all fingers.
* 🎯 **Individual Finger Control**: *"move your thumb"*, *"bend right index finger"*, *"open left pinky"*.

### 3. Voice & Conversation Intelligence
* **Zerobionic Africa Identity**: Grounded in its assistive STEM & deaf education mission.
* **English-Only Fluent Mode**: Crisp, concise 1-2 sentence replies without fluff.
* **Multi-Layer Thought Guard**: Automatically strips reasoning markers (`<thought>...</thought>`) to avoid speech loops.
* **Vision On-Demand**: Ambient vision remarks are suppressed unless you explicitly ask visual questions (*"What do you see?"*).
* **SearXNG Live Web Search**: Synthesizes live factual information cleanly without reading URLs or metadata headers.
