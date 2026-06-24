#!/usr/bin/env bash
# One-shot setup for ZERO on a fresh Raspberry Pi 5 (64-bit Raspberry Pi OS).
# Installs system audio deps, Python deps, Ollama + a chat model, whisper.cpp
# model, and the Piper binary + voice. Fish S1-mini is fetched but its inference
# command is finalized in the Phase-6 benchmark.
#
# Usage:  bash scripts/setup_pi.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"/{whisper,piper,fish}

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip portaudio19-dev libsndfile1 \
                        git build-essential cmake wget

echo "==> Python venv + deps"
python3 -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"
# openWakeWord pins tflite-runtime (no wheel for Python 3.12 on ARM) — install it
# without deps and run it on ONNX (see config: wake.inference_framework: onnx).
pip install "openwakeword>=0.6.0" --no-deps
python -c "import openwakeword.utils as u; u.download_models()"

echo "==> Ollama + chat model"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
ollama pull llama3.2:3b   # use llama3.2:1b on a 4 GB Pi or for lower latency

echo "==> whisper.cpp model (base.en)"
WHISPER_BIN="$MODELS/whisper/ggml-base.en.bin"
if [ ! -f "$WHISPER_BIN" ]; then
  wget -O "$WHISPER_BIN" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
fi

echo "==> Piper binary + voice"
# Adjust the release URL to the latest piper for linux_aarch64 if needed.
if ! command -v piper >/dev/null 2>&1; then
  echo "    Install the 'piper' binary on PATH (github.com/rhasspy/piper releases,"
  echo "    linux_aarch64). Then re-run, or set tts.piper.binary to its full path."
fi
PIPER_VOICE="$MODELS/piper/en_US-amy-medium.onnx"
if [ ! -f "$PIPER_VOICE" ]; then
  BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
  wget -O "$PIPER_VOICE" "$BASE/en_US-amy-medium.onnx"
  wget -O "$PIPER_VOICE.json" "$BASE/en_US-amy-medium.onnx.json"
fi

echo "==> Fish OpenAudio S1-mini (expressive mode — PyTorch/CPU build, NOT MLX)"
# Gated repo: run `huggingface-cli login` first if prompted.
if [ ! -f "$MODELS/fish/openaudio-s1-mini/model.pth" ]; then
  pip install -U "huggingface_hub[cli]"
  huggingface-cli download fishaudio/openaudio-s1-mini \
    --local-dir "$MODELS/fish/openaudio-s1-mini" || \
    echo "    (skip/accept license at hf.co/fishaudio/openaudio-s1-mini, then retry)"
fi
cat <<'NOTE'

    Phase-6 spike TODO for Fish:
      1. Install fish-speech and confirm its inference CLI for your version.
      2. Quantize model.pth to int8/int4 (keep codec.pth fp16).
      3. Benchmark real-time factor + RAM on the Pi 5.
      4. Set tts.fish.infer_cmd in config.yaml (placeholders {model_dir} {text} {out}).
NOTE

echo
echo "==> Done. Verify audio, then run:"
echo "    source .venv/bin/activate"
echo "    python scripts/list_audio_devices.py --loopback"
echo "    python -m zero.main"
