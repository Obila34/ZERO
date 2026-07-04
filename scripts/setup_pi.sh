#!/usr/bin/env bash
# ZERO — one-pass setup for a fresh Raspberry Pi OS (64-bit), Pi 4 or 5.
#
# Installs the LOCAL voice stack: audio libs, wake word, VAD, Piper TTS, and an
# optional local Whisper fallback. The LLM (Gemma) and STT (Whisper turbo) run on
# the GPU over SSH tunnels — see docs/GPU_OFFLOAD_PLAN.md. Those are not installed
# here; you open the tunnels separately.
#
# Usage:
#   git clone -b offline_v5 --single-branch https://github.com/Obila34/ZERO.git
#   cd ZERO && bash scripts/setup_pi.sh
set -e
cd "$(dirname "$0")/.."   # repo root

echo "== system audio libraries =="
sudo apt update
sudo apt install -y libportaudio2 libsndfile1 espeak-ng python3-venv pipewire-alsa

echo "== python venv =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "== python dependencies =="
pip install requests PyYAML numpy sounddevice soundfile cffi onnxruntime \
            webrtcvad scipy scikit-learn tqdm
pip install --no-deps openwakeword       # tflite has no wheel; we run on ONNX
pip install "setuptools<81"              # webrtcvad needs pkg_resources
python -c "import openwakeword.utils; openwakeword.utils.download_models()"

echo "== Piper TTS (voice + arm64 binary) =="
mkdir -p models/piper
[ -f models/piper/en_US-amy-medium.onnx ] || wget -O models/piper/en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
[ -f models/piper/en_US-amy-medium.onnx.json ] || wget -O models/piper/en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
if [ ! -x piper/piper ]; then
  wget -O piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
  tar -xzf piper.tar.gz
fi

echo "== identity models (voice + face embedders) =="
mkdir -p models/voiceid models/identity
[ -f models/voiceid/voxceleb_ECAPA512_LM.onnx ] || wget -O models/voiceid/voxceleb_ECAPA512_LM.onnx \
  https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/main/voxceleb_ECAPA512_LM.onnx || true
# ArcFace face embedder (InsightFace buffalo_l recognition model, 512-d).
[ -f models/identity/arcface.onnx ] || wget -O models/identity/arcface.onnx \
  https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx || true
pip install opencv-python-headless || true   # Haar face detection (identity)

echo "== Silero VAD (ONNX, torch-free) =="
mkdir -p models/vad
[ -f models/vad/silero_vad.onnx ] || wget -O models/vad/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx || true

echo "== vision extras (camera + detection: opencv, pydantic) =="
# The eyes need these; without pydantic the whole vision stack fails to import
# and ZERO silently runs voice-only. Kept light (no torch — YOLO is ONNX/GPU).
pip install -r requirements-vision.txt || true

echo "== optional local STT fallback (whisper base.en) =="
mkdir -p models/whisper
[ -f models/whisper/ggml-base.en.bin ] || wget -O models/whisper/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin || true
pip install pywhispercpp || true

echo
echo "Setup done. Next steps:"
echo "  1) Open the GPU tunnels (LLM + STT):"
echo "       ssh -fN -L 11435:localhost:11434 -L 9000:localhost:9000 <gpu-alias>"
echo "  2) Set the mic gain as a PERCENT (ranges vary by card):  amixer -c 3 set Mic 60%"
echo "  3) Run it:  source .venv/bin/activate && python -m zero.main"
echo "  (Brain-only test, no mic/audio needed:  python -m zero.main --text)"
