#!/usr/bin/env bash
# Fetch the TEN VAD + Smart Turn v3 models (gitignored, like all of models/).
# Already present on the Pi via mutagen sync; run this on any fresh checkout.
#
#   bash scripts/fetch_turn_vad_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models/vad models/turn

# TEN VAD — official WebAssembly build (no ARM64/Linux native build exists;
# see zero/vad/ten_wasm.py). ~283 KB.
if [ ! -s models/vad/ten_vad.wasm ]; then
  echo "fetching ten_vad.wasm ..."
  curl -fSL "https://raw.githubusercontent.com/TEN-framework/ten-vad/main/lib/Web/ten_vad.wasm" \
    -o models/vad/ten_vad.wasm
fi

# Pipecat Smart Turn v3 — CPU ONNX end-of-turn model. ~8.7 MB.
if [ ! -s models/turn/smart-turn-v3.2-cpu.onnx ]; then
  echo "fetching smart-turn-v3.2-cpu.onnx ..."
  curl -fSL "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx" \
    -o models/turn/smart-turn-v3.2-cpu.onnx
fi

echo "done:"
ls -la models/vad/ten_vad.wasm models/turn/smart-turn-v3.2-cpu.onnx
