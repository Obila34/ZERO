#!/usr/bin/env bash
# Serve the Cosmos 3 Nano REASONER tower (understanding only — no Generator)
# as an OpenAI-compatible endpoint for ZERO's Tier 2 narrator.
#
# STATUS: VRAM-BLOCKED on the current single RTX 5060 Ti 16 GB — the card
# already runs Gemma + Whisper + Orpheus + vision. This script is the drop-in
# for when either (a) a maintenance window frees the card for a trial, or
# (b) a second GPU / edge box arrives. ZERO side, it is ONLY a config change:
#   world.narrator.backend: openai
#   world.narrator.url:     http://<host>:9200
#   world.narrator.model:   nvidia/Cosmos3-Nano
#
# VRAM budget (why the flags below):
#   8B Reasoner FP8 ≈ 9-10 GB weights+activations; AWQ/4-bit ≈ 5-6 GB.
#   --gpu-memory-utilization caps vLLM's total share of the card.
#   Verify actual numbers with nvidia-smi during a maintenance window —
#   do NOT trust these estimates unmeasured.
set -euo pipefail

PORT="${PORT:-9200}"
MODEL="${MODEL:-nvidia/Cosmos3-Nano}"
GPU_FRACTION="${GPU_FRACTION:-0.45}"     # its slice of the 16 GB card
QUANT="${QUANT:-fp8}"                    # fp8 | awq (pre-quantized checkpoint)

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --quantization "$QUANT" \
  --gpu-memory-utilization "$GPU_FRACTION" \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}'
