#!/usr/bin/env bash
# Open (and auto-heal) the SSH tunnel from the Pi to every GPU server ZERO uses.
# One tunnel carries all four services; nothing is exposed to the internet.
#
#   local :11435 -> GPU :11434   Ollama (LLM)
#   local :9000  -> GPU :9000    Whisper (STT)
#   local :9100  -> GPU :9100    Orpheus (TTS)
#   local :8000  -> GPU :8000    Vision (depth + scene facts)
#
# These local ports match config.yaml (llm.host, stt.remote_url, tts.orpheus.url,
# vision.gpu.url). autossh restarts the tunnel if the link drops, so the Pi keeps
# working across Wi-Fi blips. For across-reboot persistence, enable the systemd
# unit in scripts/systemd/zero-tunnel.service instead.
#
# Usage (on the Pi):
#   GPU_HOST=user@gpu-box bash scripts/pi_tunnel.sh
set -euo pipefail

GPU_HOST="${GPU_HOST:-${1:-}}"
if [[ -z "$GPU_HOST" ]]; then
  echo "usage: GPU_HOST=user@gpu-box bash scripts/pi_tunnel.sh" >&2
  exit 2
fi

FORWARDS=(
  -L 11435:localhost:11434
  -L 9000:localhost:9000
  -L 9100:localhost:9100
  -L 8000:localhost:8000
)

if command -v autossh >/dev/null 2>&1; then
  echo "[pi] autossh tunnel -> $GPU_HOST (auto-reconnect)"
  exec autossh -M 0 -N \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    "${FORWARDS[@]}" "$GPU_HOST"
else
  echo "[pi] autossh not found — using plain ssh (no auto-reconnect)."
  echo "[pi] install it for resilience:  sudo apt-get install -y autossh"
  exec ssh -N \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    "${FORWARDS[@]}" "$GPU_HOST"
fi
