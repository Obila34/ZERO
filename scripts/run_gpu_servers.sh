#!/usr/bin/env bash
# Start (and keep running) every model server ZERO needs on the GPU node:
#   - Whisper  (STT)     :9000
#   - Orpheus  (TTS)     :9100
#   - Vision   (depth+facts) :8000
#   - Ollama   (LLM)     :11434   (assumed already running as a system service)
#
# Each server is launched detached with setsid + nohup so it survives this shell
# logging out, and is skipped if its port is already listening (idempotent — safe
# to re-run). Logs go to ~/zero_logs/. For true across-reboot persistence use the
# systemd units in scripts/systemd/ instead (see scripts/systemd/README.md).
#
# Usage (from the repo root on the GPU):
#   bash scripts/run_gpu_servers.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ZERO_LOG_DIR:-$HOME/zero_logs}"
mkdir -p "$LOG_DIR"

# Activate the project venv if present (override with ZERO_VENV).
VENV="${ZERO_VENV:-$REPO_ROOT/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

port_up() { # $1=port -> 0 if something is listening
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; } || return 1
}

start() { # $1=name $2=port $3...=command
  local name="$1" port="$2"; shift 2
  if port_up "$port"; then
    echo "[gpu] $name already up on :$port — skipping"
    return
  fi
  echo "[gpu] starting $name on :$port -> $LOG_DIR/$name.log"
  setsid nohup "$@" >"$LOG_DIR/$name.log" 2>&1 < /dev/null &
  disown || true
}

cd "$REPO_ROOT"

# Ollama: usually a systemd service already. Start it detached only if it's down.
if ! port_up 11434; then
  if command -v ollama >/dev/null 2>&1; then
    start ollama 11434 ollama serve
  else
    echo "[gpu] WARNING: ollama not on :11434 and 'ollama' not found — install it" >&2
  fi
fi

start whisper 9000 python server/whisper_server.py --model large-v3-turbo --port 9000

# Orpheus TTS. ORPHEUS_FP16=1 runs the full fp16 vLLM model (best quality +
# fastest streaming, wants most of a GPU); default is the quantized cpp build
# that shares a 16 GB card. Both expose /tts and /tts_stream identically.
if [[ "${ORPHEUS_FP16:-0}" == "1" ]]; then
  start orpheus 9100 python server/orpheus_server.py --port 9100
else
  start orpheus 9100 python server/orpheus_cpp_server.py --port 9100
fi

# Vision (FastAPI app lives in server/vision; run from there so its relative
# imports + config.yaml resolve). uvicorn binds 0.0.0.0:8000.
start vision 8000 bash -c "cd '$REPO_ROOT/server/vision' && exec uvicorn app:app --host 0.0.0.0 --port 8000"

echo
echo "[gpu] launched. Tail logs with:  tail -f $LOG_DIR/*.log"
echo "[gpu] quick check:"
sleep 2
for p in 11434 9000 9100 8000; do
  if port_up "$p"; then echo "   :$p  up"; else echo "   :$p  NOT up yet (check $LOG_DIR)"; fi
done
echo "[gpu] (the LLM model is pulled on first use; run: ollama pull <your model>)"
