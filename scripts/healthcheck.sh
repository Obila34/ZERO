#!/usr/bin/env bash
# Quick health readout of every ZERO backend.
#
# Run on the PI (default): checks the tunnel-forwarded localhost ports, i.e.
# whether the whole Pi->GPU chain is reachable.
#   bash scripts/healthcheck.sh
#
# Run on the GPU box: pass --gpu so it uses Ollama's real port 11434 (the Pi
# sees it as 11435 through the tunnel).
#   bash scripts/healthcheck.sh --gpu
set -u

LLM_PORT=11435
[[ "${1:-}" == "--gpu" ]] && LLM_PORT=11434

green() { printf '  \033[32m%-4s\033[0m %s\n' "UP" "$1"; }
red()   { printf '  \033[31m%-4s\033[0m %s\n' "DOWN" "$1"; }

check() {  # $1=label $2=url [$3=method $4=body]
  local label="$1" url="$2" method="${3:-GET}" body="${4:-}" code
  if [[ -n "$body" ]]; then
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' -X "$method" \
           -H 'Content-Type: application/json' -d "$body" "$url" 2>/dev/null)
  else
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$url" 2>/dev/null)
  fi
  # 200/404/422/405 all prove the server is listening and answering.
  case "$code" in
    200|404|405|422) green "$label ($url -> HTTP $code)" ;;
    *)               red   "$label ($url -> HTTP ${code:-000})" ;;
  esac
}

echo "ZERO backend health:"
check "LLM   " "http://127.0.0.1:${LLM_PORT}/api/tags"
check "STT   " "http://127.0.0.1:9000/health"
check "TTS   " "http://127.0.0.1:9100/health"
check "Vision" "http://127.0.0.1:8000/health"
check "Detect" "http://127.0.0.1:8000/perceive/detect" POST '{"image_jpeg_b64":""}'
