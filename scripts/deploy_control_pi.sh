#!/usr/bin/env bash
# Deploy the AF1 fusion surface (zero-control) to the head Pi — one shot,
# idempotent. Run from anywhere with key-auth SSH to the Pi:
#
#   bash scripts/deploy_control_pi.sh [head@192.168.150.202]
#
# What it does (and nothing else):
#   1. scp zero/control.py + zero/main.py to ~/Mzee/ZERO on the Pi
#   2. append the `control:` block to the Pi's config.yaml IF it's not there
#      (the Pi's live config is never overwritten — audio tuning stays intact)
#   3. ensure ffmpeg exists (decodes AF1's webm/opus mic uploads)
#   4. restart zero.service and probe /health + /zero/status
set -euo pipefail

PI="${1:-head@192.168.150.202}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-o ConnectTimeout=10)

echo "== 1/4 copy fusion files -> $PI:~/Mzee/ZERO"
scp "${SSH_OPTS[@]}" "$DIR/zero/control.py" "$PI":Mzee/ZERO/zero/control.py
scp "${SSH_OPTS[@]}" "$DIR/zero/main.py"    "$PI":Mzee/ZERO/zero/main.py

echo "== 2/4 ensure control: block in the Pi's config.yaml (no overwrite)"
ssh "${SSH_OPTS[@]}" "$PI" '
  set -e
  CFG=~/Mzee/ZERO/config.yaml
  if grep -q "^control:" "$CFG"; then
    echo "   control: block already present — leaving config untouched"
  else
    cat >> "$CFG" <<EOF

control:
  # The AF1 fusion surface (zero/control.py) — HTTP control plane inside the
  # ZERO process so the AF1 cockpit can push a talk turn / a spoken line at
  # the SAME brain the mic loop runs. LAN-only; rides zero.service.
  enabled: true
  host: 0.0.0.0
  port: 8090
  person_id: 1
EOF
    echo "   control: block appended"
  fi
'

echo "== 3/4 ensure ffmpeg (decodes AF1 mic uploads)"
ssh "${SSH_OPTS[@]}" "$PI" \
  'command -v ffmpeg >/dev/null && echo "   ffmpeg present" || sudo apt-get install -y ffmpeg'

echo "== 4/4 restart zero.service + probe"
ssh "${SSH_OPTS[@]}" "$PI" '
  sudo systemctl restart zero
  for i in $(seq 1 30); do
    sleep 2
    if curl -sf http://127.0.0.1:8090/health >/dev/null; then break; fi
  done
  echo "-- /health:";      curl -sf http://127.0.0.1:8090/health || true; echo
  echo "-- /zero/status:"; curl -sf http://127.0.0.1:8090/zero/status || true; echo
'
echo "DONE — AF1 can now reach ZERO at http://${PI#*@}:8090"
