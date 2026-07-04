# ZERO — Always-On + Works-From-Anywhere

Two independent goals, two tools:

| Goal | Tool | Result |
|---|---|---|
| **No downtime** — servers start on boot, restart on crash | **systemd** (`Restart=always`) | power on → everything's just *on* |
| **Pi reaches the GPU on any network** (not same Wi-Fi) | **Tailscale** (WireGuard mesh VPN) | stable private IP reachable through NAT, encrypted, no port-forwarding |

Do Tailscale first (so the tunnel target is reachable anywhere), then systemd.

---

## 1. Tailscale — reach the GPU from anywhere

Install on **both** machines and log both into the **same** Tailscale account. Each
gets a stable `100.x` IP + a MagicDNS name (its hostname), reachable across any
network.

```bash
# On the GPU (zerolabs1) AND the Pi (head):
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Confirm from the Pi that it can reach the GPU by its tailnet name:
```bash
tailscale status                     # both machines listed
ssh obilasam3@zerolabs1 'echo ok'    # must print ok with NO password prompt
```
- `zerolabs1` resolves via MagicDNS from anywhere on your tailnet — that's why
  `GPU_HOST=obilasam3@zerolabs1` in the tunnel unit just works off-LAN.
- If MagicDNS is flaky, use the GPU's Tailscale IP instead:
  `GPU_HOST=obilasam3@100.x.y.z` in `zero-tunnel.service`.
- **Passwordless SSH** is required (the tunnel is unattended). Either
  `ssh-copy-id obilasam3@zerolabs1` once, or enable **Tailscale SSH**
  (`sudo tailscale up --ssh`) so the tailnet handles auth — no keys to manage.

---

## 2. systemd — always on, no downtime

### GPU node (`zerolabs1`, `~/ZERO`) — the three model servers
Ollama installs its own `ollama.service`; leave it. These three add whisper /
orpheus / vision, each with `Restart=always` (a crashed server — e.g. the CUDA
context dying — comes back on its own).

```bash
cd ~/ZERO
sudo cp scripts/systemd/zero-whisper.service /etc/systemd/system/
sudo cp scripts/systemd/zero-orpheus.service /etc/systemd/system/
sudo cp scripts/systemd/zero-vision.service  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zero-whisper zero-orpheus zero-vision
sleep 20 && bash scripts/healthcheck.sh --gpu     # all UP
```

### Pi (`head`, `~/Mzee/ZERO`) — the tunnel (and optionally the app)
`zero-tunnel` keeps the SSH tunnel to the GPU up (auto-reconnect via autossh);
`zero.service` runs `python -m zero.main` and is ordered *after* the tunnel.

```bash
sudo apt-get install -y autossh          # one-time
cd ~/Mzee/ZERO
sudo cp scripts/systemd/zero-tunnel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zero-tunnel
systemctl status zero-tunnel --no-pager  # active (running)
curl -sS -m 5 http://127.0.0.1:8000/health   # tunnel reaches the GPU -> {"status":"ok",...}
```

Now you can just **run the app by hand any time and it connects** — the tunnel
is always up in the background:
```bash
cd ~/Mzee/ZERO && source .venv/bin/activate && python -m zero.main
```

**Or** hand the app to systemd too, so it boots headless with no manual step:
```bash
sudo cp scripts/systemd/zero.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zero
journalctl -u zero -f
```

---

## 3. Day-to-day

```bash
# Status
systemctl status zero-tunnel                     # Pi
bash scripts/healthcheck.sh --gpu                # GPU
bash scripts/healthcheck.sh                      # Pi (through the tunnel)

# Restart everything
sudo systemctl restart zero-whisper zero-orpheus zero-vision   # GPU
sudo systemctl restart zero-tunnel zero                        # Pi

# Logs
journalctl -u zero-whisper -f                    # a GPU server
journalctl -u zero-tunnel -f                     # the tunnel
```

## The guarantee this gives you
- **GPU reboot / server crash** → systemd `Restart=always` brings the servers back.
- **Pi reboot** → `zero-tunnel` (and `zero`) start on boot; the tunnel reconnects.
- **Wi-Fi blip / network change / different location** → autossh + Tailscale
  re-establish the link automatically.
- **You run `python -m zero.main`** → the tunnel is already up, so it connects
  with no manual step, on any network.

## Notes / gotchas
- The tunnel unit's `GPU_HOST` must be the **Tailscale** name/IP for off-LAN use.
- Passwordless SSH (key or Tailscale SSH) is mandatory — a password prompt would
  hang the unattended service. `pi_tunnel.sh` uses `StrictHostKeyChecking=
  accept-new` so a first connection never blocks on a host-key prompt.
- VRAM is the real ceiling on the 16 GB card — see OPERATIONS.md §7 if servers
  start crashing under memory pressure with all models loaded.
