# Always-on units (survive reboots)

These make ZERO start itself after a power cycle so the robot is just *on*, with
no manual steps — the seamless goal. The units are pre-filled for the real
machines (GPU `obilasam3@zerolabs1`, repo `/home/obilasam3/ZERO`; Pi user
`head`, repo `/home/head/Mzee/offline_v5`). Edit only if a user/path changes.

## On the GPU node (zerolabs1)

The three model servers (Whisper, Orpheus, Vision). Ollama installs its own
`ollama.service` when installed via the official installer; leave it enabled —
if it is NOT present, `run_gpu_servers.sh` still starts `ollama serve` detached.

```bash
sudo cp scripts/systemd/zero-whisper.service /etc/systemd/system/
sudo cp scripts/systemd/zero-orpheus.service /etc/systemd/system/
sudo cp scripts/systemd/zero-vision.service  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zero-whisper zero-orpheus zero-vision
sudo systemctl status zero-vision     # check it bound :8000
```

## On the Pi (head)

The SSH tunnel (auto-reconnect) and the main app, ordered so the tunnel comes up
first. Requires the Pi's SSH key on the GPU (`ssh obilasam3@zerolabs1 'echo ok'`
must succeed) and `autossh` installed.

```bash
sudo cp scripts/systemd/zero-tunnel.service /etc/systemd/system/
sudo cp scripts/systemd/zero.service        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zero-tunnel zero
journalctl -u zero -f                  # watch it boot and listen for the wake word
```

`Restart=always` on every unit means a crashed server or a dropped tunnel comes
back on its own — including a fresh, state-free Orpheus after a `llama_decode`
crash. That is the "constantly running" guarantee.

## Restarting everything after the units are installed

```bash
sudo systemctl restart zero-whisper zero-orpheus zero-vision   # GPU
sudo systemctl restart zero-tunnel zero                        # Pi
```
