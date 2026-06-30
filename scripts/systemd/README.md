# Always-on units (survive reboots)

These make ZERO start itself after a power cycle so the robot is just *on*, with
no manual steps — the seamless goal. Edit the `User=`, paths, and `GPU_HOST=`
to match your machines, then install.

## On the GPU node

The three model servers (Whisper, Orpheus, Vision). Ollama already installs its
own `ollama.service`; leave it enabled.

```bash
sudo cp scripts/systemd/zero-whisper.service /etc/systemd/system/
sudo cp scripts/systemd/zero-orpheus.service /etc/systemd/system/
sudo cp scripts/systemd/zero-vision.service  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zero-whisper zero-orpheus zero-vision
sudo systemctl status zero-vision     # check it bound :8000
```

## On the Pi

The SSH tunnel (auto-reconnect) and the main app, ordered so the tunnel comes up
first.

```bash
sudo cp scripts/systemd/zero-tunnel.service /etc/systemd/system/
sudo cp scripts/systemd/zero.service        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zero-tunnel zero
journalctl -u zero -f                  # watch it boot and listen for the wake word
```

`Restart=always` on every unit means a crashed server or a dropped tunnel comes
back on its own. That is the "constantly running" guarantee.
