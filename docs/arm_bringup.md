# Arm bring-up run sheet — one joint at a time

Preconditions (once per sitting):
- **AF1 cockpit closed** on every machine — nothing else may command the arm.
- Arm hanging in its natural rest pose.
- If the arm has been moved (by hand or cockpit) since the gateway last
  (re)started: `sudo systemctl restart af1-gateway` on the Arm Pi first —
  a restart re-zeroes the stepper counters at the current pose.
- Steppers are 160:1 geared and will push THROUGH resistance, not stall
  politely. Stop at the first sound or strain, `b` to back off, and mark
  min/max several degrees short of any hard stop.

Per joint, on the head Pi (`cd ~/Mzee/ZERO` first):

    .venv/bin/python scripts/arm_calibrate.py <joint> --stepper

Run in this order, ticking off as you go:

| # | joint | notes |
|---|---|---|
| 1 | `right_bicep_joint`   | |
| 2 | `left_bicep_joint`    | |
| 3 | `right_elbow_joint`   | gateway offset 304 — script compensates, first command is still no-motion |
| 4 | `left_elbow_joint`    | |
| 5 | `right_up_down_joint` | |
| 6 | `left_up_down_joint`  | |
| 7 | `right_in_out_joint`  | |
| 8 | `left_in_out_joint`   | |

In each session:
1. First command asserts **effective 0 — the arm must NOT move**. If it moves,
   quit (`q`) and restart the gateway (zero drifted).
2. `+` / `-` in 2° steps. Note in words which way the joint moves at `+`
   (e.g. "arm swings forward") — write it in the notes column; gestures need it.
3. Mark `min`, `max` (conservative!), `home` (= 0 for a stepper unless the
   rest pose isn't where you want it), then `park`.
4. Paste the printed `joint: {min: …, max: …, home: …}` line into
   `config.yaml` → `arms.joints` (uncomment the matching template line and
   replace the numbers).

After all steppers:
- Set `arms.allow_steppers: true`, `arms.enabled: true`, `arms.driver: http`.
- Fill the `raise_right` / `raise_left` / `handshake` gesture templates in
  `arms.gestures` using the joints + directions you noted.
- Test in `--text`: type `raise your right arm`, `arms down`.
- Run the suite: `.venv/bin/python -m pytest tests/ -q` — must stay green.

Servos (wrists/fingers, "wave", open/close hand) stay blocked until the
PCA9685 servo-power rail is fixed (see head_audit + session notes 2026-08-16:
Nano2 ACKs P commands but no servo on the PCA board moves; the nod works only
because it bypasses the PCA via the direct NOD pin).
