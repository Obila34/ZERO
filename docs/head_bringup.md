# Head bring-up protocol (one session, ~15 min)

The nod is calibrated (2026-08-13: servo 50–140, larger servo = UP, park =
max = servo 140, log in `calibration_nod_log.txt`) and the calibrated values
are live in `config.yaml`. This is the single supervised session that verifies
motion and then progressively lights up the dark features. Run it top to
bottom; stop at any step that fails and fix before continuing.

## 0. Power-on state

- **Arm Pi**: `sudo systemctl enable --now af1-gateway` (one instance only —
  never also run `python3 server.py` by hand; duplicates now fail loudly).
  `sudo journalctl -u af1-gateway -f` shows the serial TX lines.
- Close the AF1 cockpit app (the machine streaming joint commands) for the
  duration — a twitching arm confuses every judgement below.
- **Caveat**: after a servo power cycle the nod's position is unknown. The
  head system's first post asserts the park (servo 140) as ONE uncontrolled
  snap — keep hands clear of the head at step 1.

## 1. Nod motion check (head Pi)

    cd ~/Mzee/ZERO && .venv/bin/python -m zero.main --text

| type | expect |
|---|---|
| *(startup)* | head settles at the park (fully raised) |
| `look down 40 degrees` | clear dip, eases back up after ~4 s |
| `look down 10 degrees` | small dip, same auto-return |
| `look up` | says okay, does NOT move (park = ceiling, by design) |
| `turn left` then `face forward` | pan unregressed, both axes home |

If "look down" moves UP: flip `head.gateway.nod_sign` to -1 — but that
contradicts the calibration session's eyeball check, so re-verify first.

## 2. Face tracking with tilt (needs the camera)

In `config.yaml` set `head.input: face` and `head.track_tilt: true`; restart.
Stand in view, then crouch so your face sits LOW in the frame: the head must
tilt DOWN and the error must SHRINK (the one-look sign test — watch
`servo: ex=… ey=…` in the log; ey must decay toward 0, not grow). Stand tall
again: head returns toward the park (up-travel is clamped at the ceiling).
If ey GROWS while the head moves: set `head.tracker.tilt_sign: -1`.

## 3. Vertical teleop

Back to `head.input: head` (or `hand`), set `head.hand.tilt: true`; restart.
Nod your own head down (or lower your raised hand): ZERO's head follows down
and recenters when you level out. Vertical range is `hand.tilt_range_deg`
(15° default, deliberately small).

## 4. Expression

Set `head.express: true`; restart, converse by voice. Expect small nod/shake
beats during replies and the look-back at sentence ends. Gestures with upward
components are clamped at the park — visible ones are the downward beats.

## 5. Afterwards

- `git add -A` the config changes and commit (the calibrated config is
  currently deployed but uncommitted).
- `git push` needs GitHub credentials on the head Pi (42+ local commits).
- Optional widening later: lower `head.gateway.nod_offset_deg` below 50 to
  rest the head a few degrees under the ceiling and win back up-gestures
  ("look up", greet up-nod, think look-up) — retest step 1 after.
