# AF-1 Arm-Pi head services (deploy-ready)

These run **on the Arm Pi** (`arm@192.168.150.183`), co-located with the motor
gateway, so the smoothing/output loop never crosses WiFi. They are the actuation
half of ZERO's split reflex loop.

> **Not deployed/tested on hardware by the author.** The Arm Pi is password-only
> and rejected every available key, so these could not be installed or run
> against real motors here. They are stdlib-only, syntax-checked, and the
> profiler + safe-state were functionally exercised in `--dry-run` on the ZERO
> Pi. **Run the calibration tool first, with a human watching, before trusting
> the reflex service on real metal.** Confirm the `--pan-joint` / `--tilt-joint`
> names match your firmware (the two known stacks disagree — see the plan).

## `af1_gaze_reflex.py` — reflex / smoothing service

Receives gaze setpoints (UDP JSON, from ZERO's `head.driver: udp`) and drives the
neck through the local `:5000` gateway with:
- a **jerk-limited profiler** (bounded velocity/accel/jerk → organic motion, and
  the hobby tilt servo is never step-commanded), and
- a **heartbeat + safe-state**: no setpoint within `--safe-timeout` ⇒ ease to
  home and hold; `{"estop":true}` freezes and forwards `/api/stop`. This is the
  safety guarantee the bare gateway lacks.

```bash
python3 af1_gaze_reflex.py --dry-run        # print, move nothing (inspect first)
python3 af1_gaze_reflex.py                   # drive via localhost gateway
python3 af1_gaze_reflex.py --vmax 40 --amax 200 --jmax 2000 --safe-timeout 0.4
```

Then on the ZERO Pi set `head.driver: udp`, `head.setpoint.host: <arm-pi-ip>`,
`head.setpoint.port: 8099`, `head.enabled: true`.

Install as a service with the systemd unit at the bottom of the script.

## `af1_head_calibrate.py` — supervised calibration (SAFE by default)

Measures the numbers the plan flagged as estimates. **Moves nothing without
`--arm`.** With a human watching:

```bash
python3 af1_head_calibrate.py                 # dry-run: show the plan
python3 af1_head_calibrate.py --arm --sign    # which way does each axis turn?
python3 af1_head_calibrate.py --arm --sweep --film   # slow sweep; record at 120fps
```

It reports command-path latency, guides the sign test (→ set
`head.tracker.pan_sign` / `tilt_sign` on ZERO), and drives a slow sweep to film.
Track a fiducial on the head from the video and feed the angle-vs-time series to
`zero.head.smoothness.summarize(t, angle)` for objective jerk/reversal/overshoot
numbers on the **real** actuator — compare against the NullDriver baseline from
`scripts/head_smoothness.py`.

Mechanical command→motion latency and true max velocity need that video timing;
the gateway's telemetry only echoes the last commanded value, so it cannot
measure them.
