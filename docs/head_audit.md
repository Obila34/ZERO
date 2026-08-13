# Head / Attention System Audit

Date: 2026-08-13. Auditor: Claude (Fable), working from zerolabs1 with SSH to the
head Pi (`~/Mzee/ZERO`, HEAD `b5c8e33`) and the Arm Pi (Tailscale `100.67.233.65`).
Everything below was verified against the running system, the served firmware
source (`server.py` fetched from the Arm Pi), and the test suite — not from prior
claims or logs. Test suite at audit time: **551 passed, 3 skipped**; the
head/gaze-specific files (`test_head_core/head_system/gaze_commands/gaze_social/
gaze_tool/eyes_efference/smoothness`) contain **46 tests**, all passing.

## Environment truth discovered during the audit

These contradict the working assumptions and gate everything else:

1. **The gateway is DOWN.** `server.py` is run manually on the Arm Pi and the Pi
   rebooted; only `wayvnc` and `network-sentinel` are running. Nothing can move.
2. **The Arm Pi's LAN IP changed** (DHCP): it is now `192.168.150.184`, not the
   `.183` hard-coded in `config.yaml` (`head.gateway.base_url`) and in the
   gateway's own proxy constant (`server.py:35`).
3. **The head Pi cannot reach the Arm Pi over the LAN at all** — ping to `.184`
   is 100% loss (AP client isolation or firewall; `.183` is "no route"). The
   only working path today is **Tailscale `100.67.233.65`** (~24 ms RTT, fine
   for 15 Hz posts).
4. **No one was told.** With the gateway dead and the IP stale, the head system
   runs "normally": the driver swallows every failed POST at DEBUG level
   (`driver.py:174`), the controller's belief advances, `status()` reports pan
   angles, and the log looks like a working robot. The known gotcha "the log is
   belief, not reality" is not just a pitfall here — it is institutionalized:
   there is no health signal anywhere in the stack.

## Ranked findings

### CRITICAL

**C1 — Gateway startup physically moves the head (and an arm servo).**
`server.py:84` (`auto_discover_ports`): the port probe writes
`P 0 90\nNOD 90\nM left_elbow_joint 0\nS 0 0\n` to **every** serial port it
finds. On Nano 2 that commands the nod servo to servo-deg **90** and PCA ch 0
(left wrist) to 90. The head was parked at servo 50 with driving disabled;
**every gateway (re)start slams the nod ~40° at uncontrolled servo speed.**
The "parked" state cannot survive a restart, and any calibration ritual that
starts the gateway is itself an unsupervised motion event. Additionally the
keep-alive thread (`server.py:127-137`) rewrites `P 0 90` to Nano 2 every 8 s —
today that coincides with the wrist's calibrated home (offset 90), which is the
only reason it looks harmless.
*Fix:* probe with a benign command (bare newline / `GET_POS`) and identify by
banner; remove `NOD 90` from the probe; make the keep-alive a no-op command.
Must be fixed **before** the first supervised nod calibration.

**C2 — Silent total-failure mode; no link health anywhere.**
(See environment truth above.) `HttpGatewayDriver._post` logs failures at DEBUG
and returns (`driver.py:171-174`); nothing counts failures, nothing surfaces
them in `HeadSystem.status()`, nothing warns the user. The system cannot tell
the difference between "moving" and "unplugged". Combined with C1/H2 this also
creates a **whip hazard on reconnect** (see H2).
*Fix:* track consecutive post failures + last-success time in the driver;
expose `healthy` in `status()`; WARN once on transition to unhealthy and once
on recovery; on recovery, re-approach the current target from the last
*acknowledged* hardware pose instead of jumping (slew in hardware space).

**C3 — No stop path and no watchdog for the nod servo; no watchdog at all.**
`/api/stop` sends `STOP` to Nano 1 only (`server.py:290-292`) — the e-stop
cannot stop or relax the nod servo. The gateway has no heartbeat: on WiFi drop,
process kill, or head-side crash, the last commanded pose persists
indefinitely (`HeadSystem.stop()` centers for only 0.2 s before killing the
loop — from 80° at 220°/s that ease-home cannot complete, `system.py:174-177`).
A hobby servo left commanded against a linkage bind draws stall current
indefinitely with no way to kill it short of pulling power.
*Fix:* (a) firmware/gateway: extend `/api/stop` to Nano 2 (servo detach if the
sketch supports it); (b) make shutdown ease-home blocking-until-arrived (with
timeout); (c) keep the nod's software envelope well inside the mechanical
stops so a stale command is never a straining command.

### HIGH

**H1 — Failed posts are never retried; hardware silently freezes at a stale pose.**
`driver.py:155-160`: `_sent` is updated *before* `_post()`, unconditionally. If
the POST fails (WiFi blip, gateway busy), the deadband check compares future
targets against the never-delivered value, so an unchanged target is **never
re-sent**. A discrete command ("turn left" → hold) can simply not happen while
belief says it did.
*Fix:* update `_sent` only on HTTP success; on failure leave `_pending` set so
the worker retries on the next wake.

**H2 — Slew limiting exists only in belief space → reconnect/backlog jumps.**
The controller slews its *internal* position and the driver posts absolute
targets. If posts fail for a while (H1/C2) and then succeed, the hardware
receives a single absolute step of arbitrary size — the stepper executes an
up-to-160° jump at whatever profile the Nano firmware applies (unknown; the
gateway just forwards steps, with **no software limit on pan**,
`server.py:334-338`). This is the "whipping" failure mode, and it is reachable
today via any transient outage.
*Fix:* same as C2 recovery-walk; optionally clamp per-post delta in the driver.

**H3 — The social/expressive layer is dead code in the live pipeline.**
`grep` over `zero/` shows **no callers** of `HeadSystem.express()`,
`.gesture()`, `.on_sentence_end()`, or `FaceTracker.set_attention()` outside
tests. `set_state()` *is* wired (`main.py:487-489`), but the GazeScheduler's
output is only consumed in the `input: "face"` branch of `_source_tick`
(`system.py:388-394`) — and config says `input: head`. Net effect in the
current config: **no social gaze, no aversions, no thinking look-up, no
gestures, no sentence-end look-back ever reach the neck.** The three-layer
attention architecture (reflex/attention/deliberate) has, as deployed,
collapsed into teleop + voice commands. The 71%/41% Argyle rhythm exists only
in unit tests.
*Fix:* wire `express()`/`on_sentence_end()` into the reply path in `main.py`;
make social bias compose with head/hand-follow or document that it is
face-mode-only; revisit `input` default once the nod works.

**H4 — FaceTracker is not used in the current configuration.**
With `head.input: head`, `_source_tick` routes to `_hand_tick` and returns
(`system.py:385-387`). "Face tracking works" is true only when `input: face`.
The tracker also carries a full tilt path (`tracker.py:297`) that is sound in
sim but has never seen hardware.

**H5 — Tilt has no calibrated envelope in the controller (nod-enable blocker).**
`HeadController` clamps tilt to symmetric `±limit_deg` (±80°); only the driver
clamps to the safe nod window (`driver.py:159`). So the moment `drive_nod` goes
on, belief can sit at tilt +80 while hardware pins at servo 90: efference copy
reports motion that didn't happen, the tilt tracker integral winds up against
the clamp, and "look down" appears to work in the log while the servo never
moves. `set_calibration()` already supports asymmetric per-axis windows
(`controller.py:151-174`) — **nothing calls it with config values.**
*Fix:* add `head.tilt_min_deg`/`tilt_max_deg` (from calibration) and apply via
`set_calibration()` at build time, so belief and hardware share one envelope.

**H6 — Gateway serial writes are unsynchronized across HTTP threads.**
`ThreadingHTTPServer` + `send_serial_target` (`server.py:139-153`) means two
concurrent `/api/joint_cmd` posts (driver + web UI + keep-alive thread) can
interleave bytes on one serial port and corrupt command framing. The head
driver serializes its own posts through one worker, which mitigates but does
not close this. Also: on any write error the port is closed and **never
reopened** (`server.py:150-151`) — the axis silently degrades to "Sim" prints
until a manual restart.
*Fix (gateway):* a per-port `threading.Lock` around writes; reopen on error.

### MEDIUM

**M1 — Tier-1 gaze parser hijacks unrelated utterances.**
`commands.py:38`: `_HOLD` fires on *any* utterance containing "stop", "hold
on", "halt", "stay put" — "hold on, what's the time?" returns "Okay, holding
here." and the question is never answered. `_DIR` + the wide filler list
(`commands.py:46-55`, includes "it") makes "turn it up" (volume!) a 25° look-up
command. The router runs this on every utterance (`router.py:274-282`).
*Fix:* require a head/look context word (or a short imperative ≤3 words) for
`_HOLD`; exclude pronoun-object phrasings like "turn it up/down"; add
regression tests for the false positives.

**M2 — `_set_command` ordering race can drop a command.**
`system.py:283-287` writes `_cmd_target` *before* `_cmd_until`; the 15 Hz
source tick (`system.py:378-384`) reads target-then-until. Interleaved, a new
command is seen with the previous (expired) deadline and cleared on the same
tick. Small window, real race, no lock.
*Fix:* write `_cmd_until` before `_cmd_target` (or guard with a lock).

**M3 — `FULL_DEG=999` sentinel leaks into state.**
`look_direction` stores the raw 999 in `_cmd_target` (`system.py:229-231`);
the controller clamps actual motion (safe), but `look_and_settle` waits for
`position ≈ 999` and always burns its full 1.2 s timeout (`system.py:252-264`),
and dbg/aim values are nonsense during such commands.
*Fix:* clamp the target into the controller envelope at `_set_command` time.

**M4 — Motion is slew-only; accel/jerk are unbounded. Measured, not eyeballed**
(`scripts/head_smoothness.py`, config values 220°/s @ 40 Hz):

| scenario | peak speed | peak accel | peak jerk | spectral conc. | verdict |
|---|---|---|---|---|---|
| saccade 0→40° | 220°/s | 4 400°/s² | 88 000°/s³ | 0.80 | CHECK |
| small look 0→8° | 160°/s | 3 200°/s² | 128 000°/s³ | 0.56 | CHECK |
| return 40→0 | 220°/s | 4 400°/s² | 88 000°/s³ | 0.81 | CHECK |
| tilt 0→−12° | 220°/s | 4 400°/s² | 168 000°/s³ | 0.61 | CHECK |

The harness's own bar is spectral concentration >0.85; every scenario fails it.
Velocity steps instantly to the 220°/s ceiling (rectangular profile). Whether
the Nano's AccelStepper ramps this is unknown (its sketch was not auditable);
the hobby nod servo definitely does not.
*Fix:* accel-limited (trapezoidal) profiling in `HeadController._step` — a
velocity state + accel clamp, ~10 lines — then re-measure.

**M5 — Shutdown/restart hygiene.** `HeadController.stop()` doesn't join and a
stopped controller can't be restarted (`start()` early-returns on the dead
thread object, `controller.py:126-131`). `HeadSystem.stop()`'s 0.2 s ease-home
(see C3) leaves the head off-center at the gateway forever.

**M6 — Config keys that do nothing.** `head.hand.keypoint` is ignored —
`system.py:109` derives the keypoint from `head.input`. `command_deg` default
duplicates `DEFAULT_DEG` in `commands.py`. `social.up_deg` and the whole
`social:` block are inert in the current input mode (H3). `head.trace` applies
to the null driver only (documented, fine).

**M7 — Vestigial architecture: the UDP reflex path.** `UdpSetpointDriver`
(`driver.py:189-231`) targets an Arm-side `af1_gaze_reflex.py` service
(referenced in `scripts/head_say.py:104`) that **does not exist** on the Arm Pi
(directory listing checked). It is the *documented* "recommended architecture"
(jerk-profiling + heartbeat on the Arm side) but was never built. Keep the
driver as the target architecture or delete it; today it is a trap (selecting
`driver: udp` moves nothing, silently).

### LOW

**L1 — Driver posts both joints when either passes the deadband**
(`driver.py:151-160`) — with `drive_nod` on, every pan twitch also posts a nod
command to the second serial bus. Gate per-axis to halve Nano 2 traffic.

**L2 — Telemetry is an echo, and nothing consumes it.** `/api/telemetry`
returns the last *commanded* values (`server.py:314-318`). Fine as long as no
one mistakes it for sensing; worth a comment in config.

**L3 — `zero/head/hand.py` has zero tests** — OneEuro, junk rejection
(`min_shoulder_frac`), `_head_yaw` fallback logic, mirror/gain: all untested.
The junk-rejection guard is load-bearing (known model hallucination failure).

**L4 — Persona actively teaches the LLM it cannot move.**
`zero/llm/persona.py` SYSTEM_TEMPLATE: "NEVER narrate actions… **You have no
way to perform them**" — with nothing anywhere saying ZERO has a movable head.
The terse gaze tool line in the spec block is the only counter-signal, so
movement requests that miss tier-1 get "I can't move my head" refusals. (Also,
`GazeTool.run` returns exactly that sentence when the head is genuinely absent,
`gaze.py:57-58` — correct there.)
*Fix:* one persona line ("You have a neck; to look somewhere, call the gaze
tool — never claim you can't move") gated on the head being enabled.

**L5 — Test-count claims drift.** Prior notes claim "57 head/gaze tests"; the
seven head/gaze files contain 46. Not a defect — flagged for honesty.

## The "make it feel wrong" failure catalog — defended vs claimed

| Failure | Status |
|---|---|
| Revolving / hunting | **Defended in code** (settle-gate `tracker.py:287-299`, saturation reset `:301-329`, hysteresis latch `:240-246`) and partially tested — but only in face mode, which is off (H4). |
| False novelty after a pan | **Defended and tested** (efference copy `system.py:432-468`, Eyes suppress/resettle `eyes.py:401-441`, `test_eyes_efference.py`). |
| ID fragmentation | Out of head scope; nothing here defends it. |
| Sign inversion | **Procedure only** — config signs + the documented one-look test + saturation-reset keeps motion alive; no automated detection. Acceptable. |
| Catatonic freeze | Micro-idle nudge exists (`tracker.py:246-271`) but lives in the unused face mode; in `input: head`, a lost pose recenters and the head then sits dead. **Claimed, not delivered, in the current config** (H3/H4). |
| Whipping mid-sentence | Aversions are small (12°) and slewed — but socially dead (H3); the real whip risk is H2 (reconnect jump), which is **undefended**. |

## Test coverage summary

Well covered: controller core (slew/clamp/gates/gestures), command parsing
(happy paths), social scheduler rhythm, efference copy, smoothness metrics,
gaze tool dispatch, system-level command/hold/resume.
Not covered at all: **both hardware drivers** (`HttpGatewayDriver`,
`UdpSetpointDriver`, `make_driver`), driver failure/retry behavior, `hand.py`
(L3), parser false positives (M1), the `_set_command` race (M2), `FULL_DEG`
handling (M3), tilt-envelope calibration plumbing (H5).

## Recommendations, in order

1. **Before any nod work:** fix C1 (gateway probe motion) on the Arm Pi, with
   the user present for the first restart; fix the stale `base_url` to the
   Tailscale address `http://100.67.233.65:5000` (DHCP-proof; LAN is currently
   blocked anyway); get the gateway under systemd so a reboot doesn't silently
   decapitate the robot.
2. Driver honesty: H1 retry-on-failure, C2 health surfacing, H2 recovery walk.
   Add driver unit tests against a local fake gateway.
3. Nod calibration (supervised, small steps) → fills `nod_min/max/offset_deg`,
   `head.tilt_min/max_deg` (H5), home pose; then `drive_nod: true` with
   `track_tilt` still off; then tilt into tracker/teleop/social behind config.
4. M1 parser fixes + tests (it hijacks real conversations today).
5. M4 accel-limited profiling, verified by the smoothness harness.
6. H3: wire expression/social into the reply path once the nod is live (the
   gestures are mostly tilt moves — they were unshippable before the nod).
7. L4 persona line, M2/M3 small fixes, M6 config cleanup, M7 decision on the
   UDP path.
