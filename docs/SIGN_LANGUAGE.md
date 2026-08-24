# ZERO Sign Language & Motion Architecture

Merged 2026-08-24 from the Pi's uncommitted sign build (preserved verbatim on
branch `pi-snapshot-2026-08-24`), rebuilt on a new motion architecture. The
speech (STT) and intelligence (LLM) engines were deliberately left untouched.

## What ZERO can do now

- **Fingerspell in KSL** (one-handed manual alphabet, bilateral by default):
  "spell PETER", "fingerspell cow", "spell my name" (resolved from the
  recognised speaker). The spoken readout ("P - E - T - E - R") is paced to
  the hands and returned verbatim by the tool (`speaks_directly`) — the
  router never paraphrases it.
- **Single letters**: "sign the letter K", "what is B in sign language".
- **Finger poses**: peace, I-love-you, thumbs up, fist, open/close hand, OK,
  rock on, pinch, wiggle, point — voice or LLM tool.
- **Individual fingers**: "curl your left index", "bend your thumb 40 degrees".
- **Lexicon signs** ("sign hello"): engine + validator are live, the lexicon
  **ships empty by design** — entries come from a KSL signer
  (`data/sign_lexicon.yaml`, checked by `scripts/sign_lexicon_check.py`).
  Until then ZERO refuses honestly and offers to fingerspell.

### Honest limits (mechanical, per-letter metadata in `zero/sign/handshapes.py`)

| Letters | Why approximate |
|---|---|
| R | needs crossed fingers — one flexion servo per finger, no crossing |
| V | needs finger spread (abduction) — reads as U |
| P, Q | K/G pointed *down* — needs forearm rotation (stepper, dark) |
| Z | traced by the index — needs an arm path (stepper, dark) |

The engine speaks each letter as it signs, so spelled words stay unambiguous,
and the system prompt (generated from the engine's real state, never
hardcoded) tells the model to be honest about these.

## The motion architecture

```
 sign > command > gesture > gaze > idle      (fixed track priorities)
                   │
             MotionBus (one clock, zero/motion/bus.py)
   clamp · max-jump walk · deadband · offsets · retry · shared e-stop
                   │
     /api/pose_cmd (12 hand joints, ONE batched post)
     /api/joint_cmd (everything else — parity with the proven drivers)
```

- **One writer.** Head gaze (`head.driver: bus`), gestures (`arms.driver:
  bus`) and sign all write setpoints onto tracks; one tick samples them
  together — that simultaneity is what "the whole body in sync" means.
  `motion.driver: http|null` is the single metal switch.
- **Preempting writes.** A write claims the joint and evicts lower-priority
  standing claims, so a beat that fired mid-sign can never snap the hand
  back when the sign releases; live producers (gaze at 28 Hz) reclaim
  naturally on their next write.
- **Shared e-stop.** `bus.estop()` freezes every track and posts /api/stop —
  the guarantee the three independent drivers never had.
- **Hand truth in code.** Per-side servo calibration lives in
  `zero/arms/hands.py` (read off the gateway firmware): the hands are
  asymmetric (right thumb closes at 70°, left at 140°). Handshapes are
  written in **normalised closure** (0 open .. 1 closed) and expanded per
  side — the flat degree table it replaces drove the right thumb 70° past
  its stop on every closed-thumb sign.

## Signer workflow

1. Author entries in `data/sign_lexicon.yaml` (format documented in-file and
   in `zero/sign/lexicon.py`): hold-move-hold segments of handshape +
   orientation (+ arm joints + head marker, for when those go live).
2. `python scripts/sign_lexicon_check.py` — validates every entry;
   `--play <sign>` prints the exact keyframes against a NullTransport.
3. Review on the robot with the signer watching; flip `review:
   signer-approved`.

## Phase 5 — stepper bring-up (2026-08-24)

- [x] **Steppers enabled** (`arms.allow_steppers: true`). The operator
      accepted the URDF envelopes as-is in place of a fresh supervised
      sweep — they carry real runtime history from the 2026-08-17 live
      period, and boot-zero was verified (arm at rest at the 13:11 EAT
      gateway restart; no stepper commanded since). Signing stance,
      shoulder/elbow gestures and lexicon `arm:` segments are live.
- [x] **Joint-angle black box** (`zero_joints.sqlite`): the MotionBus
      records every acknowledged post with the track that won the joint;
      `scripts/joint_snapshot.py` snapshots gateway telemetry into the same
      table (`--last` prints the newest row per joint). Boot-default
      telemetry rows are excluded, not converted into fictitious angles.
- [x] `scripts/arm_calibrate.py` emission bug fixed: it printed envelopes
      in raw command space, which the driver would have offset-corrected a
      SECOND time (108 deg off on right_bicep). It now emits effective
      degrees verbatim, plus a mirrored-motor question that emits
      `arms.joint_sign` entries.
- [ ] **pose_cmd offset behaviour — still unverified.** Steppers stay on
      joint_cmd (the bus enforces this). `scripts/pose_probe.py
      left_bicep_joint --effective 10` answers it in one supervised minute:
      parks via joint_cmd, fires the identical raw value via pose_cmd — no
      motion = offsets applied, a ~16 deg step = offset-blind.
- [x] **in/out shoulder pair LIVE + signing stance** (operator decision,
      2026-08-24): home = the rest pose it sat on (untouched since boot =
      effective 0 by construction); envelopes one-sided from rest (right
      0..+136.5, left −136.5..0) so the torso side is unreachable by
      clamp. SignEngine now rises into the KSL stance before the first
      letter and lowers after the last, at a stepper-safe 90 dps cap
      (`sign.stance` config; degrades silently to hands-only when the
      steppers aren't registered). Direction follows the URDF/firmware
      convention and is NOT yet motion-verified under ZERO — watch the
      FIRST stance: if an arm presses inward, e-stop and flip the
      envelope + stance signs in config.yaml.
- [ ] Z-trace and P/Q downward orientation need the bicep/forearm rotation
      path authored (lexicon `arm:` segments) — possible now that the
      biceps are live.
