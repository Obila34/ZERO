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

## Phase 5 — stepper bring-up (NOT in this build, supervised only)

Prereqs before `arms.allow_steppers: true` goes back on:
- [ ] verify whether `/api/pose_cmd` applies the stored stepper offsets the
      way `/api/joint_cmd` does (unverified — the bus deliberately keeps
      steppers on joint_cmd with offset subtraction until then)
- [ ] supervised per-joint calibration (`scripts/arm_calibrate.py`), human
      watching, small steps
- [ ] confirm resting pose vs URDF zero (no encoders — zero is wherever the
      Nano booted), set `arms.limit_frac` accordingly
- [ ] then: signing stance (shoulder/elbow raise), Z-trace, P/Q downward
      orientation, and lexicon `arm:` segments come alive automatically —
      the sign engine already emits them and the bus drops them with a log
      until the joints register.
