# Neural Gestures — Phase E Build Plan

**ZERO · 2026-08-26 · plan of record → building**

The AI-based gesture generator from LIVING_HANDS_PLAN §Phase E, concretized
against everything the hardware taught us this week.

## The audit that shapes the design

Measured facts (not assumptions) this plan is built on:

1. **Output space is tiny**: 12 DoF — 10 finger closures (0..1, per-side
   asymmetric calibration handled by `zero/arms/hands.py`) + 2 wrist
   pitches. Research models emit 40–60 DoF SMPL-X bodies; our problem is
   ~5× smaller than theirs.
2. **Delivery quantizes**: on the loaded Pi the command cadence degrades to
   ~8–25 Hz (probe, 2026-08-26; watchdog now logs it). A model emitting
   faster than 20 Hz buys nothing; smoothness must come from Pi-side
   interpolation, which already exists.
3. **Look-ahead is free**: sentence audio exists before playout (the same
   tap Living Hands uses). The sidecar sees audio 300 ms+ ahead of the
   speaker at zero added latency.
4. **The GPU fleet is saturated** (zawadi training, vLLM). The build must
   be fully testable with a mock model in-process, and the robot must
   degrade to procedural seamlessly whenever the sidecar is absent — the
   identical pattern remote vision and STT fallback already use.
5. **The A/B gate stands**: neural ships as default only after beating
   procedural in a blind pairwise test on the robot. Everything must be
   switchable per-run (`expression.hands.neural.enabled`).

## Design decision: neural texture, symbolic meaning

The neural model replaces **beats and the floor** — the continuous,
rhythm-locked gesticulation texture that rule systems make robotic. It
does NOT replace **semantic shapes** (counting, palm-up, negation, sizing)
or **sign language**: those carry meaning, and a motion model that
hallucinates meaning is worse than rules. Blend policy, fixed:

    sign (40) > command (30) > gesture (20) > gaze (10) > idle (0)
                                                          └── procedural OR
                                                              neural texture
    semantic envelopes: always procedural, composited over the texture

## Architecture

    Pi (zero/expr/neural.py)                GPU node (server/gesture/)
    ┌──────────────────────────┐            ┌─────────────────────────┐
    │ NeuralGestureClient      │  POST      │ neural_server.py        │
    │  per-sentence session:   │  audio-so- │  GestureModel interface │
    │  tapped audio chunks ────┼───far────► │  ├─ EnergyMockModel     │
    │  frames cache ◄──────────┼──frames────│  ├─ (N3) TCN audio2hand │
    │  staleness watchdog      │  -so-far   │  └─ health/model info   │
    └───────────┬──────────────┘            └─────────────────────────┘
                ▼ frames_for(idx, t_rel) — None = fall back
    HandScheduler._render: neural closure texture if fresh, else
    procedural beats+floor; semantic envelopes composite on top either way

Incremental-stateless protocol (one POST per ~300 ms per sentence, audio
so far in → frames so far out) — no websocket state to leak, retry-free,
and a dead sidecar simply means `frames_for` returns None forever.

## Phases

- **N0 — scaffolding (BUILT NOW)**: client, frames cache, scheduler blend
  hook, EnergyMockModel (in-process + served), degrade path, config, tests.
  Off by default; procedural untouched when off or absent.
- **N1 — sidecar service (BUILT NOW, deploy when VRAM frees)**: stdlib
  HTTP server hosting GestureModel implementations; health endpoint;
  systemd unit template. Runs anywhere; CPU-capable for the mock.
- **N2 — data**: BEAT2 hand channels → 12-DoF retarget (fixed linear map
  fit offline), dataset builder scripts. GPU-free work.
- **N3 — model v1**: small causal TCN/GRU (~2–5 M params), mel+f0+energy →
  closure deltas at 20 Hz, trained on N2 data. Offline eval: beat-alignment
  vs prosody accents, velocity distributions vs BEAT2, before it ever
  touches the robot.
- **N4 — the gate**: blind A/B on hardware (`scripts/gesture_ab.py`),
  black-box traces per run. Neural becomes default only by winning.

## Honesty ledger

- The mock model is a *plumbing test*, energy-driven noise — clearly named,
  never a demo of "AI gestures".
- N3's quality is unknown until trained; if it cannot beat procedural, the
  plan's own gate keeps it a research branch (GENEA says this bar is high).
- Cadence quantization (fact 2) bounds everything: fixing the Pi tick
  priority is worth more than a bigger model.
