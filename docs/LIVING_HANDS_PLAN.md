# Living Hands — Architecture Plan

**ZERO · Physical AI · 2026-08-25 · plan of record, awaiting review**

A strictly additive layer that makes ZERO's hands and wrists move *with* its speech —
prosody-timed, semantically shaped, anticipatory — without changing one line of behavior
in the build that exists today.

---

## 00 · The thesis

The single most important experimental result in this field is not about motion models.
The GENEA challenge — the largest controlled evaluation of co-speech gesture systems ever
run — found that state-of-the-art generated motion can be rated *more human-like than
motion capture*, yet every synthetic system was judged **"vastly less appropriate for the
speech"** than the real thing. Human-likeness is nearly solved; *appropriateness* — the
sense that this movement belongs to these words at this instant — is the open problem.

Appropriateness lives in two places: **timing** (the stroke lands on the stressed
syllable, measured at roughly ±17 ms in human speakers) and **semantics** (the hand shape
means what the words mean). Both are tractable with deterministic signal processing and a
semantic mapper — no GPU required — which is why this plan builds a procedural core first
and treats motion-diffusion models as an optional refinement sidecar, not the foundation.

The second pillar: **ZERO already owns the hard parts.** The MotionBus priority arbiter,
the normalised-closure hand model, the minimum-jerk profile library, the per-sentence
expression hook, and the joint-angle black box were all built in the last two build
phases. The new layer is a fifth *producer* on an arbiter designed for exactly this — it
does not rewrite anything, and the arbiter's arithmetic guarantees it never wins a
conflict against the existing build.

## 01 · Research digest

### Timing is tighter than intuition says

- Gesture apices and pitch accents align to within **~17 ms** on average in American
  English speakers — the prominent syllable entrains the gestural stroke (JASA; J.
  Phonetics 2026).
- Preparation begins **200–350 ms before** the stress peak — the anticipatory
  co-articulation. The stroke may lead; it must never lag.
- Listeners *perceive* beats as more aligned than they physically are — perception is
  forgiving of a lead and punishing of a lag, which sets our asymmetric error budget.
- Gesture and speech are bidirectionally coupled — gesturing measurably changes
  articulation (Sci. Reports 2024) — supporting McNeill's growth-point view that they
  decouple from one idea unit rather than one following the other.

### The generation landscape, honestly assessed

- **Diffusion family**: EMAGE (masked audio-gesture modeling, CVPR 2024), EmoDiffGes
  (2025), GestureLSM (latent-shortcut, real-time on desktop GPU), DIDiffGes, and — most
  relevant to us — **Accelerated Rolling Diffusion** (2025), which generates gesture
  *streams* with a rolling denoising window: the right shape for a robot that speaks in
  sentences it hasn't finished generating.
- All of these target 40–60 DoF SMPL-X bodies at desktop-GPU rates. None runs on a Pi 4.
  ZERO's hands expose **12 physical DoF** (5 finger-curl + 1 wrist-pitch per side) — a
  retargeting problem, and also a massive simplification in our favor.
- **Rule/procedural family**: from Cassell's BEAT toolkit (2001) to modern hybrid entries
  in GENEA 2022/2023, systems that put deterministic timing and semantic rules first
  remain competitive on appropriateness — the metric that is actually unsolved.

> **Design consequence.** Build the timing spine and semantic mapper procedurally on the
> Pi (microseconds of numpy per sentence, deterministic, testable), and reserve diffusion
> for a GPU sidecar that *textures* that spine later — gated behind an A/B test it has to
> win.

## 02 · What ZERO already provides

Inventory of existing assets the layer plugs into — read-only, none modified:

| Asset | Where | What it gives the layer |
|---|---|---|
| MotionBus `idle` track | `zero/motion/bus.py` | Priority 0, reserved "for breathing sway" — **defined and never used**. The layer writes here. Sign (40), commands (30), gestures (20) and gaze (10) preempt it by arithmetic: the non-interference guarantee is structural, not procedural. |
| Closure-space hand model | `zero/arms/hands.py` | Portable 0–1 per-finger curl + symbolic wrist orientation, expanded against each hand's real (asymmetric) servo calibration. The layer thinks in closure; hardware safety is inherited. |
| Min-jerk profile | `zero/motion/profile.py` | The bell-velocity easing every existing motion uses — imported, not duplicated. |
| Per-sentence hook | `arms.express(sentence)` call in `main.py` | Already fires for every spoken sentence at playout. The layer taps the same rendezvous. |
| ~1.3 s synthesis window | TTS pipeline (Kyutai, streaming) | Audio exists in the playout buffer *before* the speaker plays it. This pre-existing interval is the anticipation budget — the layer adds zero latency because it spends time ZERO already waits through. |
| Joint black box | `zero_joints.sqlite` | Every acknowledged command, timestamped, per track. The free measurement instrument: stroke-apex-vs-accent offset is one SQL join away. |
| Room sense | `audio.room` ambient RMS | Gate for servo audibility — micro-motion amplitude ducks when the room is quiet enough for the mic to hear the PCA servos whine. |
| Cue vocabulary + inference | `zero/arms/cues.py` | McNeill-grounded classification (beat / deictic / iconic / emblematic) with stroke-on-word scheduling. The layer consumes its classification; it does not replace it. |

## 03 · Architecture

One new package — `zero/expr/` — five stages, one output.

```
  LLM reply ──► TTS synth ──► playout queue ──► speaker     (existing, unchanged)
      │             │               │
      ▼ tap:text    ▼ tap:audio     ▼ tap:playout-start clock
 ┌────────────┐ ┌─────────────────┐
 │ semantic   │ │ prosody analyzer│
 │ mapper     │ │ energy·onsets·f0│
 │ (cues.py   │ │ → accent times  │
 │  signal)   │ └────────┬────────┘
 └─────┬──────┘          │
       ▼                 ▼
 ┌───────────────────────────────┐   ┌──────────────────────┐
 │ kinematic scheduler           │◄──│ micro-motion         │
 │ closure keyframes,            │   │ 1/f noise, breathing │
 │ apex on accent, prep −250 ms  │   │ drift, room-ducked   │
 └───────────────┬───────────────┘   └──────────────────────┘
                 ▼  writes idle track (priority 0)
 ┌─────────────────────────────────────────────────────────┐
 │ MotionBus — existing arbiter                            │
 │ sign 40 > command 30 > gesture 20 > gaze 10 > idle 0 ◄──│── Living Hands
 └─────────────────────────────────────────────────────────┘
```

### Stage 1 — Tap

The layer needs exactly three signals, all of which exist in flight today: the sentence
text (known before synthesis), the synthesized audio (sitting in the playout buffer), and
the instant playout begins (the clock anchor). Total contact surface with the existing
build: **two one-line event publications** on the EventBus ZERO already runs, plus one
config block. Feature-flagged — with `expression.hands.enabled: false`, the running
system is bit-identical to today's build.

### Stage 2 — Prosody analyzer (DSP, no ML)

Per audio chunk: RMS energy envelope at a 20 ms hop, spectral-flux onset detection, and
autocorrelation pitch. Prominence candidates are local maxima of energy × f0 excursion —
the classic acoustic correlates of stress. Numpy arithmetic measured in single-digit
milliseconds per sentence on the Pi 4; no model loads, no GPU, no added latency.

### Stage 3 — Semantic mapper

Consumes the sentence text through the classification `cues.py` already performs, and
adds a hands-specific vocabulary rendered in closure space:

| Trigger | Hand behaviour | Gesture class |
|---|---|---|
| enumeration — "first… second… three things" | finger count on the raised-letter hand shapes ZERO already owns | iconic/beat |
| sizing — "huge", "tiny", "this big" | aperture: both hands' closure scales with magnitude | iconic |
| epistemic — "maybe", "I think", "who knows" | palm-up open hand (wrist to `up`, fingers extended) | emblematic |
| negation — "no", "never", "not that" | brief wrist rotation away, fingers loose | emblematic |
| temporal sweep — "over time", "from … to …" | slow wrist arc through orientation space | metaphoric |
| prosodic accent, no semantic trigger | micro-beat: 0.05–0.12 closure pulse, apex on the accent | beat |

Everything is expressed as closure fractions and symbolic orientations, so the per-side
servo asymmetry (right thumb 70°, left 140°) is handled by the existing hand model — the
mapper never sees a degree.

### Stage 4 — Micro-motion (the "alive" floor)

Underneath discrete gestures, a continuous 1/f-noise process drifts finger closures
within ±0.03 and the wrists within a few degrees at breathing rate — the difference
between a mannequin holding still and a body at rest. Two governors: amplitude ducks
against the room-sense ambient RMS (PCA servos are audible in a quiet room, and the mic
is close), and the process writes at 2–4 Hz, far below the bus tick, so its bus cost is
negligible.

### Stage 5 — Kinematic scheduler → idle track

Merges semantic keyframes, prosodic beats and the micro-motion floor into a single
timeline of closure frames, eased with the shared min-jerk profile, and written to the
bus `idle` track. Conflict policy is inherited, not invented: when ZERO signs, gestures,
obeys a command, or is e-stopped, those tracks outrank idle and the hands belong to them
— the release semantics and preempting-write rule already in the bus handle every
interleaving.

## 04 · Anticipatory timing

```
 sentence known      playout starts        prep launch   accent ★
      │                    │                    │            │
      ├─ synthesis wait ───┤─ audio playing ────┼────────────┼────────►
      │  (~1.3 s, exists)  │                    │            │
      └─ analysis runs     │                    ├── prep ────┤─ stroke ─┤
         here — FREE       │                    │◄─ 250 ms − wire latency ─►
```

1. **The look-ahead is free.** Audio is synthesized before it plays; the interval between
   synthesis and playout (TTFA ≈ 1.3 s, plus queue depth for later sentences) already
   exists. Analysis runs inside it, in parallel with a wait the system already performs.
   Nothing is delayed to buy anticipation.
2. **Schedule backwards from the accent.** For each prominence at playout-relative time
   `t★`, the scheduler launches preparation at `t★ − 250 ms` and shapes the min-jerk
   segment so peak velocity — the perceptual apex — lands at `t★ − L`, where `L` is the
   measured command-to-motion latency of the gateway + PCA chain (expected 40–80 ms;
   Phase A measures it rather than guessing).
3. **Streaming reality.** Kyutai streams — later chunks of a sentence may not exist when
   the first chunk plays. The analyzer therefore runs as a rolling process over the
   playout buffer, always analyzing 300–500 ms ahead of the playout cursor. When the
   buffer runs shallower than the required lead, the layer degrades per-accent: the beat
   fires on detection (apex ≈ +30 ms) instead of leading. Perception punishes lags, not
   leads; a near-zero apex is inside tolerance, and the black box will tell us how often
   it happens.
4. **The error budget is asymmetric**, matching perception: target median
   |apex − accent| ≤ 50 ms, hard ceiling on lag of 100 ms, no ceiling on graceful lead —
   a slightly early hand reads as intent, a late one as puppetry.

## 05 · The constraint, made structural

> **Nothing existing changes.** Touched: one new package (`zero/expr/`), two one-line
> event publications, one config block, default *off*. Untouched: MotionBus, ArmSystem,
> SignEngine, head system, TTS, STT, LLM, router, tools, config semantics — everything.
> The guarantee is structural, not disciplinary: the layer's only output channel is the
> priority-0 idle track, which every existing behavior preempts by the bus's fixed
> arithmetic. Turn the flag off and the running system is bit-identical to today's.

Interaction cases, resolved by the existing arbiter with no new policy:

- *ZERO starts fingerspelling mid-beat* → sign (40) claims the hands; idle writes are
  outranked instantly; the preempting-write rule evicts the layer's stale claims. When
  the sign releases, the layer's next micro-motion write re-adopts the hands from
  wherever they rest.
- *Operator command / gesture cue fires* → same story at priorities 30/20.
- *E-stop* → the bus freezes every track including idle. Inherited.
- *A hand is holding something* → the layer reads `hand_state` (read-only) and silences
  that side, same convention the gesture layer follows.

## 06 · Phased plan

### Phase A — Timing spine & instrumentation

Build the tap, the playout clock, and the scheduler skeleton against NullTransport.
Measure — don't assume — the two constants everything depends on: command-to-motion
latency of the gateway+PCA chain, and real playout-buffer depth under streaming TTS.
Ship a measurement notebook: black-box joint rows joined against audio-accent times.

*Gate: apex-placement error measurable end-to-end · risk: low · moves metal: no*

### Phase B — Prosodic beats + the living floor

DSP analyzer → micro-beats landing on accents; 1/f micro-motion with room-sense ducking.
This phase alone delivers most of the perceptual win — hands that breathe and pulse with
the voice. First hardware validation: beats visible, servos inaudible on the mic, apex
error within budget on the black box.

*Gate: median |apex−accent| ≤ 50 ms measured on hardware · risk: low*

### Phase C — Semantic hand vocabulary

The mapper table above, rendered in closure space: counting, aperture sizing, palm-up
epistemic, negation, temporal sweeps. Extend the LLM's inline-cue vocabulary additively
(new cues, none removed) so the model can also *ask* for a hand shape the way it already
asks for [wave]. Rate-governed by the same pacing philosophy as the gesture layer: most
sentences carry only the floor and the beats.

*Gate: blind A/B vs phase-B build on 20 utterances · risk: low*

### Phase D — Co-articulation polish

Kendon-unit chaining across sentence boundaries (hands stay lifted between consecutive
gestural sentences), anticipatory prep across the sentence queue (the layer can see
sentence N+1's text before N finishes playing), retraction easing tuned against video
review.

*Gate: side-by-side video review with operator · risk: low*

### Phase E — Diffusion sidecar *(optional, gated)*

A rolling-diffusion gesture model (streaming variant, per the 2025 literature) on the
idle RTX 5070 Ti (zerolabs0), conditioned on the same tapped audio+text, emitting 12-DoF
closure trajectories at 30 Hz over the existing tunnel pattern. Retargeting: SMPL-X hand
pose → closure space via a fixed linear map fit offline on the BEAT2 dataset. The Pi
blends sidecar frames with the procedural spine and **falls back to procedural
seamlessly** when the GPU is absent — the identical degradation pattern remote vision
uses today. Entry condition: it must beat the procedural build in a blind pairwise test,
or it stays a research branch. GENEA says that bar is higher than it sounds.

*Gate: wins blind A/B on appropriateness · risk: medium · GPU: zerolabs0*

## 07 · Risks, stated plainly

| Risk | Reality | Mitigation |
|---|---|---|
| Servo audibility | PCA servos whine; the mic is centimeters away; self-noise can leak into VAD/wake word | Room-sense ducking (built into stage 4); Phase B explicitly measures mic bleed before enabling the floor by default |
| Servo wear | Continuous micro-motion is duty cycle the hands were not budgeted for | ±0.03 closure at 2–4 Hz is near-zero load; amplitude budget in config; black box gives exact motion-per-hour accounting |
| Pi CPU | Main loop already carries VAD (~8%), wake word, TEN, tracker | Analyzer is numpy DSP, single-digit ms per sentence; budget test in Phase A with the same stage-timer discipline main.py already uses |
| Streaming buffer too shallow for lead | Kyutai chunk cadence varies | Per-accent graceful degradation to on-detection beats (§04); frequency measured, not assumed |
| Gesture spam | Over-gesturing reads as nervous — the existing layer's hard-won lesson | Inherit the pacing philosophy: the floor is subtle, beats are small, semantic shapes are rate-limited exactly like cues |

## 08 · Sources

- [GENEA Challenge 2022 evaluation](https://arxiv.org/abs/2303.08737) · [GENEA 2023](https://dl.acm.org/doi/10.1145/3577190.3616120)
- [The timing of speech-accompanying gestures with respect to prosody (JASA)](https://pubs.aip.org/asa/jasa/article/115/5_Supplement/2397/538929/The-timing-of-speech-accompanying-gestures-with)
- [Temporal alignment of prosody and gesture in focus (J. Phonetics, 2026)](https://www.sciencedirect.com/science/article/pii/S0095447026000318)
- [Auditory cues to lexical stress and visual perception of gestural timing (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12331874/)
- [Co-speech gestures influence articulatory movements (Scientific Reports, 2024)](https://www.nature.com/articles/s41598-024-84097-6)
- [Streaming co-speech gesture generation via accelerated rolling diffusion (2025)](https://arxiv.org/html/2503.10488v1)
- [EMAGE (CVPR 2024)](https://www.researchgate.net/publication/384202544_EMAGE_Towards_Unified_Holistic_Co-Speech_Gesture_Generation_via_Expressive_Masked_Audio_Gesture_Modeling) · [EmoDiffGes (CGF 2025)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70261) · [GestureLSM (2025)](https://arxiv.org/pdf/2501.18898) · [MDT-A2G (2024)](https://arxiv.org/pdf/2408.03312)
- [Perturbation, prosody and speech-gesture coordination (Speech Communication)](https://www.sciencedirect.com/science/article/abs/pii/S0167639313000708)

---

## Implementation status (2026-08-25, same day)

Built as `zero/expr/` — tap, prosody, semantics, floor, scheduler, system —
exactly along the contact surface declared in §05 (two tap call sites in the
speech loop + build/stop lifecycle + one config block; `enabled: false` in
code, opted in by config with the floor still gated).

| Phase | Status |
|---|---|
| A — timing spine & instrumentation | **done** — tap + playout clock + scheduler; apexes land ±20 ms of the accent on synthetic speech (measured); `latency_ms` compensation in config, to be refined from black-box data on hardware |
| B — prosodic beats + living floor | **done** — DSP analyzer (contrast-gated: monotone speech correctly yields zero beats), beats through the bus; floor implemented but `floor.enabled: false` until the mic-bleed measurement |
| C — semantic hand vocabulary | **done** — count / aperture / palm-up / negation / sweep, closure-space, one per sentence max, 4 s rate floor |
| D — co-articulation polish | **partial** — anticipatory prep and barge-in turnaround (a stroke for words never spoken decays out instead of completing) are in; cross-sentence Kendon chaining is the open item |
| E — diffusion sidecar | not started (gated on a blind A/B the procedural build must lose) |

Bugs caught by the test pass, for the record: an inverted decay envelope
(every beat double-pumped), z-score inflation crowning four accents in a
pure tone (absolute-variation floor added), and a write-then-release race
that could abandon the hands mid-decay (park is now confirmed on the wire
before the claim is released). 679 tests green.

*Living Hands · plan of record → implemented · ZERO build, 2026-08-25.*
