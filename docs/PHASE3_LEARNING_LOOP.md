# Phase 3 — The learning loop (complementary learning systems)

Date: 2026-07-24. Status: **implemented and validated**; LoRA training and
INT8-on-Pi are flagged pending hardware (see Flags).

The frame: a slow, stable cortex (the pretrained LLM — never touched live)
plus fast, plastic peripheral stores (episodes, memory, vocab, embeddings),
with sleep consolidating between them. Live weights are never gradient-updated
— catastrophic forgetting is designed out, not mitigated.

## The experience spine

`zero/learning/episodes.py` — ONE schema for everything ZERO lives through:
`(v, ts, kind, person_id, payload JSON, reward, surprise, consolidated_at)`,
kinds `turn | scene | proactive | action` (action reserved for limbs — same
rows, same consolidation, no schema change). Versioning: `PRAGMA user_version`
+ an ordered `_MIGRATIONS` list; each row records the schema version it was
written under. Adding a field = appending a migration.

## Reward (dopamine) — zero/learning/reward.py

Per turn, in [-1, 1]: affect valence×confidence while speaking (0.4 weight)
+ explicit next-utterance verdicts ("no, that's wrong" → −1 retro-tag; narrow
regexes so "this weather is awful" is NOT a verdict on ZERO) + barge-in
(−0.5) + engagement (+0.1). Wired in `main.py` behind the SAME privacy gate
as the corpus — an unstorable turn tags nothing.

Consumers:
* **Policy bandit** — proactive outcomes (reply sentiment; silence = −0.2)
  feed `InteractionPolicy.record_outcome`; per-kind EMA scales cooldowns
  ×0.5 (landing) to ×3.0 (falling flat).
* **Memory salience** — consolidation maps a session's mean reward into
  episodic importance (4 + 3·r̄, the lever memory's activation formula
  already multiplies by).
* **Training data** — corpus records inherit session reward: ≥ +0.3 doubled,
  ≤ −0.4 dropped.

## Surprise (prediction error) — zero/world/surprise.py

The world state predicts its own statistics: Laplace-smoothed (label, kind)
event counts, persisted (`data/world_surprise.json`) so "normal" accumulates
across runs. surprise = −log2 p in bits. `SurpriseGate` (thread on
`world.wait_for_change`, ~0.7 µs/event) routes: ≥ 4 bits → a `scene` episode
(memory-bound); ≥ 6 bits → `narrator.poke()` (wake the expensive model for
the unexpected; it sleeps through routine). Bug caught by tests: naive
smoothing gave the first-ever event 0 bits — fixed with reserved unseen mass.

## Sleep — scripts/consolidate.py (+ zero-consolidate.{service,timer}, 03:30)

flock single-instance; per-step journal (`data/consolidation/journal.json`);
every step idempotent; exit code = failed steps. Steps: **distill** (episodes
→ reward/surprise-weighted memories, then marked consolidated), **decay**
(existing `memory.consolidate()`), **corpus** (reward-weighted full rebuild
of `data/train/chat.jsonl`), **vocab** (re-export Pi ONNX iff the LIVE
`vocab.runtime.txt` outgrew it — words added via POST /perceive/vocab reach
the Pi overnight), **objects** (VACUUM; re-embedding N/A — no raw crops
stored, flagged), **lora** (runs `learning.training.cmd` if set; default
null → SKIPPED, VRAM).

## Pruning + budgets

* `scripts/prune_models.py` — INT8 dynamic quantization + magnitude-sparsity
  report, with measured accuracy drift. Real numbers on
  `yolov8s-worldv2-480.onnx`: **51.6 → 14.4 MB (−72%)**, but **slower on x86**
  (88.8 vs 55.2 ms) and 1/2 image label agreement → NOT shipped by default;
  re-measure on the Pi where int8 typically wins. Magnitude pruning reported,
  not applied (needs recovery fine-tune).
* `zero/world/budget.py` — enforced ceilings: `DutyBudget` (Tier 1 detector
  ≤ 60% duty over 10 s, even under constant motion) and `RateBudget` (Tier 2
  ≤ 20 inferences/min, pokes included). Wired in Eyes and Narrator via
  config `world.budgets.*`.

## Measured vs budgets

| Path | Measured | Budget |
|---|---|---|
| reward tag + episode write (hot path) | 0.856 ms/turn | ≤ 1 ms tag + ≤ 5 ms write ✓ |
| surprise observe | 0.7 µs/event | ≤ 1 ms ✓ |
| consolidation dry-run (all 6 steps, real repo data) | < 0.2 s | minutes ✓ |

Tests: 20 new (`tests/test_learning_loop.py`) — full suite **415/415**.

## Flags

1. **LoRA nightly training**: pipeline delivers the reward-weighted dataset
   and the hook (`learning.training.cmd`); actually training needs GPU
   headroom the shared 16 GB card doesn't have. Not faked — SKIPPED with
   reason in the journal.
2. **INT8 on the Pi**: 72% smaller but slower on x86; ship only after an
   on-Pi latency/accuracy run (`scripts/prune_models.py quantize ...`).
3. **Learned-object re-embedding** offline is impossible without stored raw
   crops (deliberate privacy/simplicity trade); noted in the objects step.
4. Engagement signal is coarse (continuation +0.1); a follow-up-length or
   sentiment-trend signal is a Phase 3.5 refinement once real reward data
   accumulates.
