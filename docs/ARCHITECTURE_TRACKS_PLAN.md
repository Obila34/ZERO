# Architecture Tracks — Build Plan (2026-09-09)

Plan of record for the three tracks the operator green-lit after the
2026-09-09 architecture audit. Track 2 (Swahili STT on zerolabs0) is
HELD by operator decision — zerolabs0 stays empty.

Standing rules, same as every plan here: each piece ships dark or
behind its own flag; nothing alters existing behavior until verified;
anything that moves metal gets watched by the operator first; the full
test suite gates every merge.

---

## Track 1 — Truth: ZERO must know its own body and health

### T1.1 Black-box repair (the flight recorder)
The joint log (zero_joints.sqlite) has recorded nothing for weeks — no
tables at all. Every calibration hunt this month would have been
minutes instead of hours with it working.
- Diagnose: is JointAngleLog constructed in the live service at all,
  and if so why the schema never lands (suspect: the Aug-27 probe run
  recreated an empty DB and init is once-only / error-swallowed).
- Fix + a test that builds the bus the way the SERVICE builds it and
  asserts acknowledged posts land as rows.
- Size cap + retention so it can run for months.
- Deploy; verify rows appear from a live conversation.
Owner: model. Robot needed: only for final verification.

### T1.2 The monitor learns to speak
The fleet page shows failures; nobody is told. Add state-change
actions to fleet_monitor:
- Every UP->DOWN and DOWN->UP appends to the log (exists) AND fires a
  configurable alert command. Default: push to an ntfy.sh topic the
  operator subscribes to on their phone (free, no account, one URL).
- New self-guard row: "black box freshness" (age of last joint row) —
  the recorder can never die silently again.
- Later (when Pi is up): a DOWN alert can also post to ZERO's event
  bus so the robot can literally say "my vision server just died."
Owner: model. Operator: install ntfy app, subscribe to the topic.

### T1.3 True degrees for every stepper (per-joint scale ritual)
One global steps-per-degree constant lied for all nine motors; the
gateway now supports a per-joint table (steps_scale.json) but it is
EMPTY. Guided measurement ritual, one joint at a time, robot on,
operator watching:
- scripts/step_scale_ritual.py commands a known sweep, operator
  estimates the physical degrees moved (rough is fine; a phone
  clinometer app makes it exact), script computes and writes the
  joint's true scale AND rescales that joint's stored zero-offset so
  the calibrated rest pose is preserved.
- Order: head_tilt first (gaze amplitude depends on it), then the six
  arm joints. Stance/sign/gesture amplitudes re-checked by eye after —
  they were tuned in the old fake degrees and will need one pass.
Owner: operator + model together, ~30 min with the robot on.

### T1.4 Standing rule (documented, no code)
Encoderless steppers forget their position on every controller
reboot. Rule: the Arm Pi only reboots with the robot at rest; if it
ever reboots otherwise, run the re-zero ritual before starting ZERO.
Hardware ask to Kamau (open): one limit switch per arm joint.

---

## Track 3 — Close the sign-language circle

### S3.1 Fingerspell READING goes live (photo test first)
Built and dark. Gate: measured accuracy on real hands.
- Operator drops labeled photos (filename = the letter spelled, e.g.
  a.jpg, b_2.jpg) into ~/sign_test_photos on zerolabs1.
- scripts/spell_eval.py runs each through the live sidecar, reports
  per-letter accuracy and the confidence-threshold sweep, recommends
  sign_sense.min_confidence.
- If accuracy is usable: enable sign_sense on the Pi, live test —
  spell a word at the robot, hear it answer.
Owner: operator (photos, 10 min) then model.

### S3.2 Real translation: the gloss experiment
Today ZERO signs content words in ENGLISH order (sign-supported
speech). Real sign languages have their own grammar. The brain is a
language model — translation is its native skill.
- Per spoken sentence, an ASYNC side-call to Gemma: "render this
  sentence as ASL gloss order, only words from this vocabulary" (the
  dictionary's gloss list, cached in the prompt). Arrives while TTS
  speaks the first words; sign-along plays the gloss sequence instead
  of its picked words; on timeout/miss it falls back to today's picker
  — so the feature can only improve, never stall signing.
- Ships dark (sign.along.gloss.enabled). Judged by eye, ideally with a
  signer present; promoted the way everything here is promoted.
Owner: model. No robot needed until judging.

### S3.3 Signer curation afternoon
2,298 machine-retargeted signs; only a signer can say which read true.
- scripts/sign_review.py: plays the N most-used signs (frequency from
  sign-along logs once the black box records them) one at a time;
  signer marks good / replace / never. "Never" blocks the sign
  (fingerspell instead); "replace" queues a hand-authored lexicon
  entry, which automatically shadows the dictionary.
- Output: data/sign_curation.yaml consumed by the engine.
Owner: build = model, session = operator + signer, ~1 afternoon.

---

## Track 4 — main.py surgery (background, in slices)

3,201 lines, 92 methods, one class. Extraction slices, each landed
alone behind the full 714-test suite, never during a demo week:

- M4.1 Announcements/recovery: _drain_events, recovery lines, filler
  synthesis -> zero/conversation/announce.py. Small, self-contained.
- M4.2 Turn pipeline, one stage per slice, each a module with a narrow
  context object: (a) transcription (live STT + finalize + rescue +
  afterthought merge), (b) identity/privacy gate, (c) reply+tools,
  (d) speak/playout. Zero class shrinks to lifecycle + orchestration.
- M4.3 Subsystem boot registry: the dozen copy-pasted try/except
  builder blocks -> one declarative list with a uniform
  build-or-warn-and-continue rule.
Rule of engagement: at most one slice in flight; a slice that fights
back gets reverted, not heroically finished.

---

## Order of work

1. T1.1 black box (now — no robot needed)
2. T1.2 monitor alerting (now — no robot needed)
3. S3.2 gloss experiment built dark (now — no robot needed)
4. M4.1 first surgery slice (background)
5. T1.3 scale ritual + S3.1 photo test + rest-of-migration restart —
   the next session the robot is powered on
6. S3.3 when the signer is booked
