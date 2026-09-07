# Mocap Capture Protocol — Rokoko suit → ZERO gesture training data

One "take" = one BVH export from Rokoko Studio + one WAV of the speaker's
voice, recorded at the same time. The converter
(`scripts/mocap_dataset.py`) turns a take into training shards identical
in format to the BEAT2 ones — the existing trainer, metrics and blind A/B
gate consume them unchanged.

## Equipment

- Smartsuit + Smartgloves, calibrated in Rokoko Studio (gloves REQUIRED —
  all 12 trained outputs are fingers and wrists).
- Any decent microphone near the speaker (lapel or phone is fine). It
  records continuously through the take.

## The take, start to finish

Every take begins with the same 15-second prologue. It is not optional —
the converter rejects takes without it.

| When       | Do                                                        |
|------------|-----------------------------------------------------------|
| 0–3 s      | Three sharp CLAPS, hands up where the suit sees them      |
| 4–6 s      | Both hands fully OPEN, held still                         |
| 7–9 s      | Both FISTS fully closed, held still                       |
| 10–14 s    | Roll both wrists end-to-end, a few full sweeps            |
| 15 s →     | CONTENT: talk naturally / sign                            |

Why: the claps are the sync marker (one physical event visible in the
audio and the motion at once — the converter aligns the clocks from it and
measures drift). The open/fist/roll "calibration sandwich" defines THIS
person's finger and wrist ranges, which is what maps their motion onto the
robot's 0–1 closure scale regardless of rig conventions or hand size.

- Takes: **5–10 minutes** each. Short takes keep sensor drift near zero.
  Recalibrate the suit in Rokoko Studio between takes.
- Content that trains well: explaining your work, telling a story,
  arguing a point, giving directions — natural, animated talking. Vary
  energy across takes. For sign data: a KSL signer running vocabulary,
  same prologue.
- Don't fold arms, sit on hands, or hold objects — occluded/parked hands
  teach stillness.

## Export

From Rokoko Studio: export the take as **BVH** (with finger animation
included). Keep the WAV as recorded. Names don't matter; keep the pair
together.

## Convert

    python scripts/mocap_dataset.py take.bvh take.wav --out data/gesture_shards

If the prologue timing differed, pass it, in MOTION seconds:

    --calib "open:5-7,fist:8-10,wrist:11-15"

The converter refuses loudly (missing claps, clock drift, dead
calibration, under 10 s of content). A rejected take costs minutes to
re-record; silent bad data poisons a training run — rejection is the
feature.

## Train

Own-data is for FINE-TUNING the incumbent model on a mix (BEAT2 for
variety, our takes weighted heavier per hour). Every result faces the
incumbent in the blind A/B (`scripts/gesture_ab.py --b-url`). The rule
does not change because the data is ours.
