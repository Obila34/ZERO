# Conversation rework — what changed, and why

A record of the work on ZERO's conversational layer: what was added, what was
removed, what was tried and reverted, and what is still open. Written so the
reasoning survives, not just the diff.

Branch: `claude/agent-human-feel-diagnostic-5al4hb`

---

## 1. The original bug: barge-in did nothing

Barge-in had been implemented but "nothing changed in practice". The cause was
three independent failures stacked, each of which degraded silently:

| Cause | Effect |
|---|---|
| `barge_in_on_speech: false` in config | The speech detector was never constructed |
| `models/vad/ten_vad.wasm` missing | TEN VAD fell back to webrtc |
| `models/turn/smart-turn-v3.2-cpu.onnx` missing | Turn detection fell back to a text heuristic |

Every fallback worked exactly as designed and logged one line at startup. Three
of them together meant the system ran as if none of the upgrade existed.

**Added:** an `engines live:` startup line naming what actually loaded, so a
silent degradation is a two-second log read instead of an investigation.

---

## 2. Barge-in detection

**Removed:** the echo floor as *max of the learning window*. One loud syllable
early in a reply wedged the gate above the user's voice for the rest of it.

**Added:**
- Percentile-based echo floor (80th) over a rolling window.
- **Envelope correlation** — the mic envelope is compared against what the
  speaker actually played, at any lag up to ~400 ms. Echo moves *with* the
  reply; a person does not. This is what makes barge-in survivable on a
  Bluetooth speaker, where sample-accurate AEC cannot work because the latency
  drifts.
- Pre-roll carry: speech starting before the reply's first audio cannot be
  echo, so it triggers almost immediately instead of waiting out the window.

**Later added:** a voiced-ratio gate. Loudness plus VAD was cleared by coughs
and chairs — the log showed `interruption '' -> noise` after an audible duck.
Now a confirmation window measures how much of the interruption is genuinely
speech-shaped before anything reacts.

---

## 3. What an interruption *means*

**Added:** `zero/audio/interrupt.py` — interruptions are classified instead of
all being treated as "stop":

- **Correction** ("no, I meant—", "wait", "stop") → cut the reply mid-word.
- **Backchannel** ("yeah", "mhm") → keep talking; agreeing with ZERO no longer
  silences it. Recorded as context for the next turn.
- **Anything else** → finish the current chunk, then answer, with the
  interrupting words already transcribed so the answer starts immediately.

Classification is lexical and instant (~3 µs). An LLM classifier was rejected:
~500 ms is unaffordable here, and the opening words of a real interruption
carry the intent almost unambiguously.

---

## 4. Listening while speaking

**Removed:** the deaf window. The mic used to be muted from the moment a turn
committed until playback began (~0.5–1.5 s) — exactly where people add "…oh,
and make it two".

**Added:**
- The duplex monitor runs from turn commit, not first audio.
- **Afterthoughts**: a remark that starts *and* ends before the reply's first
  audio is transcribed and merged into the pending turn, either before the LLM
  runs (free) or by restarting the reply on a warm prefix.
- Backchannels become context rather than being discarded.

---

## 5. Speech recognition: batch → streaming

Whisper (batch) was benchmarked against Kyutai (streaming), same clips:

| | Whisper | Kyutai (batch mode) |
|---|---|---|
| English | **472 ms** | 1173 ms |
| Swahili | **642 ms**, correct | 1157 ms, mangled |

That comparison was **unfair to Kyutai** — it was fed a finished clip, the one
job a streaming recogniser is worst at. With a live session the recognition
overlaps the speech itself:

- first words returned **1.0 s into** a 4.6 s utterance
- **187 ms** tail after speech ends (vs 472 ms for Whisper, which cannot start
  until you stop)

**Kept:** Whisper as the fallback — Kyutai's model is `stt-1b-en_fr`, English
and French only. Swahili was later dropped as a requirement.

### Bugs found and fixed in this path

1. **Marker deadlock.** The model only advances when fed audio, so the
   end-of-stream marker never returns unless you keep sending silence after it.
   The first implementation sent the marker then went quiet and timed out every
   time.
2. **Sender flooding.** Silence was emitted every 5 ms while each frame is
   80 ms of audio — ~16× realtime, which buried the speech and produced an
   *empty* transcript. Now paced at realtime while listening, sprinting only
   after the marker.
3. **Closed before read.** `_close_live_stt()` ran ~40 lines before
   `finalize()`. Result: a 2 s blocking join on every reply, `settled in 0 ms`
   on every turn (nothing happening, not speed), and the last word of every
   sentence silently dropped.
4. **Double-fed VAD.** The bandwidth gate called the endpointer's *stateful*
   TEN VAD on frames that `capture()` also consumed, desynchronising its hop
   buffer and damaging both utterance detection and barge-in. Replaced with a
   stateless level check.

---

## 6. Speech synthesis

Measured, same line, 3 runs:

| | Orpheus | Kyutai |
|---|---|---|
| First audio | 1699 ms | **215 ms** |
| Rate | 0.8× realtime | 2.9× realtime |

**Orpheus was the real latency bottleneck all along** — 1.7 s before the first
sound of every reply, which made sub-second unreachable no matter what the
brain did. Cut over to Kyutai; Orpheus became the failover.

**Trade-off accepted:** Kyutai cannot perform the `[laughs]` / `[sighs]` cue
tags the persona uses. They are stripped, never spoken as literal words.

---

## 7. The brain

Cut over from Ollama (Gemma 8B) to **vLLM serving Gemma 4 12B QAT** on
zerolabs0. Measured **39–66 ms to first token**, ~80 tok/s.

**Added:** `zero/llm/openai_engine.py`, an OpenAI-compatible client with the
same streaming / warmup / prefill surface as the Ollama engine, so the cutover
was a config flip.

**Fixed:** the cutover silently broke semantic memory. `build_memory` passed
`host=llm.host`, so the embedder followed the brain to vLLM and began asking it
for `/api/embeddings` — an endpoint vLLM does not serve. Every session degraded
to hash matching. The embedder now has its own explicit host.

---

## 8. Predicting the end of a turn

**Added:**
- **Predictive endpointing** — when Smart Turn is confident at the ~180 ms
  pause mark, the turn commits there instead of waiting out the full silence
  window.
- **Speculative reply** (`zero/speculate.py`) — generation starts before the
  person finishes, when the turn model is confident and the partial transcript
  already reads like a complete request.

The gate is the load-bearing part: a speculative reply is spoken **only** if
the finished sentence matches word-for-word what it was bet on. A prefix match
is explicitly rejected, because *"book me a flight"* → *"book me a flight,
actually no, cancel that"* inverts the meaning. A wrong bet costs one wasted
GPU stream and nothing else.

---

## 9. Reading the room

**Added:** `zero/audio/room.py`. Every loudness threshold was a constant tuned
in one quiet room. The ambient floor is now measured continuously from idle mic
frames and drives three things:

| Room | Voice | Interrupt gate |
|---|---|---|
| Silent | ×0.83 | 250 |
| Quiet | ×1.00 | 341 |
| Office | ×1.22 | 845 |
| Hall | ×1.60 | 1745 |

Capped so a loud room cannot drive the output into clipping.

---

## 10. Volume — three attempts

This one went back and forth, and the history is the point.

1. **Duck to 0.35 on any detected sound.** Fired on coughs and chairs; the
   reply audibly stuttered. Wrong.
2. **Remove ducking entirely.** An overcorrection — it conflated two different
   things.
3. **Where it landed:** volume as *communication*, not reflex.
   - **Removed:** the reflex dip (now `barge_in_duck: 1.0`). A queued
     interruption means "I will answer after this thought" — the sentence being
     finished should sound *normal*, not trail off.
   - **Added:** spoken volume control that persists — "talk quietly" (0.72),
     "whisper" (0.5), "speak up" (1.35), "I can't hear you", "normal volume".
     Length-capped, so a story that merely *mentions* whispering is not an
     instruction.
   - Asked-for level **multiplies** with the room's Lombard gain: quiet still
     means quiet in a loud hall, without becoming inaudible.

**Fixed:** `unduck()` slammed gain to a bare 1.0, discarding both the
asked-for level and the room gain.

---

## 11. Exhibition hardening

- **Degeneracy guard.** A live reply came out as `thought` ×11, spoken aloud.
  Repetition is now detected and the stream cut before any sound. The guard was
  initially in the engine, which missed it — the repetition arrived via the
  **tool router**, a separate path. Moved to the final stream, so everything
  reaching the voice passes it.
- **Never go silent.** Any failed turn speaks a recovery line. These are
  synthesised at *startup* and held as waveforms, so they still play when the
  TTS server is unreachable. A startup warning fires if none could be cached.
- **Network retries.** LLM connect retried 3× with backoff, connect and read
  timeouts separated; Kyutai TTS retries once, but never after audio has
  started (reconnecting mid-sentence would repeat words aloud). All verified
  against dead endpoints and blackhole IPs — every path returns cleanly, no
  exception reaches the caller.
- **GPU services:** `Restart=on-failure` → `Restart=always`, and systemd's
  start limit removed (default gives up permanently after 5 restarts in 10 s —
  the failure that would have ended an exhibition day). Verified by `kill -9`:
  back **serving in ~14 s**, unattended.
- **Bandwidth.** The STT socket sent float32 PCM continuously — measured
  **962 kbps, 7.2 MB per minute merely waiting** for someone to speak, to a
  host reached over a relay. The wire now stays silent until real speech.

---

## 12. Latency

Found by adding stage timers rather than guessing — after one wrong guess
earlier in this work.

| Stage | Was | Now |
|---|---|---|
| `arm-bargein` | **526 ms** | 2–8 ms |
| `identity` | 500–1000 ms | 150 ms (time-boxed) |
| `transcript` | n/a (batch) | 150–300 ms |
| `llm+tts` | — | now the largest block |

`arm-bargein` was `wake.reset()`, measured at **521 ms** — openWakeWord
rebuilding its state, in front of every reply, for no benefit (the wake word is
not needed to continue a conversation). Moved onto the monitor thread.

Identity (voice embedding, diarization, registry) was ~0.5–1.0 s on the reply
path — a third of the budget spent deciding *who* was speaking, which the
answer never needed. Now time-boxed at 150 ms; past that the turn replies with
the identity already known and the result lands for the next turn.

**Clause splitting** was the last change: playback cannot start until the first
chunk is complete, so a 40-word opening sentence delayed both the first sound
*and* the handover on a queued barge-in (measured at **11 seconds**). Long
sentences are now split at commas and dashes as well as full stops.

---

## 13. Guests

**Removed:** guest clustering (`identity.guests.enabled: false`).
`match_threshold` (0.55) was stricter than a real person's own turn-to-turn
variation — Sam scored **0.46–0.60 against his own enrolled profile** — so a
new guest was minted almost every turn (`guest-69`…`guest-74` in one short
session). Over a day of visitors that churns the database for no benefit.

**Kept:** enrolled-person recognition, which is a separate path. Sam is still
identified by name.

---

## Still open

- **Turns that produce no transcript.** A loud 3.3 s utterance (rms 1825) still
  returned nothing. Now logged with duration and level rather than vanishing,
  but the cause is not yet found.
- **Both model hosts are reached through a Tailscale DERP relay**, not directly
  — zerolabs2 shows 81 ms RTT and 33 ms jitter *on the local network*. This is
  routing, not code, and getting direct connections would help more than any
  remaining optimisation.
- **`llm+tts` is now the largest block.** Clause splitting should cut it; needs
  a live run to confirm.
- **Crowd behaviour is untested.** `barge_in_voiced_ratio` (0.55) and the room
  anchors (120/900 rms) are estimates, not measurements.
- **The Pi does not auto-start ZERO** — deliberate, by request.
