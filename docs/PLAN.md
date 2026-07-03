# ZERO — Roadmap to a Living Companion

The plan to take ZERO from a working voice pipeline to a **present, learning,
proactive companion**. Every capability below is committed to implementation.
This document is the source of truth for *what* we're building, *why*, *what
others have done*, and *exactly where it plugs into this codebase*.

Read alongside [OPERATIONS.md](OPERATIONS.md) (how the running system works) and
[GPU_OFFLOAD_PLAN.md](GPU_OFFLOAD_PLAN.md) (the Pi↔GPU split).

---

## 0. Vision & principles

ZERO today is a request→response voice box: wake word → question → answer. The
goal is a companion that **recognises the people it knows, acts in the world,
speaks up on its own, remembers like a person, and learns continuously —
including on its own.**

Design principles carried from the existing codebase:

1. **Everything behind an interface, selected in `config.yaml`.** New faculties
   are new packages under `zero/` wired through `zero/factory.py`; `main.py`
   changes as little as possible.
2. **Graceful degradation.** Every remote/heavy capability has a local fallback
   and fails soft — a missing camera drops ZERO to voice-only, never a crash.
3. **The embed→cosine→threshold pattern is reused everywhere.** Voice ID already
   does it (`zero/voiceid/speaker.py`); face ID, object learning, and semantic
   recall are the *same shape*. Build it once, reuse it four times.
4. **Presence over chatter.** Proactivity and curiosity are gated by policy —
   the hard part is *when to stay quiet*, not what to say.
5. **Privacy is a module, not a footnote.** Always-listening + face data +
   memory of people is exactly the system the research warns about. Consent,
   on-device retention, and a visible "listening" signal are first-class.

### The faculty model

Everything is organised into six faculties:

| Faculty | Meaning | Modules |
|---|---|---|
| **Perception** | senses | emotion recognition, diarization, sound localization |
| **Cognition** | mind | human-like memory, spatial/object memory |
| **Action** | hands | agentic tool use |
| **Learning** | growth | object-name learning, curiosity, preference learning |
| **Expression** | body | face/gaze, self-state narration |
| **Social & Trust** | relating | identity, proactivity, privacy/consent, multilingual |

---

## 1. Phased roadmap

Dependency-ordered. Each phase is independently demoable and independently
valuable — we do not build all of it at once.

| Phase | Capability | Depends on | Why here |
|---|---|---|---|
| **0** | Always-on (systemd) | — | Stop fighting the servers while building. Mechanical. |
| **1** | **Identity** (face + voice fusion) | 0 | The keystone — proactivity, per-person memory, and greeting all need "who is this?" |
| **2** | **Agentic tool use** | 0 | Biggest capability jump: talker → assistant that *does* things. |
| **3** | **Human-like memory** (layered + retrieval + consolidation) | 1 | Per-person, relevance-retrieved, self-consolidating. |
| **4** | **Learning** (object names + curiosity + preferences) | 1, 3 | Grows past COCO-80; learns on its own during idle. |
| **5** | **Proactive / ambient** | 1, 2, 3 | Speaks up on its own — now that it knows people and can act. |
| **6** | **Perception polish** (emotion, diarization) | 1 | Reads tone and tracks who's speaking. |
| **7** | **Expression + Privacy + Multilingual** | all | The "companion" layer + trust. |
| **H** | **Hardware-gated** (sound localization, embodiment, navigation) | new hardware | Only if/when the physical form supports it. |

---

## 2. Phase 0 — Always-on (systemd)

**What:** install the `scripts/systemd/` units so both boxes self-start and
self-heal (`Restart=always`). Also fixes the Orpheus `llama_decode` crash
(a crashed server auto-restarts clean, state-free).

**How:** the units already ship with real values (GPU `obilasam3` /
`/home/obilasam3/ZERO`; Pi `head` / `/home/head/Mzee/offline_v5`), then:

```bash
# GPU (zerolabs1)
sudo cp scripts/systemd/zero-{whisper,orpheus,vision}.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now zero-whisper zero-orpheus zero-vision
# Pi (head)
sudo cp scripts/systemd/zero-{tunnel,}.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now zero-tunnel zero
```

**Effort:** ~1 hour. **Files:** `scripts/systemd/*` only.

---

## 3. Phase 1 — Identity (face + voice fusion)

**What:** replace the single-owner voice gate with a multi-person registry that
answers *"who is this?"* from **face and voice together**, and greets people by
name.

**Why:** the keystone. Proactive greetings, per-person memory, and "only my
family" gating all require identity. Two weak signals (a turned-away face + a
noisy voice) confirm each other; either alone still works.

**Prior art:** face embeddings via **ArcFace / InsightFace** (ONNX, offline) —
the exact embed→cosine→threshold math already in `zero/voiceid/speaker.py`.

**Design — new package `zero/identity/`:**
- `FaceRecognizer` — ArcFace ONNX. Runs on the Pi CPU via onnxruntime (like
  YOLO) or on the GPU vision server (`server/vision/`). Input: the keyframes
  `Eyes` already grabs. Output: a 512-d face embedding.
- `PersonRegistry` — SQLite table of enrolled people, each with a *voice*
  embedding **and** a *face* embedding (multiple samples averaged). Replaces
  the single `voiceprint.npy`.
- `IdentityFuser` — `fused = w_face·face_cos + w_voice·voice_cos`; returns a
  `person_id` or `stranger` above/below a threshold.
- **Conversational enrolment:** "I'm David" → capture face + voice → store.
  No more CLI-only `test_voiceid.py enroll`.

**Codebase integration:**
- Generalise `build_voiceid` in `zero/factory.py` from binary owner-gate → the
  identity service.
- `main.py._converse()`: resolve identity per turn; pass `person_id` to memory
  and proactivity.
- Reuse `Eyes.visual_context()` frames for the face crop.

**Config:** `identity.enabled`, `identity.face_model`, fusion weights,
thresholds. **Effort:** ~1–2 weeks.

---

## 4. Phase 2 — Agentic tool use (function calling)

**What:** let ZERO *act* — timers, reminders, live data, smart-home control,
queries — not just talk.

**Why:** the largest single capability jump. A companion that can't do anything
stays a toy. This is also the mechanism proactivity uses to *act*.

**Prior art:** the [Home Assistant + local-LLM](https://www.home-assistant.io/blog/2025/09/11/ai-in-home-assistant/)
stack does exactly this, offline, with the same streaming-TTS trick ZERO uses.
The proven pattern is a **two-tier router**: fast pattern-match for known
commands, LLM function-calling for the rest ([acon96/home-llm](https://github.com/acon96/home-llm),
[InsiderLLM guide](https://insiderllm.com/guides/home-assistant-local-llm-guide/)).
HA's assistant can now also *initiate* action-suggestions — that feeds Phase 5.

**⚠️ Stack risk:** function-calling needs a **tool-calling-capable model**.
Verify `gemma4:latest` supports structured tool calls; if not, either swap the
LLM or add a JSON-router prompt layer that parses intents ourselves.

**Design — new package `zero/tools/`:**
- `Tool` base interface: `name`, `description`, `parameters`, `run()`.
- `ToolRegistry` + a router in the LLM path (`zero/llm/`): tier-1 regex/intent
  match → tier-2 LLM function-call → tier-3 plain chat.
- Starter tools: `timer`, `reminder` (ties to memory), `time/date`,
  `remember`/`recall`, `home_assistant` (optional, if HA is present).
- Safety: an allow-list of tools; nothing that moves money / sends messages
  without explicit confirmation.

**Codebase integration:** wrap `OllamaLLM.stream` so tool-call tokens are
intercepted, executed, and the result fed back before the spoken reply. Keep
streaming so latency stays low. **Effort:** ~2 weeks.

---

## 5. Phase 3 — Human-like memory

**What:** evolve `zero/memory/sqlite_memory.py` from a flat key/value store that
**dumps everything into the prompt** into a layered, relevance-retrieved,
self-consolidating memory.

**Why:** the current `as_block()` injects *all* facts + recent episodes every
turn. Humans recall what's *relevant now*. This is the difference between a
notepad and a memory.

**Prior art — the field has converged on Stanford's Generative Agents**
(observation → reflection → retrieval, scored by **importance × recency ×
relevance**), extended by 2025 work:
- Layered working / episodic / semantic / procedural memory
  ([Multi-Layer Framework](https://arxiv.org/html/2603.29194v1),
  [Human-Inspired Architecture](https://arxiv.org/pdf/2605.08538))
- **ACT-R activation** — recall strength = frequency + recency + decay +
  relevance ([ACT-R-inspired](https://dl.acm.org/doi/10.1145/3765766.3765803))
- **Sleep-phase consolidation** + interference-based forgetting
  ([dynamic recall + consolidation](https://arxiv.org/html/2404.00573))
- Consensus: **episodic memory is the missing piece**, and episodic + semantic
  beats either alone ([position paper](https://arxiv.org/pdf/2502.06975)).

**Design — four upgrades, in value order:**
1. **Relevance retrieval (biggest win):** store an embedding per memory; each
   turn retrieve top-K by `relevance × recency_decay × importance`, instead of
   dumping all. Implementation: `sqlite-vec`, or a `/embed` endpoint on the GPU
   box we already run.
2. **Importance scoring at write:** `_extract_facts` already runs an LLM pass —
   have it also rate salience 1–10. Cheap, big retrieval-quality gain.
3. **Reflection / "sleep" consolidation:** extend the existing background thread
   (`_save_memories`) with a periodic pass that synthesises higher-level
   insights ("David asks about weather every morning → David has a routine").
4. **Layered + per-person:** key memories to `person_id` (needs Phase 1);
   separate working (this convo) / episodic (events) / semantic (world facts &
   learned concepts) / procedural (habits). Add gentle **forgetting** so
   unreinforced memories decay and it doesn't calcify.

**Codebase integration:** keep the `SqliteMemory` public API (`remember`,
`as_block`, `add_episode`) but back it with the new tables + retrieval; `main.py`
barely changes. **Effort:** ~2–3 weeks.

---

## 6. Phase 4 — Learning

The faculty that makes ZERO grow. Three distinct capabilities.

### 6a. Learning new object names (few-shot, no retraining)

**What:** ZERO's YOLO detector is frozen at COCO-80 (`zero/vision/detector.py`).
Teach it new objects by *example*: "that's a French press" → bind the name to a
visual embedding → recognise it next time.

**Why:** grows past 80 classes without a GPU training run — essential for a
robot that lives in *your* specific room with *your* specific things.

**Design — the embed→cosine pattern, a third time:**
- Embed the detected region with a CLIP-style encoder (ONNX) or the multimodal
  LLM's vision tower.
- `LearnedObjects` table: `name`, visual embedding(s), `person_id` who taught
  it, timestamp.
- At detection, match unknown regions against learned embeddings by cosine;
  above threshold → speak the learned name.
- Complements the existing **YOLO-World open-vocab** path
  (`vision.detect.names_path`) — YOLO-World handles a *known vocabulary*,
  this handles *novel, user-taught* objects.

**Integration:** new `zero/vision/learned.py`; `Eyes` consults it after YOLO;
the "teach" intent is a Phase-2 tool ("remember this is a …"). **Effort:** ~1–2 weeks.

### 6b. Curiosity / self-directed learning (learn when alone)

**What:** when no one is conversing, ZERO doesn't idle — it **consolidates and
forms questions**, then asks them opportunistically when a known person returns.

**Why (and the nuance):** a robot talking to an empty room is unsettling and
wasteful. Curiosity that *respects presence* is the valuable version:
- **Alone → learn silently:** run memory consolidation (6/Phase 3), review
  objects it couldn't name, detect knowledge gaps, and **queue questions**.
- **Known person returns → ask opportunistically:** *"Earlier I saw something
  on your desk I couldn't identify — what was it?"*

This is intrinsic-motivation / active-learning: the "sleep-phase consolidation"
from the memory research doing double duty as a learning loop.

**Design — new `zero/proactive/curiosity.py`:**
- A `QuestionQueue` of open items (unnamed objects, unresolved references,
  follow-ups from past chats), each with a priority and a "who to ask."
- A consolidation tick during idle that populates it.
- Emits proactive-interaction *candidates* consumed by Phase 5's policy — it
  never speaks directly; the policy decides *when*.

**Effort:** ~1–2 weeks (leans on Phases 3 & 5).

### 6c. Continual preference learning

**What:** update *behaviour* from correction, not just store facts. "Stop
calling me that", "you talk too fast" → persistent behavioural adjustment.

**Design:** a `preferences` layer in memory (procedural) that injects active
preferences into the system prompt and, where possible, tunes engine params
(e.g. Orpheus/Piper `length_scale` for speaking rate). **Effort:** ~1 week.

---

## 7. Phase 5 — Proactive / ambient

**What:** let ZERO leave IDLE and speak **without a wake word** — greet a
recognised person, surface a queued curiosity question, offer a timely action.

**Why:** the shift from reactive to present. Today `main.py` only leaves IDLE on
a wake word.

**Prior art:** ambient agents that "continuously read context, spot a trigger,
initiate within guardrails" ([ambient agents](https://zbrain.ai/ambient-agents/),
[proactive AI](https://vanishlabs.ai/news/proactive-ai)); Google's
[Sensible Agent](https://research.google/blog/sensible-agent-a-framework-for-unobtrusive-interaction-with-proactive-ar-agents/)
for *unobtrusive* timing. The universal warning: **privacy in always-listening
social settings** ([CONCORD](https://arxiv.org/pdf/2604.13348)) → see Phase 7.

**Design — new package `zero/proactive/`:**
- `TriggerSource` — a background watcher over the `Eyes` stream + timers +
  curiosity queue, emitting events: *known person entered frame*, *someone
  waved*, *person present + long silence*, *reminder due*.
- `InteractionPolicy` — the hard part. Gate every proactive utterance on:
  known person (never a stranger), per-person cooldown, "not already greeted
  this session", not mid-conversation, time-of-day appropriateness.
- On fire → inject into `main.py`'s loop to enter SPEAKING without a wake word.

**Codebase integration:** a new event path parallel to `_wait_for_wake()`; the
policy is the safety valve so ZERO isn't exhausting. **Effort:** ~2–3 weeks.

---

## 8. Phase 6 — Perception polish

### 8a. Multimodal emotion recognition
**What:** face expression + voice tone + text sentiment → an affect estimate
that steers the reply *and* Orpheus's delivery. **Why:** the defining feature of
2025 companion robots ([survey](https://www.mdpi.com/2078-2489/16/11/948),
[multimodal emotion survey](https://arxiv.org/pdf/2312.05735)). **Design:** a
small facial-expression ONNX model on the keyframes + prosody features from the
audio (reuse the fbank front-end in `speaker.py`) + LLM sentiment; fuse into an
`affect` note attached like the vision note in `_attach_vision`.

### 8b. Speaker diarization
**What:** in a group, track *who said what* so ZERO answers the right person and
names them. **Why:** natural extension of identity to multi-person rooms.
**Design:** segment utterances by voice embedding (already computed for Phase 1)
+ timing; tag each turn with a `person_id`.

---

## 9. Phase 7 — Expression, Privacy, Multilingual

### 9a. Expression (face / gaze) — ⚠️ hardware-gated
An LED/screen face with lip-sync + eye contact. Companion-robot research is
unanimous that *visual/physical expression* creates emotional connection
([reshaping interaction](https://roboticsandautomationnews.com/2025/10/31/social-and-companion-robots-more-than-just-machines/96093/)).
**Depends on the physical form** — a display and/or actuators.

### 9b. Self-state narration
ZERO already *has* degraded-mode signals (Piper vs Orpheus). Let it *narrate*
them: *"I'm on my backup voice right now."* Cheap, and it makes ZERO honest.
**Design:** expose the fallback state from `zero/tts/fallback.py` &
`zero/stt/fallback.py` to the persona context.

### 9c. Privacy & consent (first-class)
Once always-listening + face data + memory-of-people combine, this is a module,
not a footnote ([CONCORD](https://arxiv.org/pdf/2604.13348)):
- Visible "I'm listening" indicator (LED/screen).
- Bystander handling — don't transcribe/act on unknown voices (ties to
  identity + `voiceid` gating).
- Strict on-device retention; "forget that" / "forget me" commands
  (`SqliteMemory.forget_all` exists — extend to per-person).
- Consent for enrolment.

### 9d. Multilingual / code-switching
Whisper + Orpheus both support it. English⇄Swahili code-switching is directly
useful (deployment is in Nairobi) and is mostly a config/prompt change plus
per-language voice selection.

### 9e. Wake-word upgrade
Train a real **"Hey Zero"** openWakeWord model to replace `hey_jarvis`
(`wake.model` in config).

---

## 10. Hardware-gated (Phase H)

Only if/when ZERO's physical form supports it. **Open question to resolve first:
what is ZERO physically — a stationary Pi + camera + speaker, or does it have a
screen, mic array, moving parts, or a mobile base?** That answer decides these:

- **Sound-source localization** — turn toward the speaker. Needs a **mic array**
  (the single Brio can't do direction-of-arrival).
- **Spatial memory / semantic mapping** — "where are my keys?" A lightweight
  object-location log over the existing YOLO + depth gets ~80% of this *without*
  full SLAM ([HoloAgent 3D spatial memory](https://arxiv.org/pdf/2606.23565),
  [open-vocab semantic mapping / embodied AI list](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List)).
- **Navigation / mobile base** — full SLAM only if there are wheels.

---

## 11. New packages summary

```
zero/
  identity/        FaceRecognizer, PersonRegistry, IdentityFuser   (Phase 1)
  tools/           Tool, ToolRegistry, router + starter tools      (Phase 2)
  memory/          (evolve) layered store, embeddings, retrieval,  (Phase 3)
                   consolidation, forgetting, per-person, prefs
  vision/learned.py  few-shot learned-object recognition          (Phase 4a)
  proactive/       TriggerSource, InteractionPolicy, curiosity     (Phase 4b, 5)
  perception/      emotion (affect), diarization                   (Phase 6)
  privacy/         consent, listening indicator, bystander gate    (Phase 7c)
```

Everything is wired through `zero/factory.py` and selected in `config.yaml`, so
`main.py` stays the thin orchestrator it is today.

---

## 12. Sources

Memory: [ACT-R-inspired](https://dl.acm.org/doi/10.1145/3765766.3765803) ·
[dynamic recall + consolidation](https://arxiv.org/html/2404.00573) ·
[episodic-semantic for long-horizon agents](https://arxiv.org/pdf/2605.17625) ·
[episodic memory is the missing piece](https://arxiv.org/pdf/2502.06975) ·
[human-inspired architecture](https://arxiv.org/pdf/2605.08538) ·
[multi-layer memory](https://arxiv.org/html/2603.29194v1) ·
[memory systems in AI agents](https://www.analyticsvidhya.com/blog/2026/04/memory-systems-in-ai-agents/)

Proactive/ambient: [ambient agents](https://zbrain.ai/ambient-agents/) ·
[proactive AI](https://vanishlabs.ai/news/proactive-ai) ·
[Sensible Agent (Google)](https://research.google/blog/sensible-agent-a-framework-for-unobtrusive-interaction-with-proactive-ar-agents/) ·
[CONCORD privacy-aware](https://arxiv.org/pdf/2604.13348)

Agentic / smart home: [Home Assistant AI](https://www.home-assistant.io/blog/2025/09/11/ai-in-home-assistant/) ·
[acon96/home-llm](https://github.com/acon96/home-llm) ·
[InsiderLLM guide](https://insiderllm.com/guides/home-assistant-local-llm-guide/)

Companion robots / emotion: [companion robots for elderly](https://www.mdpi.com/2078-2489/16/11/948) ·
[multimodal emotion recognition survey](https://arxiv.org/pdf/2312.05735) ·
[social robots reshaping interaction](https://roboticsandautomationnews.com/2025/10/31/social-and-companion-robots-more-than-just-machines/96093/)

Embodied / spatial: [HoloAgent 3D spatial memory](https://arxiv.org/pdf/2606.23565) ·
[Embodied AI paper list](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List)
