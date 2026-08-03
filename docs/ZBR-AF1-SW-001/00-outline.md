# ZBR-AF1-SW-001 · ZERO Brain Build / Intelligence Stack
## Document outline

Front matter carries the cover page, the document control block (document ID,
owner and reviewers, repository and commit SHA, target hardware, related
documents, revision history) and the contents page. Section order follows the
template and does not change.

---

### 1 Purpose & Scope
What it does, what it does not, who reads this. The single user-visible
behaviour, the four-state loop stated at a glance, the two design commitments
(GPU offload, local fallback everywhere), the in-scope faculty list, the explicit
out-of-scope boundaries with their owning documents, and the four reader
profiles with the sections each can stop at.
*Status: drafted.*

### 2 Architecture
The block diagram and the process set. Pi 5 head and GPU node as the two
physical blocks joined by one SSH tunnel. Inside the Pi process: the main loop
thread plus every background thread it owns (eyes, surprise gate, proactive,
control server, speculative STT, recall, compaction, memory save, barge-in
monitor). On the GPU node: the Whisper server, the Orpheus voice server, the
vision server with its depth and VLM components. One table row per running
process, thread or loop, with where it runs, what it does and its rate. Covers
the factory pattern in `zero/factory.py` that builds every faculty from config
and returns `None` cleanly when a faculty is disabled or unavailable.

### 3 Interfaces & Data Contracts
Everything that crosses the subsystem boundary. The HTTP control plane on port
8090 in full: `/health`, `/zero/status`, `/zero/say`, `/zero/turn`,
`/zero/turn_text`, `/zero/control`, with request and response shapes, status
codes, the audio decode path through ffmpeg, the body size cap and the CORS
posture. The GPU server endpoints ZERO calls and the payload schemas in
`zero/vision/schemas.py` and `server/vision/shared/schemas.py`. The tunnel port
map. The internal event bus contract. Audio device contracts, sample rates and
block sizes. The on-disk contracts: the SQLite schema, the corpus file format,
the taught-object store and the voiceprint registry.

### 4 Behaviour & Logic
The longest section. The state machine and its legal transitions. Wake detection
and the near-miss scoring path. Endpointing, including the semantic hold that
distinguishes a pause in the middle of a thought from the end of one, and the
tri-state hold decision that stopped half-sentences shipping. Speculative
transcription and the prefix check that decides whether the speculative result
can be reused. Identity fusion, the write-confidence threshold that protects one
person's memory from another's voice, guest clustering and its quality gates,
diarisation. The visual-question classifier and why bare demonstratives and
polysemous verbs were deliberately excluded from it. Vision note attachment,
presence grounding against the detector rather than the model's imagination, and
identity-note deduplication across turns. Relevance recall under a hard time
budget. Affect and cross-turn mood. Streaming generation, the filler race,
barge-in and what gets written to history after an interruption. Command paths
that bypass the model: enrolment, object teaching, preferences, erasure, stop
phrases. Rolling compaction. End-of-session persistence split per speaker.
Proactive triggers, curiosity and the adaptive cooldown policy. The reward and
episode loop.

### 5 Build & Deploy
Environment and dependencies for both boxes. The Pi setup script, the virtual
environment, the audio stack, Ollama and Piper. The optional vision requirements
and why they are separate and torch-free. Model acquisition, the ONNX exports
and the model pruning script. GPU node setup and the server launch script. The
SSH tunnel, its systemd unit and its key material. The systemd units for ZERO
itself. Device index discovery and the `config.local.yaml` override. First-run
verification.

### 6 Testing & Validation
The test matrix. Unit coverage in `tests/`. The microphone and voice ID harnesses.
The detection and world benchmarks. The wake-word scoring script and how a
near-miss log becomes a threshold change. Acceptance criteria per faculty with
numeric thresholds: wake false-accept and false-reject rates, endpoint latency,
transcription accuracy on the evaluation set, first-token and first-audio
latency, detection mAP and frame rate, identity precision at the write
threshold. Manual test procedures for the behaviours no automated test can cover.

### 7 Failure Modes & Recovery
One row per failure with symptom, detection, response and safe state. Tunnel
loss and the degradation of STT and TTS to local engines, plus the self-state
narration that makes the robot honest about it. Camera absent at startup and the
voice-only path. GPU server down or slow. Ollama cold or out of memory. Audio
device disappearing. Microphone echo and the mute-during-speech discipline.
Whisper hallucinating text from near silence and the gates that stop it minting
phantom guests. Memory database lock or corruption. Control server exceptions.
The single-instance lock. What is deliberately caught and logged rather than
raised, and why a faculty failure must never take down the loop.

### 8 Configuration & Calibration
Every key in `config.yaml`, grouped by its top-level block: audio, privacy,
control, conversation, memory, learning, perception, proactive, preferences,
tools, identity, voiceid, wake, vad, stt, llm, tts, vision, world. Each key gets
its type, default, effect and the failure mode of setting it wrong. The deep
merge with `config.local.yaml`. Calibration procedures: microphone gain and
device selection, wake threshold from the near-miss log, VAD silence and padding
windows, identity thresholds, camera intrinsics, and the retrieval budget.

### 9 Performance & Resource Budget
CPU, RAM, storage, latency, power and thermal on the Pi 5, measured rather than
estimated. The end-to-end latency chain broken into wake, endpoint, transcribe,
prefill, first token, first audio, with the streaming and speculative
optimisations attributed to the milliseconds they save. Token accounting and the
KV cache budget on the GPU. Memory growth per day and the consolidation that
bounds it. The 8 GB residency problem and the tradeoffs available.

### 10 Security, Privacy & Data Handling
What personal data the system holds: voiceprints, face embeddings, transcripts,
derived facts and preferences. Where each lives, how long it is kept and how it
is erased. The privacy guard modes and the per-turn store decision. The bystander
gate and why in strict mode an unknown voice is not even transcribed. The visible
recording indicator. Secrets and key material for the tunnel. The attack surface
of an open control port on a local network and the mitigations in place.

### 11 Observability & Diagnostics
The logging setup and levels, what each faculty logs and at which level. The
health endpoints and the status payload. The near-miss and degradation logs. The
camera preview tool. How to read a session from the logs end to end. Which log
lines are load-bearing for triage and must not be removed.

### 12 Operations Runbook
Start, stop, update, roll back and triage, as exact commands for both boxes.
Bringing up the tunnel. Restarting a single GPU server without restarting ZERO.
The health-check script. The field triage table mapping observed symptom to the
log line to the fix. Deploying a config change safely. Backing up and restoring
the memory database.

### 13 Dependencies & Licences
Third-party code, models and weights with versions and licences. The models in
particular: the wake model, silero, whisper.cpp and its weights, the language
model, Piper voices, Fish and Orpheus, YOLO11 and YOLOv8-world, Depth Anything
V2, the embedding model. Supply-chain risk notes on each, especially weights
without a clear redistribution licence.

### 14 Open Items & Decisions Log
Known gaps and deferred work. The decisions worth recording with their reasoning:
why the session is owned by voice rather than by face, why perception runs
continuously instead of on demand, why the control plane lives inside the process,
why speculation is skipped while STT is degraded, why guests are minted only after
a transcript proves the turn was real.

### Appendices
Glossary, the full SQLite schema, the message and payload schemas, the complete
default `config.yaml` as shipped, port and device maps, and the detailed changelog.
