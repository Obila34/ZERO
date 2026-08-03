# 2 Architecture

ZERO is arranged around a single organising rule: there is one conversation loop,
it runs on one thread, and nothing is allowed to block it. Every faculty that
could take an unpredictable amount of time, which in practice means every faculty
that touches a model, a network socket or a disk, runs on a thread of its own and
either publishes its result to a place the loop can read at leisure or is joined
under an explicit timeout. When one of them is slow, the loop does not wait for
it. When one of them fails, the loop does not stop. This is why the architecture
looks the way it does, and almost every structural decision in this section
follows from it.

The system spans two machines. The Raspberry Pi 5 in the AF-1 head runs the ZERO
process itself and keeps all the work that is real-time or physically bound to
the robot: capturing the microphone, detecting the wake word, deciding when a
person has stopped speaking, reading frames off the camera, detecting objects,
tracking them, and pushing audio to the speaker. A separate GPU node runs the
model servers: speech recognition, expressive speech synthesis, the language
model, depth estimation, the vision-language model, and the embedding services
for faces, voices and objects. The two are joined by exactly one SSH tunnel that
carries five forwarded ports. Nothing is exposed to the internet, and there is no
second link to fall back to, which is why the failure behaviour of that tunnel is
treated as an architectural concern rather than an operational one.

## 2.1 The Two Machines

On the Pi, everything lives inside one Python process started as
`python -m zero.main`. That process holds the state machine, the audio devices,
the camera, the memory database, the tool registry, the identity registry and the
HTTP control plane. It is deliberately a single process rather than a set of
cooperating services, because every faculty in the system needs cheap access to
the same conversation history, the same memory store and the same speaker
identity, and paying inter-process serialisation costs on the reply path would
show up directly as latency the user can hear.

On the GPU node the arrangement is the opposite: several independent server
processes, each owning one model, each addressable over HTTP, each restartable
without disturbing the others. The Whisper server on port 9000 runs
large-v3-turbo through faster-whisper and answers a POSTed WAV with a transcript.
The Orpheus server on port 9100 runs a 3B speech model through vLLM and answers
text with 24 kHz audio. The vision server on port 8000 is a FastAPI application
exposing a health probe, a `/facts` endpoint that turns a frame plus its
detections into per-object distance and bearing using Depth Anything V2 with
camera intrinsics, and an `/analyze` endpoint that runs those facts and then a
vision-language model grounded in them. Ollama on port 11434 serves the
conversational model, which as configured is `gemma4:latest`, an 8B model at
roughly 100 tokens per second on that hardware. A SearXNG instance on port 8080
backs the web-search tool.

The tunnel is opened from the Pi by `scripts/pi_tunnel.sh` and maintained by
autossh, which rebuilds it within roughly 45 seconds of a stall. It forwards
local 9000, 9100, 8000 and 8080 straight through to the same ports on the node.
The one asymmetry is Ollama: local port 11435 forwards to remote 11434, so that
the Pi's own Ollama, if one is installed for local fallback, can keep 11434 for
itself without a collision. The config reflects this, with `llm.host` pointing at
`http://127.0.0.1:11435`. It is worth knowing that this offset exists, because a
tunnel that appears healthy on 9000 while `llm.host` has been edited to 11434
produces a system that transcribes perfectly and then cannot think.

Both boxes are intended to run under systemd in production. The units in
`scripts/systemd/` cover the tunnel and the ZERO process on the Pi and the model
servers on the node. The shell script `scripts/run_gpu_servers.sh` exists for
interactive use and is idempotent: it checks whether each port is already
listening and skips anything that is up, launching what is missing under
`setsid nohup` so it survives the shell that started it. That script also sets
`OLLAMA_FLASH_ATTENTION` and `OLLAMA_KV_CACHE_TYPE=q8_0`, which halves the KV
cache footprint of the language model. This is a memory-pressure decision rather
than a speed one. The card is shared by the conversational model, Whisper,
Orpheus, the detector, CLIP and the face models, and when it fills, the chat
model gets evicted and the next reply pays a multi-second reload. Quantising the
cache is what keeps it resident.

## 2.2 The Process Set on the Pi

The Pi process is best understood as one loop plus two kinds of thread. Persistent
threads start once and run for the life of the process. Transient threads are
created for a single turn, or a single background job, and are gone within
seconds. Every thread in the system is a daemon thread, so nothing can keep the
process alive after the main loop exits.

The persistent set begins with the PortAudio callback thread inside `MicCapture`,
which is the only thread that touches the microphone. It delivers 30 ms frames of
16 kHz mono audio onto a bounded queue, and every downstream audio consumer, the
wake word and the endpointer alike, pulls from that one queue. Having a single
capture rather than one per consumer is what stops the stages fighting over the
device. That thread also carries two pieces of resilience worth naming here: a
software gain multiplier, because some USB microphones capture too quietly for
the wake model to fire, and an automatic resample path for devices that refuse
16 kHz and can only open at 44.1 or 48 kHz. The queue is also where the echo
guard lives: while ZERO is thinking or speaking, capture is paused rather than
drained downstream, so the robot cannot transcribe its own voice off the speaker.

The camera has the same shape. `CameraStream` runs its own thread doing nothing
but pulling frames, and the `Eyes` thread consumes them. Keeping the grab
separate from the perception work means a slow detection pass cannot cause frames
to back up inside the driver. Above `Eyes` sit three more persistent threads,
each of which exists to keep expensive work off the loop: the `Narrator`, which
periodically asks the vision-language model to describe the scene; the
`SurpriseGate`, which scores world events by prediction error and turns the
unexpected into stored episodes; and the proactive `TriggerSource`, which watches
for a recognised person arriving, a curiosity question worth asking, or an idle
moment suitable for memory consolidation. The control server contributes one more
persistent thread, a `ThreadingHTTPServer` on port 8090, and the optional camera
web preview contributes another on 8008, bound to localhost by default so an
unauthenticated video stream is never published to the network by accident.

The transient threads are where the latency work happens, and they are the part
of the architecture most worth understanding. When the endpointer notices the
person has paused, well before it is willing to declare the utterance finished,
it starts a `stt-spec` thread that transcribes the audio captured so far. That
speculative transcript serves two purposes. If the pause turns out to be the real
end of the utterance, the transcript is already in hand and the round trip to the
GPU has been overlapped with the silence wait rather than added after it. If the
pause turns out to be mid-sentence, the transcript is still useful, because the
endpointer inspects its last word to decide whether the speaker had actually
finished a thought. Either way the work is not wasted. Once the utterance is
confirmed, a `stt` thread runs the final transcription in parallel with identity
and diarisation, which need only the audio and not the text. A `recall` thread
searches memory for anything relevant to what was just said, and is joined under
a hard 300 ms budget, after which the turn simply proceeds without the recall
note rather than becoming slower. An `llm-stream` thread starts generation in the
background so that the model's prefill overlaps the spoken filler, and a
`tts-producer` thread synthesises the next sentence while the current one is
still playing. A `bargein` thread watches the microphone for the entire time ZERO
is making sound, so speech or a wake word can interrupt it. Two more transient
threads run outside the reply path: `compaction`, which folds trimmed-away turns
into a rolling summary, and `memory-save`, which writes the session to disk after
the conversation ends so the robot returns to listening for the wake word
immediately instead of going deaf for several seconds. Each active timer or
reminder holds a thread of its own until it fires.

**TABLE 1 · PROCESS SET**

*A ready-styled Word version of this table, matching the template's own table
markup, is at `tables/table-01-process-set.docx`. Open it, select the title line
and the table together, and paste into the document. The markdown below is the
same content in plain form.*

| PROCESS / THREAD | RUNS ON | FUNCTION | RATE |
|---|---|---|---|
| `MainThread` | Pi 5 | The four-state conversation loop | Event driven |
| PortAudio callback | Pi 5 | Microphone capture, gain, resample, echo pause | 30 ms frames |
| `CameraStream` | Pi 5 | Frame grab only | 640 × 480, 30 fps requested |
| `Eyes` | Pi 5 | Tiered perception, colour, tracking, scene diff | Per frame, gated |
| `Narrator` | Pi 5 | Tier 2 scene narration via the VLM | 3.0 s, budget capped |
| `SurpriseGate` | Pi 5 | Prediction-error scoring into episodes | On world change |
| `proactive` | Pi 5 | Presence, curiosity, idle consolidation | 3.0 s tick |
| `zero-control` | Pi 5 | HTTP control plane | Request driven, port 8090 |
| `EyesWebPreview` | Pi 5 | Optional camera preview | Request driven, port 8008 |
| `stt-spec` | Pi 5 | Speculative transcription at first pause | Once per pause |
| `stt` | Pi 5 | Final transcription | Once per turn |
| `recall` | Pi 5 | Relevance search in memory | Once per turn, 300 ms cap |
| `llm-stream` | Pi 5 | Background generation | Once per turn |
| `tts-producer` | Pi 5 | Synthesis ahead of playback | Once per turn |
| `bargein` | Pi 5 | Interrupt monitor while speaking | Once per turn |
| `compaction` | Pi 5 | Rolling summary of trimmed turns | On trim, single flight |
| `memory-save` | Pi 5 | Per-speaker durable write, corpus append | Once per session |
| `timer-<id>` | Pi 5 | One per pending timer or reminder | Until fired |
| Whisper server | GPU node | Speech to text, large-v3-turbo | Port 9000 |
| Orpheus server | GPU node | Expressive speech, 3B via vLLM | Port 9100 |
| Vision server | GPU node | Depth, scene facts, VLM, embeddings | Port 8000 |
| Ollama | GPU node | Conversational model, gemma4 8B | Port 11434 |
| SearXNG | GPU node | Web search backend | Port 8080 |

## 2.3 Tiered Perception

Vision is the one faculty that would happily consume the entire machine if it
were allowed to, so it is organised into three tiers with a strict cost ceiling
on each. The tiering is not an optimisation added later; it is the reason vision
can be always-on at all, and always-on is what allows the robot to answer a
question about the room without a visible pause while it looks.

Tier 0 is motion detection, and it runs on every single frame. It is deliberately
crude: the frame is downscaled to 160 pixels wide, differenced against the last,
and the fraction of changed pixels compared against a threshold of 0.02. It costs
almost nothing, it publishes a motion level to the shared `WorldState`, and its
real job is to gate the tier above it. Motion is considered to have stopped only
after 15 consecutive still frames, which prevents the gate flapping on noise.

Tier 1 is object detection with colour naming and tracking. While the scene is
moving it runs on every frame. Once the scene goes still it drops to a keepalive
pass every 2 seconds, after a 3 second linger at full cadence so that a brief
pause in movement does not immediately starve the detector. Above the motion gate
sits a second, harder constraint: a duty budget that caps the detector at 60% of
loop time regardless of what the motion gate says. The budget wins. This matters
because motion and cost are not the same thing; a busy scene can ask for more
detection than the Pi can afford, and without the ceiling the perception loop
would starve the frame grab and the whole camera would stutter. When a frame is
gated out, the previous detections are held as current but the fresh frame is
still published, so identity, keyframes and the preview all keep working and the
system never looks blind just because it chose not to re-detect.

Tier 2 is narration, where the vision-language model is asked for a sentence
about the scene. It runs at most every 3 seconds and is additionally capped at 20
inferences per minute, and it self-skips while nothing has changed. This is the
only vision tier that reaches the GPU on a schedule rather than on demand.

All three tiers publish into one `WorldState` object, and that is the read
surface everything else uses. The main loop, the proactive watcher and the
surprise gate do not call the camera or the detector; they read a snapshot, or
wait for a change. Keeping a single published state rather than letting consumers
pull from the perception stack directly is what allows the tiers to be re-rated,
gated or disabled entirely without any consumer noticing.

## 2.4 The Factory and the Fallback Chain

Every engine in the system is constructed in exactly one file, `zero/factory.py`.
It maps the engine names in `config.yaml` to concrete classes and nothing else in
the codebase does that mapping, which is what makes an engine swap a
configuration change rather than a code change. The factory also owns a
convention that matters more than it first appears: a faculty that is disabled,
or whose model files are missing, or whose optional dependencies are not
installed, returns `None` rather than raising. The main loop is written to expect
`None` from every optional faculty. This is why a Pi with no camera runs
voice-only, a Pi with no face model runs voice-only identity, and a machine with
neither identity model runs anonymously, all without a code path dedicated to
each case.

Layered on top of that is the fallback chain, which is the architectural
expression of the local-fallback commitment stated in Section 1. Each faculty
that prefers the GPU is wrapped in a fallback object that holds the remote
implementation and a builder for the local one. The pattern appears six times:
`FallbackSTT` over the remote Whisper client with whisper.cpp behind it,
`FallbackTTS` over Orpheus with Piper behind it, `FallbackDetector` over the
server-side detector with local YOLO behind it, and `FallbackSpeaker`,
`FallbackFace` and `FallbackObjectEmbedder` over their server-side embedders with
local models behind them. In every case the local implementation is built lazily,
on the first remote failure, rather than at startup. That laziness is deliberate:
loading whisper.cpp and a Piper voice into an 8 GB Pi that is also holding the
detector and the identity models would cost memory that is only needed if the
tunnel ever drops.

The last piece of the chain is that degradation is visible rather than silent.
Each fallback wrapper exposes a `degraded` flag, and the main loop polls those
flags once per turn. On the transition into or out of degradation it appends a
note to the outgoing prompt telling the model, in effect, that its fast hearing
is down and it is transcribing locally, or that it is speaking through its backup
voice. The robot can then say so when asked why it is slow, instead of behaving
as though nothing has changed. Treating self-knowledge of failure as part of the
architecture, rather than as a logging concern, is what keeps a degraded system
honest to the person in front of it.

## 2.5 The Control Plane Inside the Process

The HTTP control plane deserves its own subsection because its placement is the
single most consequential integration decision in the system. It runs as a daemon
thread inside the ZERO process, not as a sibling service. The consequence is that
a push-to-talk turn arriving from the AF-1 application over HTTP is handled by
the same `Conversation` object, the same SQLite memory, the same tool registry,
the same voice and the same physical speaker as a turn spoken into the
microphone. There is no synchronisation problem between a network brain and a
local brain, because there is only one brain.

The server exposes a health probe that answers immediately at startup, before the
model is warm, with a `ready` flag that flips true once the language model has
been pinned in memory. This lets the application distinguish a Pi that is booting
from a Pi that is broken. Beyond health it offers a status endpoint reporting
state, the last external turn and the degradation flags; a say endpoint that
speaks a line on the Pi speaker without involving the model; a turn endpoint that
accepts raw audio in any container ffmpeg can decode and runs a complete brain
turn; a text turn endpoint for typed input; and a control endpoint whose only
action is to end an open conversation. External turns are serialised against each
other by one lock and kept off the native loop's think and speak phase, so the
microphone and the network cannot both be driving the state machine at once.

CORS is deliberately wide open, because the Tauri application fetches from the
Rust side and needs none while browser development needs a wildcard. The server
binds `0.0.0.0` by design, which means it is reachable by anything on the local
network. That is a real attack surface and is treated as such in Section 10
rather than hidden here.

---

## Figures for this section

Four figures. Files live in `figures/`. Insert at 7.17" wide, wrap Top and
Bottom, caption in Courier 8 pt all caps, figure note in serif 9.5 pt grey.

---

### FIGURE 8 · SYSTEM DATA FLOW

**File:** `figures/fig-08-system-data-flow.svg`

**Place:** at the very start of Section 2, immediately after the two opening
paragraphs and before 2.1. This is the block diagram the template asks for, and
everything after it is an elaboration of one part of it. Print it at full width;
if the document is ever bound, this is the page to allow a fold-out for.

The master sheet. Three horizontal zones separated by two dashed boundaries: the
GPU node above the tunnel boundary, the Pi process between the boundaries, and
the persistent stores below the disk boundary. Inside the Pi, five labelled rows,
SEE, HEAR, KNOW, SPEAK and KEEP, carry the flow left to right. Everything in the
upper tier converges on a tall PROMPT ASSEMBLY block, and a sheet connector
labelled A carries that into the lower tier, which runs generation through to the
speaker and then to persistence.

Direction is shown without a single arrowhead, three ways at once: connectors are
tapered, wide at the source and narrow at the destination; layout runs left to
right, stated in the legend; and the tier break uses a standard drawing-sheet
continuation connector. Round trips across the tunnel are drawn as accent cables
rather than tapers, because they are request and response rather than one-way
flow. Blocks carrying a small disk glyph read or write persistent state without
needing a drawn cable to it.

**Caption:**

> FIGURE 8 · SYSTEM DATA FLOW

**Figure note:**

> Two dashed boundaries divide the sheet: above the upper one is the GPU node,
> below the lower one is disk, and between them is the single Pi process. The SEE
> row runs continuously and independently of the others, which is why a question
> about the room can be answered without a pause. Tapered paths carry one-way
> data; accent cables are round trips across the tunnel.

**Body sentence before the image:**

> Figure 8 is the whole subsystem on one sheet, and the rest of this section
> works through it a zone at a time: the two machines and the boundary between
> them, the threads that populate the middle zone, the tiering that keeps the SEE
> row affordable, and the fallback arrangement that decides which side of the
> tunnel boundary any given faculty is actually running on.

---

### FIGURE 3 · TURN TIMING CHART

**File:** `figures/fig-03-turn-timing.svg`

**Place:** at the end of 2.2, after the paragraph ending "until it fires."

A logic-analyser style trace. One horizontal lane per thread, a common time axis
across a single turn from wake to first audio, and a bar in each lane showing
exactly when that thread is alive. It shows the overlaps that prose cannot: the
speculative transcription running inside the silence wait, identity and
diarisation running alongside the final transcription, generation starting under
the filler, and synthesis running one sentence ahead of playback.

**Caption:**

> FIGURE 3 · TURN TIMING CHART

**Figure note:**

> Lanes are threads, the axis is one turn. Every overlap in this chart is
> deliberate. The work that would otherwise sit end to end on the reply path has
> been moved sideways into the silence, the filler and the preceding sentence.

**Body sentence before the image:**

> Figure 3 puts the whole set on one time axis for a single turn, and the shape
> of the architecture is easier to read there than in any list: almost nothing in
> the reply path happens after the thing it depends on, because almost everything
> has been started before it was needed.

---

### FIGURE 4 · PERCEPTION TIER CADENCE

**File:** `figures/fig-04-tier-cadence.svg`

**Place:** at the end of 2.3, after the paragraph ending "without any consumer
noticing."

Three stacked lanes over a shared 12 second axis, drawn as tick marks rather than
bars. Tier 0 ticks on every frame. Tier 1 ticks densely while a shaded motion
band is active, then thins to one tick every 2 seconds after the 3 second linger
expires. Tier 2 ticks every 3 seconds. A duty meter on the right shows the 60%
ceiling against actual usage.

**Caption:**

> FIGURE 4 · PERCEPTION TIER CADENCE

**Figure note:**

> The shaded band is motion. Tier 1 follows it, then decays to keepalive after
> the linger expires. The duty ceiling is independent of motion and overrides it,
> which is what protects the frame grab when the scene is busier than the Pi can
> afford.

**Body sentence before the image:**

> Figure 4 shows the three cadences against one another over a period in which
> the room goes from moving to still, which is the case that makes the gate and
> the budget do visibly different jobs.

---

### FIGURE 5 · FALLBACK CHANGEOVER

**File:** `figures/fig-05-fallback-changeover.svg`

**Place:** at the end of 2.4, after the paragraph ending "honest to the person in
front of it."

Drawn as an electrical changeover schematic. Six faculties as rows. Each row has
a remote contact on the tunnel rail and a local contact on the Pi rail, joined by
a changeover switch shown resting on the remote side. The local side is drawn
dashed to signal that it is not yet constructed. A degraded flag line runs from
each switch to a common bus feeding the persona prompt.

**Caption:**

> FIGURE 5 · FALLBACK CHANGEOVER

**Figure note:**

> Six faculties, one pattern. The local contact is drawn dashed because it does
> not exist until the first remote failure builds it. The flag bus is what lets
> the robot tell the person it is running on backup rather than simply behaving
> worse.

**Body sentence before the image:**

> Figure 5 draws the pattern once rather than six times, since the six faculties
> differ only in which models sit on either side of the switch.
