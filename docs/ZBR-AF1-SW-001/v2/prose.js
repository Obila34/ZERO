// All prose for ZBR-AF1-SW-001, v2.
// Each section is an array of paragraphs. Subsections carry a Roman numeral.
// No em-dashes; commas, periods, parentheses only.

// ── Helpers exported as sentinels the builder recognises ────────────────
const SUB = (roman, title) => ({ kind: 'sub', roman, title });
const FIG = (num, place) => ({ kind: 'fig', num, place });
const H = (title) => ({ kind: 'sec', title });

module.exports = {

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1  ·  PURPOSE AND SCOPE
// ═══════════════════════════════════════════════════════════════════════════
section1: [
  H('Purpose and Scope'),

  `ZERO is the conversational intelligence subsystem of the AF-1, and it is
the part of the robot a person actually talks to. Everything the subsystem does
resolves into a single user visible behaviour: somebody speaks near the robot,
and the robot answers out loud in its own voice, promptly, knowing who is
speaking, remembering what has been said before, and aware of what is currently
in front of its cameras. Every module described in this document exists to
make that one behaviour happen, and to keep making it happen when parts of the
system fail. It is the yardstick every design choice in the sections that
follow is measured against, and if a choice seems strange, the reason will
almost always be that it protects that single behaviour against a particular
class of failure or delay.`,

  `Inside the AF-1 head, the subsystem runs as a single Python process on the
Raspberry Pi 5, started as python -m zero.main. That process holds a
four state loop that turns continuously. In IDLE the robot rests with the
microphone open and only the wake word detector consuming frames. On a wake
detection the loop moves to LISTENING and captures the utterance until a voice
activity endpointer decides the person has finished speaking. It then moves to
THINKING, where transcription, speaker identification, emotional read, memory
recall and language model generation all run, and several of them run in
parallel. Finally the loop enters SPEAKING, where the reply is synthesised and
played back sentence by sentence, and then it returns to LISTENING for the
next turn without requiring the wake word again. The conversation stays open
until the person says a stop phrase, or falls silent long enough to trip the
configured sleep timeout, at which point the robot returns to IDLE and writes
the session to memory in the background. The legal moves between those four
states are declared as data in zero/state.py and checked on every transition,
so a wiring error surfaces as a logged warning during development rather than
as a silent misbehaviour in front of a person.`,

  FIG(1, 'end'),  // Head cutaway sits at the end of Section 1

  `Two design commitments shape the whole subsystem, and both are worth
stating before any architecture is discussed because almost every structural
decision later follows from them. The first commitment is that heavy models do
not run on the Pi. Speech recognition, the language model, expressive speech
synthesis, depth estimation, the vision language model, and the embedding
services for faces, voices and objects all execute on a separate GPU node
reached over a single SSH tunnel. The Pi keeps only the work that is
real time or physically bound to the robot: microphone capture, wake word
detection, endpointing, object detection, tracking, playback, and the state
machine itself. The second commitment is that every remote faculty holds a
local fallback. If the tunnel drops, transcription falls back to whisper.cpp
running on the Pi CPU, and the voice falls back to Piper. The robot keeps
talking, more slowly and less expressively, and it is deliberately told about
its own degradation so that it can say so honestly instead of behaving as
though nothing had changed. This transparency, treated in Section 2 and again
in Section 7, is not a logging concern. It is the mechanism that lets a
degraded system stay honest to the person in front of it.`,

  `The subsystem is also responsible for the faculties that make the robot
feel present rather than merely transactional, and these are in scope
precisely because they are inseparable from the conversation loop. Identity
fuses a face embedding derived from the current camera frame with a voice
embedding derived from the current utterance, and the fused signal decides who
the robot is talking to and, critically, whose memory a turn is allowed to be
written into. Memory is held in a set of small SQLite stores that keep durable
facts, per person preferences, rolling session summaries and semantic
embeddings for relevance recall. Vision runs continuously from startup rather
than on demand, so the scene is already perceived by the time anyone asks about
it, and perception is never on the critical path of a reply. Tool use gives
the model timers, reminders, web search, and explicit verbs to remember and
recall. A privacy guard decides, per turn, whether an unrecognised voice is
answered at all, and whether their words are allowed to be stored. A proactive
layer lets the robot open a conversation by itself, greeting a person it
recognises walking in, or asking a question about something new it has
noticed. A learning loop turns each exchange into a reward tagged episode and
appends the session to a training corpus for offline fine tuning.`,

  `Finally, the subsystem exposes a small HTTP control plane on port 8090,
running as a daemon thread inside the same process. This is the surface the
AF-1 application uses to reach the robot from other machines on the local
network, and its placement inside the process rather than beside it is one of
the more consequential integration decisions in the design. A push to talk
turn arriving over HTTP drives the same Conversation object, the same SQLite
memory, the same tool registry, the same voice, and the same speaker as a turn
spoken into the microphone. There is one brain, not two, and the network is
simply another way into it.`,

  SUB('I', 'What Is In Scope'),

  `This document covers the ZERO process in full. That means the four state
conversation loop and its transition rules; wake word detection, voice
activity endpointing, and the semantic hold that keeps the robot listening
through a pause in the middle of a thought; speech to text in both its remote
and local forms, including the speculative transcription that starts work at
the first pause rather than waiting for the endpoint; the language model, its
persona prompt, its history window, and the rolling compaction that keeps a
long session bounded; text to speech across the Piper, Fish and Orpheus
engines, together with the orchestrator that translates one shared cue
vocabulary into whatever the active engine understands; and barge in, which
lets a person interrupt the robot mid sentence and have the interrupting words
carried into the next turn so they are never repeated. Each of these carries
one or more configuration keys that live in config.yaml, and this document
names those keys where behaviour depends on their values, so that a claim in
the prose can always be checked against a value in the file.`,

  `The perception and state faculties are in scope on the same footing: the
always on camera loop, open vocabulary and closed vocabulary object detection,
tracking, colour naming, scene phrasing, taught objects, the live world state
model and its surprise gate; face and voice identity, provisional guest
clustering for unfamiliar speakers, and diarisation across a multi speaker
conversation; affect estimation and the cross turn mood it feeds; the memory
schema, embeddings, preferences, consolidation, and erasure on request; the
tool registry and its router; the privacy guard and its visible indicator; the
proactive policy, its triggers, and its adaptive cooldowns; the episode and
reward machinery and the corpus export that feeds fine tuning. On the GPU side
the document covers the servers this subsystem calls and owns the client
contract for: the Whisper server, the Orpheus voice server, and the vision
server with its depth and vision language components. It covers config.yaml in
full, the config.local.yaml override mechanism, the systemd units, the tunnel
and health check scripts, and the HTTP control plane contract.`,

  SUB('II', 'What Is Out of Scope'),

  `Nothing below the neck belongs to this document. Locomotion, balance, the
arms and end effectors, and any motion planning or safety interlock are
covered by the motion control document and are not reachable from this
subsystem. The chest Pi 4B and the Nano compute nodes are named here only
where ZERO exchanges messages with them, and their own processes, boot order,
and firmware are documented separately. Power delivery, thermal design,
mechanical assembly, and teardown are outside this document entirely and live
in the Engineering Corpus and teardown volumes.`,

  `The AF-1 application is out of scope as a product. This document defines
and owns the HTTP contract the application calls, and it stops at that
boundary. The application's own interface, its voice picker, its state
handling, and its packaging are documented by the application team.
Provisioning of the GPU node is likewise out of scope: this document specifies
what the servers must expose and how ZERO behaves when they are unreachable,
but the machine's operating system, drivers, and physical hosting are not
described here. Model training is out of scope beyond the point where this
subsystem writes the corpus; the fine tuning pipeline that consumes that
corpus is documented on its own. Cloud or hosted language model providers are
out of scope by design, because the subsystem is built to run without internet
access apart from the optional web search tool.`,

  SUB('III', 'Intended Readers'),

  `This document is written for four readers, and each of them should be
able to stop at a different depth. An engineer joining the voice stack should
be able to read Sections 1 through 4 and understand the loop well enough to
trace a single utterance from microphone to speaker, then use Sections 5 and
8 to get a working machine on their bench. An operator or field technician
should be able to work almost entirely from Sections 7, 11, and 12, which
cover what breaks, what the logs are saying, and exactly which commands
restart which box. An integration engineer building against the robot from the
AF-1 application needs Section 3, which is the authoritative interface
contract, and Section 7 for the failure semantics they must handle. A
technical reviewer conducting due diligence should find Sections 9, 10, 13,
and 14 sufficient: what the system costs in compute and power, what it does
with personal data, what it depends on and under which licences, and what is
still open. Everything in this document is written against the repository and
commit named on the cover page. Where behaviour is configurable, the exact
config.yaml key is given rather than described, so that a reader who wants to
verify a claim in the prose can always compare it against a value in the
file.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2  ·  ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════════════
section2: [
  H('Architecture'),

  `ZERO is arranged around a single organising rule: there is one
conversation loop, it runs on one thread, and nothing is allowed to block it.
Every faculty that could take an unpredictable amount of time, which in
practice means every faculty that touches a model, a network socket, or a
disk, runs on a thread of its own and either publishes its result to a place
the loop can read at leisure or is joined under an explicit timeout. When one
of those threads is slow, the loop does not wait for it. When one of them
fails, the loop does not stop. This is why the architecture looks the way it
does, and almost every structural decision in this section follows from that
rule.`,

  `The subsystem spans two machines. The Raspberry Pi 5 in the AF-1 head
runs the ZERO process itself and keeps all the work that is real time or
physically bound to the robot: capturing the microphone, detecting the wake
word, deciding when a person has stopped speaking, reading frames off the
camera, detecting objects, tracking them, and pushing audio to the speaker. A
separate GPU node, on the same local network, runs a small set of model
servers. Speech recognition, expressive speech synthesis, the language model,
depth estimation, the vision language model, and the embedding services for
faces, voices, and objects all live there. The two are joined by exactly one
SSH tunnel that carries five forwarded ports. Nothing is exposed to the
internet, and there is no second link to fall back to, which is why the
failure behaviour of that tunnel is treated in this document as an
architectural concern rather than an operational one.`,

  FIG(2, 'here'),

  `Figure 22, placed at the end of this section, walks a whole turn
through the system as a numbered data flow: microphone frames enter on
the left, are transcribed and reasoned about in the middle, and leave
as synthesised audio on the right. Every hop names the data type it
carries and the process that carries it. Section 2 is the physical
architecture that Figure 22 flows across.`,

  SUB('I', 'The Two Machines'),

  `On the Pi, everything lives inside one Python process. That process holds
the state machine, the audio devices, the camera, the memory database, the
tool registry, the identity registry, and the HTTP control plane. It is
deliberately a single process rather than a set of cooperating services,
because every faculty in the system needs cheap access to the same
conversation history, the same memory store, and the same speaker identity,
and paying inter process serialisation costs on the reply path would show up
directly as latency the user can hear. There are a dozen daemon threads
running inside that process, and they will be catalogued shortly, but from
the outside the observable unit is one process managed by one systemd unit,
which either is or is not running.`,

  `On the GPU node the arrangement is the opposite: several independent
server processes, each owning one model, each addressable over HTTP, each
restartable without disturbing the others. The Whisper server on port 9000
runs large-v3-turbo through faster-whisper and answers a POSTed WAV with a
transcript. The Orpheus server on port 9100 runs a 3B speech model through
vLLM and answers text with 24 kHz audio. The vision server on port 8000 is a
FastAPI application exposing a health probe, a /facts endpoint that turns a
frame plus its detections into per object distance and bearing using Depth
Anything V2 with camera intrinsics, and an /analyze endpoint that runs those
facts and then a vision language model grounded in them. Ollama on port 11434
serves the conversational model, which as configured is gemma4:latest, an 8B
model at roughly 100 tokens per second on that hardware. A SearXNG instance on
port 8080 backs the web search tool. Each of the model servers can be
restarted independently, and each fails independently, which is the property
that lets the Pi treat their failures as ordinary degradation events rather
than as system wide outages.`,

  `The tunnel is opened from the Pi by scripts/pi_tunnel.sh and maintained
by autossh, which rebuilds it within roughly 45 seconds of a stall. It
forwards local 9000, 9100, 8000, and 8080 straight through to the same ports
on the node. The one asymmetry is Ollama: local port 11435 forwards to remote
11434, so that the Pi's own Ollama, if one is installed for local fallback,
can keep 11434 for itself without a collision. The config reflects this, with
llm.host pointing at http://127.0.0.1:11435. It is worth knowing that this
offset exists, because a tunnel that appears healthy on 9000 while llm.host
has been edited to 11434 produces a system that transcribes perfectly and
then cannot think, which is a much harder failure to diagnose than a system
that fails outright.`,

  `Both boxes are intended to run under systemd in production. The units in
scripts/systemd/ cover the tunnel and the ZERO process on the Pi and the
model servers on the node. The shell script scripts/run_gpu_servers.sh exists
for interactive use and is idempotent: it checks whether each port is already
listening and skips anything that is up, launching what is missing under
setsid nohup so it survives the shell that started it. That script also sets
OLLAMA_FLASH_ATTENTION and OLLAMA_KV_CACHE_TYPE=q8_0, which halves the KV
cache footprint of the language model. This is a memory pressure decision
rather than a speed one. The card is shared by the conversational model,
Whisper, Orpheus, the detector, CLIP, and the face models, and when it fills,
the chat model gets evicted and the next reply pays a multi second reload.
Quantising the cache is what keeps it resident.`,

  SUB('II', 'The Process Set on the Pi'),

  `The Pi process is best understood as one loop plus two kinds of thread.
Persistent threads start once and run for the life of the process. Transient
threads are created for a single turn, or a single background job, and are
gone within seconds. Every thread in the system is a daemon thread, so
nothing can keep the process alive after the main loop exits, which matters
for shutdown correctness under systemd. The main loop itself owns the state
machine and is the only thread that transitions it. Everything else either
publishes into a shared object the loop reads, or is joined by the loop with
an explicit timeout.`,

  `The persistent set begins with the PortAudio callback thread inside
MicCapture, which is the only thread that touches the microphone. It delivers
30 ms frames of 16 kHz mono audio onto a bounded queue, and every downstream
audio consumer, the wake word and the endpointer alike, pulls from that one
queue. Having a single capture rather than one per consumer is what stops the
stages fighting over the device. That thread also carries two pieces of
resilience worth naming here: a software gain multiplier, because some USB
microphones capture too quietly for the wake model to fire, and an automatic
resample path for devices that refuse 16 kHz and can only open at 44.1 or 48
kHz. The queue is also where the echo guard lives: while ZERO is thinking or
speaking, capture is paused rather than drained downstream, so the robot
cannot transcribe its own voice off the speaker.`,

  `The camera has the same shape. CameraStream runs its own thread doing
nothing but pulling frames, and the Eyes thread consumes them. Keeping the
grab separate from the perception work means a slow detection pass cannot
cause frames to back up inside the driver. Above Eyes sit three more
persistent threads, each of which exists to keep expensive work off the loop:
the Narrator, which periodically asks the vision language model to describe
the scene; the SurpriseGate, which scores world events by prediction error
and turns the unexpected into stored episodes; and the proactive
TriggerSource, which watches for a recognised person arriving, a curiosity
question worth asking, or an idle moment suitable for memory consolidation.
The control server contributes one more persistent thread, a
ThreadingHTTPServer on port 8090, and the optional camera web preview
contributes another on 8008, bound to localhost by default so an
unauthenticated video stream is never published to the network by accident.`,

  `The transient threads are where the latency work happens, and they are
the part of the architecture most worth understanding. When the endpointer
notices the person has paused, well before it is willing to declare the
utterance finished, it starts a stt-spec thread that transcribes the audio
captured so far. That speculative transcript serves two purposes. If the pause
turns out to be the real end of the utterance, the transcript is already in
hand and the round trip to the GPU has been overlapped with the silence wait
rather than added after it. If the pause turns out to be mid sentence, the
transcript is still useful, because the endpointer inspects its last word to
decide whether the speaker had actually finished a thought. Either way the
work is not wasted. Once the utterance is confirmed, a stt thread runs the
final transcription in parallel with identity and diarisation, both of which
need only the audio and not the text. A recall thread searches memory for
anything relevant to what was just said and is joined under a hard 300 ms
budget, after which the turn simply proceeds without the recall note rather
than becoming slower. An llm-stream thread starts generation in the
background so that the model's prefill overlaps the spoken filler, and a
tts-producer thread synthesises the next sentence while the current one is
still playing. A bargein thread watches the microphone for the entire time
ZERO is making sound, so that speech or a wake word can interrupt it. Two
more transient threads run outside the reply path: compaction, which folds
trimmed away turns into a rolling summary, and memory-save, which writes the
session to disk after the conversation ends so that the robot returns to
listening for the wake word immediately, rather than going deaf for several
seconds while it flushes. Each active timer or reminder holds a thread of its
own until it fires.`,

  FIG(3, 'here'),

  FIG(4, 'here'),

  SUB('III', 'Tiered Perception'),

  `Vision is the one faculty that would happily consume the entire machine
if it were allowed to, so it is organised into three tiers with a strict cost
ceiling on each. The tiering is not an optimisation added later. It is the
reason vision can be always on at all, and always on is what lets the robot
answer a question about the room without a visible pause while it looks.`,

  `Tier 0 is motion detection, and it runs on every single frame. It is
deliberately crude: the frame is downscaled to 160 pixels wide, differenced
against the previous frame, and the fraction of changed pixels compared
against a threshold of 0.02. It costs almost nothing, it publishes a motion
level to the shared WorldState, and its real job is to gate the tier above
it. Motion is considered to have stopped only after 15 consecutive still
frames, which prevents the gate flapping on noise. Tier 0 is what allows Tier
1 to sleep when nothing is happening.`,

  `Tier 1 is object detection with colour naming and tracking. While the
scene is moving it runs on every frame. Once the scene goes still it drops to
a keepalive pass every 2 seconds, after a 3 second linger at full cadence so
that a brief pause in movement does not immediately starve the detector.
Above the motion gate sits a second, harder constraint: a duty budget that
caps the detector at 60 percent of loop time regardless of what the motion
gate says. The budget wins. This matters because motion and cost are not the
same thing; a busy scene can ask for more detection than the Pi can afford,
and without the ceiling the perception loop would starve the frame grab and
the whole camera would stutter. When a frame is gated out, the previous
detections are held as current, but the fresh frame is still published, so
identity, keyframes, and the preview all keep working, and the system never
looks blind just because it chose not to re-detect.`,

  `Tier 2 is narration, where the vision language model is asked for a
sentence about the scene. It runs at most every 3 seconds and is additionally
capped at 20 inferences per minute, and it self skips while nothing has
changed. This is the only vision tier that reaches the GPU on a schedule
rather than on demand, which is why it also has to justify its own cost by
mattering to the reply, and it therefore stays off unless a change threshold
in world.surprise.narrate_bits has been crossed.`,

  `All three tiers publish into one WorldState object, and that is the read
surface everything else uses. The main loop, the proactive watcher, and the
surprise gate do not call the camera or the detector; they read a snapshot,
or wait for a change. Keeping a single published state rather than letting
consumers pull from the perception stack directly is what allows the tiers to
be re-rated, gated, or disabled entirely without any consumer noticing.`,

  SUB('IV', 'The Factory and the Fallback Chain'),

  `Every engine in the system is constructed in exactly one file,
zero/factory.py. It maps the engine names in config.yaml to concrete classes,
and nothing else in the codebase does that mapping, which is what makes an
engine swap a configuration change rather than a code change. The factory
also owns a convention that matters more than it first appears: a faculty
that is disabled, or whose model files are missing, or whose optional
dependencies are not installed, returns None rather than raising. The main
loop is written to expect None from every optional faculty. This is why a Pi
with no camera runs voice only, a Pi with no face model runs voice only
identity, and a machine with neither identity model runs anonymously, all
without a code path dedicated to each case. The loop does not know it is
degraded; it just does whatever is possible with what it has.`,

  `Layered on top of that is the fallback chain, which is the architectural
expression of the local fallback commitment stated in Section 1. Each
faculty that prefers the GPU is wrapped in a fallback object that holds the
remote implementation and a builder for the local one. The pattern appears
six times: FallbackSTT over the remote Whisper client with whisper.cpp
behind it, FallbackTTS over Orpheus with Piper behind it, FallbackDetector
over the server side detector with local YOLO behind it, and FallbackSpeaker,
FallbackFace, and FallbackObjectEmbedder over their server side embedders
with local models behind them. In every case the local implementation is
built lazily, on the first remote failure, rather than at startup. That
laziness is deliberate: loading whisper.cpp and a Piper voice into an 8 GB
Pi that is also holding the detector and the identity models would cost
memory that is only needed if the tunnel ever drops.`,

  FIG(5, 'here'),

  FIG(6, 'here'),

  `The last piece of the chain is that degradation is visible rather than
silent. Each fallback wrapper exposes a degraded flag, and the main loop
polls those flags once per turn. On the transition into or out of
degradation it appends a note to the outgoing prompt telling the model, in
effect, that its fast hearing is down and it is transcribing locally, or that
it is speaking through its backup voice. The robot can then say so when asked
why it is slow, instead of behaving as though nothing had changed. Treating
self knowledge of failure as part of the architecture, rather than as a
logging concern, is what keeps a degraded system honest to the person in
front of it.`,

  SUB('V', 'The Control Plane Inside the Process'),

  `The HTTP control plane deserves its own subsection because its placement
is the single most consequential integration decision in the system. It runs
as a daemon thread inside the ZERO process, not as a sibling service. The
consequence is that a push to talk turn arriving from the AF-1 application
over HTTP is handled by the same Conversation object, the same SQLite memory,
the same tool registry, the same voice, and the same physical speaker as a
turn spoken into the microphone. There is no synchronisation problem between
a network brain and a local brain, because there is only one brain. Two
things follow directly from that. First, integration testing is really just
regular testing done through a different door. Second, when the loop is in
the middle of a spoken turn, HTTP requests do not race it; they are
serialised behind it by one lock, and held off the loop's think and speak
phase, so the microphone and the network cannot both be driving the state
machine at once.`,

  `The server exposes a health probe that answers immediately at startup,
before the model is warm, with a ready flag that flips true once the language
model has been pinned in memory by the warmup call. This lets the
application distinguish a Pi that is booting from a Pi that is broken. Beyond
health it offers a status endpoint reporting state, the last external turn,
and the degradation flags; a say endpoint that speaks a line on the Pi
speaker without involving the model; a turn endpoint that accepts raw audio
in any container ffmpeg can decode and runs a complete brain turn; a text
turn endpoint for typed input; and a control endpoint whose only action is to
end an open conversation. Everything else the application might want, it
does through those doors. CORS is deliberately wide open, because the Tauri
application fetches from the Rust side and needs none, while browser
development needs a wildcard. The server binds 0.0.0.0 by design, which means
it is reachable by anything on the local network. That is a real attack
surface, and it is treated as such in Section 10 rather than hidden here.`,

  FIG(22, 'end'),
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3  ·  INTERFACES AND DATA CONTRACTS
// ═══════════════════════════════════════════════════════════════════════════
section3: [
  H('Interfaces and Data Contracts'),

  `An interface, for the purposes of this document, is anything that crosses
the boundary drawn around the ZERO subsystem. That includes the HTTP control
plane the AF-1 application calls, every request ZERO sends up the SSH tunnel
to a model server, the physical devices the Pi opens, the GPIO pin that
drives the recording indicator, and every file the subsystem writes to disk.
Internal function calls between modules are not interfaces and are not listed
here; they are covered by the behavioural walk in Section 4. The distinction
matters because everything in this section is a contract that something
outside the subsystem depends on, and therefore cannot be changed without
changing that something else at the same time. A code refactor inside the
process is invisible to the world; a schema change here is not.`,

  `Three conventions run through all of the contracts below and are worth
stating once rather than repeating in every row. First, all audio crossing
any boundary is mono, 16 bit signed PCM, wrapped in a self describing WAV
container, so that the sample rate travels with the data rather than being
agreed out of band. The one exception is speech coming back from Orpheus,
which is 24 kHz because that is what its vocoder produces, and which the Pi
resamples on receipt. Second, all images crossing any boundary are JPEG
encoded and then base64 encoded into a JSON string field named
image_jpeg_b64, at a quality set by vision.gpu.jpeg_quality and defaulting
to 80. Third, every embedding stored on disk is a float32 array written as a
raw little endian blob, always accompanied by an integer dim column in the
same row. Readers compare dim against their own model's dimension and skip
mismatched rows rather than failing, which is what allows an embedding model
to be swapped without invalidating the database.`,

  `There is a fourth convention that is easy to miss and causes real
confusion when missed. Throughout the system, a person identifier is a
signed integer whose sign carries meaning. A positive value is an enrolled
person in the identity registry. A negative value is a provisional guest,
minted by clustering an unfamiliar voice, and deliberately kept separate so
that one stranger's words never merge with another's. A null value is an
anonymous turn, which is what text mode produces and what happens when
identity is disabled or the speaker could not be placed. Every store that
carries person_id, and every corpus record, uses this convention, and code
that treats the field as an opaque key will silently mix guests with people,
which is exactly the failure mode that motivated the convention in the first
place.`,

  SUB('I', 'The Control Plane'),

  FIG(7, 'here'),

  `The control plane listens on TCP 8090, bound to 0.0.0.0 by control.host,
and is the only inbound interface the subsystem exposes. All responses are
JSON unless stated otherwise. A cross origin preflight is answered with 204
and the permissive header set described at the end of this subsection. Any
request body larger than 32 MB is rejected, which is generous for the
intended use since a minute of WebM Opus is well under it. Unhandled
exceptions inside a handler are caught, logged with a traceback, and returned
as HTTP 500 with the exception text truncated to 200 characters, so that a
malformed request can never take down the process that is also holding the
conversation. The failure envelope is uniform across handlers: an ok boolean
field is false on failure, and an error field carries a short
machine readable reason.`,

  `GET /health, also reachable as GET /zero/health, returns
{ok, service, state, ready}. The service field is the constant string
zero-control. The state field is the current conversation state as its
lowercase value, one of idle, listening, thinking, or speaking. The ready
field is the one that matters operationally: it is false from process start
until the language model has been pinned in memory by the warmup call, and
true afterwards. A client that treats a reachable but not ready Pi as broken
will report false failures during the first few seconds after a restart,
which is why the flag exists.`,

  `GET /zero/status returns the state, the last external turn, and the
current degradation flags. This is the endpoint that lets the application
show that the robot is running on its backup voice without having to ask it.
POST /zero/say takes {text, voice?} and speaks the line on the Pi speaker
without involving the language model or writing anything to memory. The text
is truncated to 500 characters. An empty or missing text returns 400 with
{ok:false, error:"empty-text"}. The optional voice is an Orpheus speaker
name such as leo, tara, or draco, and it overrides the configured default
for that request only, leaving ZERO's own default untouched.`,

  `POST /zero/turn is the push to talk path. The request body is raw audio
bytes, not JSON and not multipart, in any container ffmpeg can decode, which
in practice means WebM, OGG, WAV, or MP4. Options travel as query
parameters, ?voice= and ?person_id=. The server shells out to ffmpeg to
convert the body to mono float32 at the configured sample rate. A body that
decodes cleanly runs a complete brain turn, is spoken on the Pi speaker, and
returns {ok, heard, reply}, where heard is the transcript and reply is what
was said. An empty body returns 400. A decode failure returns 422 with the
tail of ffmpeg's stderr in the error field, which is deliberate: a caller
sending an unsupported container needs to see why.`,

  `POST /zero/turn_text takes {text, voice?, speak?, person_id?} and runs
the same brain turn from typed input, with text truncated to 1200 characters
and speak defaulting to true. POST /zero/control accepts {action:"sleep"}
and ends the open conversation; any other action returns 400 naming the
unknown action. Anything else returns 404. External turns are serialised
against each other by a single lock and are held off the native loop's think
and speak phase, so the microphone and the network cannot both drive the
state machine at once. The person_id supplied by the application defaults to
control.person_id, which ships as 1, meaning AF-1 turns are attributed to
the logged in operator rather than to an anonymous speaker.`,

  `CORS is deliberately unrestricted: Access-Control-Allow-Origin: *,
methods GET, POST, OPTIONS, headers Content-Type. The Tauri application
fetches from the Rust side and needs no CORS at all, but browser based
development does, and no authentication is implemented at this layer.
Combined with the 0.0.0.0 bind, this means anything on the local network
can make the robot speak. That is a deliberate trade for a LAN only device
and it is assessed properly in Section 10, not defended here.`,

  SUB('II', 'The Model Server Contracts'),

  `Every one of the model server calls is reached through the SSH tunnel at
a local port, so from ZERO's point of view they are all on 127.0.0.1. The
port offset described in Section 2.I applies: Ollama alone is reached at
local 11435, and the tunnel rewrites the number to 11434 on the far side.`,

  `The Whisper server accepts POST /transcribe with a raw WAV body and
Content-Type: audio/wav, and returns {text}. An optional ?language= query
parameter is appended by the client when stt.language is set, accepting a
language code such as sw, or the literal auto to let Whisper detect, which
is the right setting for code switched speech. The client raises rather than
returning an empty string on failure, and that distinction is load bearing:
the fallback wrapper must be able to tell a dead tunnel from a silent room,
and empty text is the answer for a silent room while an exception is the
answer for a dead tunnel.`,

  `The Orpheus server accepts POST /tts with {text, voice} and returns a
WAV at 24 kHz. A streaming variant at /tts_stream is derived by the client
from the configured URL by simple substring replacement, which means a non
standard path in tts.orpheus.url will produce a stream URL that does not
exist. GET /health returns {ok} reflecting whether the model has loaded.
The client keeps one HTTP session alive across sentences rather than
reconnecting, and retries a connection failure once on a fresh connection so
that a server restarted between turns recovers rather than erroring. Before
sending, the client rewrites ZERO's shared cue vocabulary into Orpheus's
native tags, mapping [laughs] to <laugh>, [sighs] to <sigh> and so on,
rendering [hmm] and [pause] as text the model performs naturally, and
stripping any cue it does not recognise.`,

  `The vision server exposes GET /health, which reports CUDA availability
and VRAM and is what proves the tunnel is alive; POST /facts, which takes an
AnalyzeRequest and returns an AnalyzeResponse whose reply is empty because
ZERO's own model writes the spoken answer; and POST /analyze, which runs the
same facts and then a vision language model grounded in them, returning a
populated reply. The shared wire types are Pydantic v2 models. A Detection
carries label as a string, bbox as exactly four floats in [x, y, w, h] in
pixels of the frame that was sent, confidence constrained to the range 0 to
1, and an optional color string from HSV naming. An AnalyzeRequest carries
image_jpeg_b64, a list of Detection, a question string, and a history list
of {role, content} dictionaries. A SceneFact carries label, optional color,
optional distance_m constrained to be non negative, and an optional bearing
which is one of the coarse strings left, center, or right. An AnalyzeResponse
carries a list of SceneFact and a reply string.`,

  `These models exist in two files, one on each side of the tunnel, because
the two nodes do not share a filesystem. Those two copies have drifted. In
the Pi copy at zero/vision/schemas.py, both AnalyzeRequest.question and
AnalyzeResponse.reply have empty string defaults and are therefore optional.
In the server copy at server/vision/shared/schemas.py, both are declared
required. Today this is benign, because the Pi serialises its own model and
so always emits both keys even when they are empty, but the contract is no
longer single sourced and the next edit to either file can diverge further
without anything failing loudly. The server copy's docstring compounds this
by instructing the reader to edit a canonical file and run python
shared/sync_schemas.py, and by naming a mirror at pi/shared/schemas.py.
Neither that script nor that path exists in this repository. This is
recorded as an open item in Section 14.`,

  SUB('III', 'The Perception Offload'),

  `Separate from the vision endpoints above, the GPU node serves a set of
embedding and detection endpoints under /perceive/ on the same port 8000,
reached through a client with its own timeouts: 5 seconds by default and 10
seconds for detection. Every one of these endpoints raises a RuntimeError
naming the path on any failure, again so that the fallback wrappers can
distinguish an unreachable server from an empty result. POST /perceive/detect
takes {image_jpeg_b64} and returns {detections: [...]} in the Detection
shape. POST /perceive/face takes {image_jpeg_b64, max_faces}, defaulting to
three faces, and returns {faces: [{embedding: [float, ...], bbox: [...]},
...]}; the client discards any face whose embedding array is empty. POST
/perceive/speaker is the one endpoint in this group that does not take JSON:
the body is a raw WAV with Content-Type: audio/wav, built from the utterance
at the capture sample rate, and the response is {embedding: [float, ...]}.
POST /perceive/embed_object takes {image_jpeg_b64} holding a crop rather
than a full frame and returns {embedding: [float, ...]}.`,

  SUB('IV', 'Local Device Interfaces'),

  `The microphone is opened through PortAudio at the index given by
audio.input_device, and the pipeline contract downstream of it is fixed:
mono, 16 bit, 16 kHz, in blocks of audio.block_ms milliseconds, which ships
as 30 and therefore yields 480 samples per block. Two adaptations sit inside
that contract. A software gain from audio.input_gain multiplies each block
before int16 conversion, because some USB and webcam microphones capture too
quietly for the wake model to fire. And when a device rejects 16 kHz
outright, which is common for cheap USB dongles that only support 44.1 or
48 kHz, the stream is opened at the device's native rate and each block is
resampled down, so that every downstream consumer still sees the frames it
would otherwise. The speaker is opened at audio.output_device. The camera is
opened at vision.camera.index and configured for 640 by 480 at a requested
30 frames per second in MJPG, which is deliberately small so that detection
keeps up on the Pi.`,

  `The recording indicator is a single GPIO pin named by
privacy.indicator_gpio_pin, driven through gpiozero. It is lit for the
listening, thinking, and speaking states, and dark for idle. With no pin
configured, or with gpiozero unavailable, it degrades to state transition
log lines so that the same information is still observable through
journalctl. This is the only physical output the subsystem drives. The
optional camera preview serves HTTP on vision.preview_port, which ships as
8008, bound to vision.preview_host. That default is 127.0.0.1 on purpose:
the preview is an unauthenticated video stream, and publishing it to the
network requires an explicit change to 0.0.0.0.`,

  SUB('V', 'On Disk Contracts'),

  `Six SQLite databases and two flat files make up the persistent state.
All of them are created on first use, all of them are opened with
check_same_thread=False because several threads write, and the episode store
additionally runs in WAL mode so that writers never block readers.
zero_memory.sqlite holds one table, memories, with columns id, person_id,
layer, key, value, importance defaulting to 5.0, emb as a float32 blob,
emb_dim, created_at, last_access, access_count, and protected. There is an
index on (layer, person_id). The protected flag marks rows that the storage
cap must never prune, which is what keeps the durable per person last
conversation record alive when ordinary facts are being aged out. This store
also carries a one time legacy migration that folds rows from an older flat
schema, with tables named memory and episodes, into the current shape and
then drops them, so that an old database upgrades in place on first open.`,

  `zero_identity.sqlite holds people, with id, a name that is unique and
case insensitive, and created_at; and embeddings, with id, person_id, kind,
dim, a vec blob, and created_at, with a foreign key to people. The kind
column is what allows face and voice vectors to live in one table.
zero_guests.sqlite holds guest_samples, with id, guest, dim, vec, and
created_at. Guest identifiers are negative by construction: a new guest
takes the minimum of minus one and one below the lowest existing identifier.
The store caps samples per guest and total guests, dropping the least
recently heard. zero_episodes.sqlite holds episodes, with id, v recording
the schema version at write time, ts, kind drawn from turn, scene,
proactive, and action, person_id, a payload JSON string defaulting to {},
reward where null means untagged, surprise, and consolidated_at. It carries
two indexes, one on (kind, ts) and a partial index on unconsolidated rows.
Alone among the stores it uses a real migration mechanism driven by PRAGMA
user_version, applying each migration step in its own transaction. Rewards
are clamped to the range minus one to one on write. zero_curiosity.sqlite
holds questions, with id, a unique source_key that prevents the same
observation queuing twice, text, priority, person_id, created_at, and
asked_at. zero_objects.sqlite holds objects, with id, a case insensitive
name, person_id, dim, vec, and created_at, which is where a taught object
binds a name to an embedding.`,

  `The interaction corpus is newline delimited JSON at
data/corpus/interactions.jsonl, appended under a lock so that records never
interleave. One record is written per speaker per session, holding that
speaker's turns, their identifier under the sign convention described above,
and a timestamp. Because the session has already been split by speaker
before it is written, one person's speech never contaminates another's
training data. The enrolled voiceprint is a NumPy array at the path in
voiceid.profile_path, and the surprise predictor keeps its running
statistics as JSON at world.surprise.stats_path.`,

  FIG(8, 'here'),

  SUB('VI', 'The Internal Event Bus'),

  `The event bus is the one internal mechanism documented here, because it
is how anything that is not the main loop gets the robot to speak, and
because its overflow behaviour is a contract rather than an implementation
detail. An Event carries kind, text already phrased for speech, created_at,
an optional person_id, and a meta dictionary. The queue holds 64 events.
Posting to a full queue drops the event and returns false rather than
blocking, on the reasoning that a lost nudge is better than a wedged timer
thread. The main loop drains the bus only at safe moments, meaning the idle
wake wait and turn boundaries, so that an announcement can never talk over a
reply in progress. One meta key is load bearing: open_conversation, which
tells the loop that the announcement expects an answer and that it should
begin listening rather than returning to idle. That one flag is what turns a
proactive greeting into a conversation opener.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4  ·  BEHAVIOUR AND LOGIC
// ═══════════════════════════════════════════════════════════════════════════
section4: [
  H('Behaviour and Logic'),

  `This section describes what the subsystem actually decides, and on what
evidence. Two habits run through all of it and are worth naming before the
detail. The first habit is that every judgement the robot makes has an
explicit numeric threshold that lives in config.yaml rather than in the
code, so that behaviour can be tuned in the field without a deployment. The
second habit is that a faculty which cannot reach a confident answer degrades
to a weaker one rather than failing: an unrecognised voice still gets a
reply, it just does not get to write into anyone's memory; a scene that
cannot be re-detected still publishes its last detections rather than
reporting an empty room. Almost every rule below is shaped by one of those
two habits.`,

  SUB('I', 'The State Machine'),

  `Four states exist: IDLE, LISTENING, THINKING, and SPEAKING. The legal
moves between them are declared as data in zero/state.py and checked on
every transition. IDLE may only go to LISTENING. LISTENING may go to
THINKING, or back to IDLE when nothing was said. THINKING may go to
SPEAKING, to IDLE, or back to LISTENING, that last case covering an empty
reply where the robot stays in the conversation rather than ending it.
SPEAKING may go to IDLE or back to LISTENING, the latter being barge in.`,

  `An illegal transition is logged as a warning and then performed anyway.
That is a deliberate choice: a wiring bug should be loud in the logs during
development, but it must never crash a conversation in front of a person.
The transition helper also returns early when the destination equals the
current state, which is what allows the filler and the reply to both request
SPEAKING without churn.`,

  `The most important structural fact about the loop is not visible in the
state list. After the wake word admits the first turn, the cycle is
LISTENING to THINKING to SPEAKING and back to LISTENING, repeatedly, with no
wake word between turns. IDLE is left behind until a stop phrase or a
silence long enough to trip conversation.sleep_timeout_ms returns the robot
to it. A reader who models this as a four state ring will get the behaviour
wrong, because the ring is only the initial arc; the steady state is a
tight three state loop that lives inside a conversation.`,

  SUB('II', 'Waking'),

  `Wake detection runs openWakeWord over every 30 ms frame in IDLE,
comparing its score against wake.threshold. The detector is reset and the
microphone drained whenever the robot enters or re-enters the wait, which
matters because the announcement path can speak while IDLE and the system
must not hear its own voice as a wake attempt. A conversation can also be
opened without a wake word at all. When a proactive event carries the
open_conversation flag in its metadata, the idle wait returns as though
woken, and the spoken opener is seeded into the fresh history as the first
assistant turn so that the model knows it said it. This is what allows the
robot to greet someone walking in and then listen for an answer, rather than
speaking into the air and going back to sleep.`,

  SUB('III', 'Deciding When a Person Has Finished'),

  `Endpointing is the single most behaviour defining piece of logic in the
system, because getting it wrong is immediately obvious to a user: too eager
and it interrupts, too patient and it feels unresponsive. Two backends
exist, and the distinction between the code's default and the shipped
configuration matters. The factory defaults to Silero VAD, but config.yaml
explicitly selects webrtc with an aggressiveness of 2, and that is what runs
on the deployed robot. The reason is recorded in the config itself: Silero
is sharper on clean audio but stricter, and it was missing speech from a
quiet microphone. Anyone reading the factory in isolation will conclude
Silero is active; it is not.`,

  `Silero remains available and is worth understanding because it is the
fallback plan for a noisy environment. It runs through ONNX Runtime rather
than torch, which is what keeps the Pi torch free. Silero v5 requires
exactly 512 sample windows at 16 kHz while the audio pipeline delivers 480
sample frames, so the endpointer buffers samples and feeds whole 512 sample
windows in order, carrying the model's recurrent state between them. Its
ONNX interface is smoke tested at construction with a zero filled window,
so that a version or API mismatch raises there and the factory falls back to
webrtcvad, rather than producing an endpointer that silently never detects
speech. That failure mode, a microphone that appears alive but is
functionally deaf, is the reason the smoke test exists. If Silero is
selected, vad.silero_threshold ships at 0.3 rather than the library default
of 0.5, again to accommodate a quiet microphone.`,

  `Starting and continuing an utterance use deliberately different rules.
To start, a frame must pass the VAD and exceed vad.energy_threshold in RMS,
which rejects quiet background onsets and distant voices. To continue, only
the VAD is consulted. That asymmetry is the fix for a real fragmentation
bug: quiet syllables and short pauses inside a sentence were failing the
energy test and splitting one utterance into several. A short pre-roll of
vad.speech_pad_ms is retained and prepended when speech starts, so that the
first word is not clipped.`,

  `The endpoint fires after vad.silence_ms of trailing silence, expressed
internally as a count of blocks. Two modifiers sit on top. The first is an
adaptive endpoint: if the amount of speech collected so far is less than
vad.min_speech_for_fast_end_ms, the required silence is doubled. A slow
starter who says "um" and then thinks is not cut off, while a finished
sentence still commits at the normal speed. The second is the semantic hold
described below. Two outcomes are also carefully distinguished. A true idle
timeout, meaning no speech at all for conversation.sleep_timeout_ms, returns
nothing and ends the conversation. An utterance that was captured but whose
average RMS falls below vad.min_utterance_rms is dropped and the endpointer
keeps listening with a fresh idle window. This is a proximity gate: the
owner's voice close to the microphone is loud, a conversation across the
room is not, and a stray background blip must not put the robot to sleep
mid conversation. There is also a hard length cap at vad.max_utterance_ms.`,

  `While waiting for speech, the endpointer logs a heartbeat every five
seconds. That line is load bearing for triage and should not be removed: if
it ticks but speech is never captured, the problem is VAD or level; if it
never ticks at all, frames are not arriving and the microphone stream itself
has died. Those are different faults with different fixes, and this is what
tells them apart.`,

  SUB('IV', 'Speculation and the Semantic Hold'),

  `Roughly 180 ms into any silence run, well before the endpoint would
fire, the endpointer calls back with the audio captured so far and
transcription begins on a background thread. This speculative transcript
does two jobs at once. Its first job is latency. If that pause turns out to
be the real end of the utterance, the transcript is already in hand and the
round trip to the GPU has been spent inside the silence wait rather than
added after it. The main loop decides whether it may reuse the speculative
result by two tests: the speculative audio must be a verbatim prefix of the
final utterance, and the amount of audio that arrived afterwards must be
within a slack of twice vad.silence_ms plus vad.speech_pad_ms plus 400 ms. A
longer tail means speech resumed or the length cap fired, so the
speculative text is only a prefix and the full transcription runs instead.`,

  `Its second job is judgement. When the endpoint condition is met, the
loop is asked whether the speaker was mid thought, and the answer is derived
from the last word of the speculative transcript. A transcript ending in a
conjunction, a preposition, an article, or a filler is treated as
unfinished, as is one ending in a comma, a colon, a dash, or a literal
ellipsis, which is what Whisper writes when speech trails off. The word list
deliberately excludes words that legitimately end sentences, such as
pronouns.`,

  `The critical detail is that this check is tri state rather than boolean.
True means hold for one more silence window. False means commit now. None
means the transcript is still in flight, and in that case the endpointer
waits a bounded vad.semantic_hold_wait_ms rather than committing. Without
that third state the mid thought check raced the STT round trip and lost:
the answer arrived just after the endpoint had already committed, and half
sentences were being shipped to the model. The hold applies at most once per
pause; if speech resumes, the hold is cleared and the next pause gets a
fresh decision. Speculation is skipped in three cases, each for a distinct
reason. It is skipped when voice ID is enabled, because nothing may be
transcribed before the owner check runs. It is skipped in strict privacy
mode for the same reason applied to bystanders. And it is skipped while STT
is degraded to the local CPU engine, because speculating there simply runs a
slow job twice, once at the pause and once at the endpoint, with no overlap
to gain.`,

  FIG(9, 'here'),

  SUB('V', 'Deciding Who Is Speaking'),

  `Identity fuses two independent signals. When a face match and a voice
match name the same person, the score is a weighted sum, w_face times the
face cosine plus w_voice times the voice cosine, with the weights normalised
to sum to one, and the result must clear identity.fusion.threshold, which
ships at 0.50. When the two signals name different people, they do not
average. Instead the stronger weighted signal competes alone against its own
single channel threshold, 0.42 for face and 0.45 for voice, both stricter
than the fused threshold because a single channel is inherently less
reliable. This is what prevents two weak and contradictory matches from
manufacturing a confident identification.`,

  `Sessions are owned by voice, not by face. Under
identity.session.voice_only, which defaults true, the speaker's voice
decides whose memories the turn belongs to, and the face is treated as
perception only. There is a second, stricter gate on top: a turn only
credits durable memory when the identity score reaches
identity.session.write_min_score, which ships at 0.55. A borderline match
still gets a normal conversation, it simply does not write into anyone's
permanent record. That single threshold is what stops a multi speaker
session cross contaminating memory.`,

  `An unfamiliar voice is not immediately made into a guest. The voiceprint
is held, and only after the transcript proves the turn was real is a
provisional guest assigned. Three gates decide "real": at least
identity.guests.min_words words, at least identity.guests.min_ms of audio,
and at least identity.guests.min_rms in level. These exist because Whisper
hallucinates plausible text from near silence, producing phantom guests and
polluting the training corpus. Guest identifiers are negative, and
clustering an unfamiliar voice against existing guests keeps different
strangers separate. Diarisation compares consecutive turns' voice embeddings
and raises a speaker change note when the cosine falls below
perception.diarize.change_threshold. The note is ephemeral and is attached
to that turn only.`,

  `One rule about sight deserves emphasis, because it is a behaviour users
notice. The robot may only claim to see someone when their face is in the
current frame. Recognising a voice off camera produces a different note,
explicitly saying it recognises the voice but cannot see the face.
Furthermore, the identity note is attached only when recognition changes, or
when the user asks a visual question. Repeating it every turn fed the model
greeting fodder, and it kept re-greeting people mid topic.`,

  FIG(10, 'here'),

  SUB('VI', 'Deciding What Is Being Asked About'),

  `Most turns carry only a cheap text hint about the scene. A turn
classified as visual additionally pulls recent keyframes so that the
multimodal model can actually look. The classifier is deliberately tight in
two directions. Bare demonstratives such as "this", "that", and "there" are
excluded, because they appear in most ordinary sentences. Polysemous verbs
such as "see", "look", "watch", and "picture" are excluded as bare words,
because in speech they are usually non visual: "I see", "looking forward to
it", "picture this scenario". Those words appear instead inside the phrase
list in their genuinely visual forms, such as "what do you see", "take a
look", and "look around".`,

  `Beyond the fixed word and phrase lists, the classifier consults the live
detections. If the utterance names an object the camera can see right now,
the turn is visual. This is what makes "what colour is the cup" work
without anyone having to enumerate every possible object: matching is done
against the full label and its head noun, so that "cell phone" also matches
"phone", plus a naive plural, on word boundaries. The label "person" is
excluded from this test, because a person is on screen almost always and
the word turns up constantly in abstract speech.`,

  `Presence is answered from the detector rather than from the model.
Whether a person is in frame is decided by YOLO and stated to the model as
fact, because without that grounding the model happily answers "yes, I see
you" to an empty room. Three cases are distinguished: the camera never came
up, in which case the model is told it is blind and instructed not to invent
a scene; the camera is working and there is nobody in frame; and there is a
person in frame. Spontaneous scene changes are surfaced as an optional note
the model may mention or ignore, but never on a turn classified as a
question. Answering "which planet did Thanos visit" with "is that a remote
on the table?" reads as not listening. Ungated, the changes stay queued and
self expire until a calmer turn.`,

  SUB('VII', 'Deciding What to Remember and What to Recall'),

  `Durable facts are injected once at conversation start rather than per
turn, which keeps the prompt prefix stable and the model's cache warm. Per
turn relevance recall is separate and ephemeral. Recall ranks candidate
memories by an activation score with four factors: activation equals
relevance times (0.2 + 0.8 times recency) times (importance divided by 10)
times frequency. Relevance is the cosine between the query embedding and the
stored embedding when both exist and their dimensions agree, floored at
zero; it falls back to keyword overlap when there is no embedder, and to
1.0 when there is no query at all, which reduces the ranking to recency
times importance. Recency decays exponentially from last access rather than
creation, so recalling something refreshes it.`,

  `The decay parameter deserves a correction that anyone tuning it needs.
It is named memory.retrieval.half_life_days and ships at 14, but the
implementation computes exp(-age / half_life_seconds), which makes it a
time constant, not a half life. At an age equal to the configured value the
recency factor has fallen to 1/e, roughly 0.368, rather than to 0.5. The
true half life is the configured value multiplied by the natural logarithm
of 2, so the shipped setting of 14 days is really a half life of about 9.7
days. The name is misleading rather than the behaviour being wrong, but a
reader who sets this expecting half life semantics will get a decay about
30 percent faster than intended. This is recorded as an open item in
Section 14.`,

  `The 0.2 plus 0.8 times recency term puts a floor under the decay so that
an old but highly relevant and important fact can still surface. Frequency
is 1 + 0.1 times ln(1 + access_count), a deliberately gentle logarithm so
that frequently accessed memories are favoured slightly without dominating.
Two constraints sit around it. Recall runs on its own thread under a hard
budget of memory.retrieval.budget_ms, defaulting to 300 ms; if it does not
return in time the turn proceeds without the note rather than becoming
slower, because a slow embedder in the reply path had been showing up as
multi second first token lag. And any hit whose text already appears in the
durable block injected at conversation start is dropped, so that the same
fact is never spent twice in one prompt.`,

  `In session compaction keeps a long conversation bounded. History is
allowed to grow to llm.history_trim_at turns and only then trimmed back to
llm.history_turns, which ships as 12 and 6. This asymmetry is a cache
optimisation: appending is cheap because the model reprocesses only the new
messages, whereas dropping the oldest message shifts the prefix and forces
a full re-read, so that the expensive operation is made rare rather than
constant. Trimmed turns are not discarded; they are held pending and folded
into a rolling summary by a background thread, and the trim window is
aligned to a user turn because a history opening with a dangling assistant
message reads as replying to nothing. The summary installer carries a stale
apply guard: if the conversation was reset while the summary was being
computed, the covered turns no longer match and the summary is dropped, so
that a dead session can never ghost into a fresh one.`,

  FIG(11, 'here'),

  SUB('VIII', 'Generating and Speaking'),

  `Generation starts on a background thread before the robot has decided
whether to play a filler, so that model prefill overlaps the filler rather
than following it. Handing back a bare generator would not do: a generator
does not begin work until first consumed, so the worker thread is what makes
prefill actually start.`,

  `The filler races the reply. A filler is chosen with probability
conversation.filler_probability and matched to what was said: an utterance
ending in a question mark or opening with a question word gets "good
question, let me think"; a reply of two words or fewer gets a short
acknowledgement; everything else gets a neutral line. It is then played
only if no reply audio has arrived within conversation.filler_grace_ms. A
fast answer is therefore never delayed by a canned "let me think". Fillers
are pre-synthesised at startup, and that pre-synthesis aborts after two
consecutive failures rather than hammering a dead TTS at thirty seconds per
call.`,

  `The reply is spoken sentence by sentence. A producer thread splits the
streaming text into sentences and pushes each sentence's audio pieces onto
a bounded queue, while a single gapless output stream plays them as they
arrive, so there are no inter sentence pauses. Anything the model writes in
parentheses is stripped before synthesis and never enters history, because
those are hallucinated stage directions rather than speech.`,

  `Two failure modes in this path were fixed and both are worth recording.
The producer's completion sentinel must be delivered with a blocking put
rather than a non blocking one: with put_nowait the sentinel was dropped
whenever the queue was full, which happens as soon as synthesis outpaces
playback, and a dropped sentinel left the consumer blocked forever,
freezing the conversation in SPEAKING so the robot went deaf after its first
reply. Symmetrically, the producer's own queue writes must time out and re
check the stop flag, because on a barge in the consumer stops draining and
a plain blocking put would wedge the producer thread.`,

  `Barge in runs for the whole time the robot is making sound, covering the
filler as well as the reply, and triggers on either the wake word or
sustained user speech over the reply. Speech based interruption needs no
wake word and is echo aware, learning the ambient level for
conversation.barge_in_learn_ms and requiring conversation.barge_in_speech_ms
of speech at conversation.barge_in_ratio above it. The interrupting words
are captured and fed into the next turn's transcription so that the person
never has to repeat themselves, but only the trigger window plus a short
lead in is kept, because a longer ring buffer's prefix is the robot's own
reply echo, and it garbled the following turn.`,

  `After an interruption only what was actually spoken enters history,
sliced by the index of the last sentence whose audio reached the speaker.
The model must not "remember" saying sentences the person never heard. The
language model stream is also stopped and its HTTP connection closed, so
that the GPU stops generating a reply nobody is listening to. One ordering
rule in the barge in shutdown is subtle enough to note. The monitor thread
is joined while the microphone is still live, and only then is the
microphone paused. Pausing first would block the monitor forever inside the
frame iterator, and it would then wake up and steal the next turn's audio
off the shared queue, which is precisely how the robot used to go deaf
after its first reply.`,

  SUB('IX', 'Command Paths That Bypass the Model'),

  `Several utterances are handled before the model is ever consulted, each
with its own parser, and each replying with a fixed line. Stop phrases end
the conversation. Matching is on word boundaries and only in utterances of
five words or fewer, so that a sentence that merely mentions one, such as
asking about a film called Goodbye Lenin, does not put the robot to sleep.`,

  `Enrolment has two entry points, an explicit command such as "remember
my face" and a plain introduction such as "I'm David", and both run the
same guided multi angle capture. Object teaching is distinguished from
person introduction by the article: "this is a french press" teaches an
object, "this is Peter" does not. Behavioural corrections such as "speak
slower" or "keep it short" are stored as standing preferences and, where an
engine knob exists, applied immediately. Erasure distinguishes forgetting
the last item from forgetting a person entirely, and the latter clears the
face and voice registrations as well as the stored facts.`,

  SUB('X', 'Deciding When to Speak Unprompted'),

  `The hard part of proactivity is staying quiet, and every proactive
utterance must pass every gate. A person must be present, and an
unrecognised person only counts when proactive.engage_unknown is set. Quiet
hours must not be active. The per kind cooldown must have elapsed. The
person must not already have been greeted during this arrival, where an
arrival ends after proactive.presence_reset_s. And a global cap of
proactive.max_per_hour utterances must not be exceeded. Timers and
reminders bypass the presence gates, because the user explicitly asked for
those, but they still respect quiet hours by being deferred rather than
dropped.`,

  `On top of the fixed cooldowns sits a bandit style adaptation. Each
proactive kind keeps an exponential moving average of how its utterances
land, scored in the range minus one to one, and that average scales the
kind's base cooldown. An average of zero or unknown leaves the cooldown
unchanged. A positive average shortens it, down to half at plus one. A
negative average lengthens it, up to three times at minus one. A kind of
nudge that keeps falling flat therefore backs off by itself.`,

  SUB('XI', 'Scoring Outcomes and Surprise'),

  `Every exchange becomes a reward tagged episode, with the reward
assembled from three signals that already exist and cost nothing extra.
Affect contributes the speaker's valence and confidence while speaking, on
the reasoning that tone is itself feedback. Interaction contributes a
negative for a barge in, since being cut off is a verdict, and a mild
positive for the conversation continuing within 90 seconds. The strongest
signal is an explicit verdict in the opening of the next utterance, matched
by two deliberately narrow regular expressions: generic negativity such as
"this weather is awful" is affect's job, not a judgement on the robot.`,

  `Because a verdict usually arrives one utterance after the reply it
judges, tagging is retrospective: the next utterance writes its reward back
onto the episode it actually judges. A pending proactive utterance is
resolved the same way, with any reply at all counting as a partial success
and an explicit verdict overriding. Surprise is scored separately and
drives attention rather than reward. The world state keeps online, Laplace
smoothed counts of each label and event kind, and scores each new event by
its rarity in bits, which is the negative base two logarithm of its
smoothed historical share. First ever events score highest, and daily
routine decays toward zero. Two thresholds consume that score: events above
world.surprise.remember_bits become scene episodes for later consolidation,
and events above world.surprise.narrate_bits wake the Tier 2 narrator. The
statistics persist across runs as a JSON sidecar, so that the robot's
sense of what is normal accumulates over its lifetime rather than resetting
at every boot.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5  ·  BUILD AND DEPLOY
// ═══════════════════════════════════════════════════════════════════════════
section5: [
  H('Build and Deploy'),

  `The build is deliberately unglamorous. A working ZERO on the AF-1 head
requires a Raspberry Pi 5 with an SD card holding Raspberry Pi OS Lite
(bookworm, 64 bit), a USB microphone that PortAudio can open at 16 kHz mono,
a speaker that PipeWire can route to, an optional USB camera, and network
reach to a GPU node where the model servers run. Nothing is unusual about
any of those pieces individually. The engineering work is in the order they
have to be installed, and in the small number of quirks that reliably
consume an afternoon when they are not written down.`,

  SUB('I', 'The Pi Head'),

  `The head build starts from a clean Raspberry Pi OS image. The system
packages required are documented in scripts/setup_pi.sh and are limited to
what PortAudio, PipeWire, and OpenCV need to open the audio and camera
devices reliably. The script installs the pipewire-alsa bridge, which is
what allows both the microphone (opened as pipewire) and the speaker
(routed through PipeWire to a Bluetooth or 3.5 mm sink) to coexist without
fighting for the same ALSA card. The script does not install torch, and it
does not install any large model dependency, because the Pi is a frontend
and everything that would consume its RAM lives on the far side of the
tunnel. The one exception is the local fallback stack, which is installed
optionally by a separate step so that the fallback is available if the
tunnel drops without loading whisper.cpp and Piper into memory in the
common case.`,

  `The Python environment is a plain virtual environment under the repo
root. The zero package is installed in editable mode so that a git pull is
the whole update procedure. Runtime dependencies are pinned in
requirements.txt, and optional vision requirements (opencv-python, the
ONNX runtime bindings for YOLO and ArcFace and ECAPA) are pinned in
requirements-vision.txt. The split matters on an 8 GB machine: a Pi with
no camera does not need to import opencv, and the audio-only path stays
importable in isolation, which makes it easy to bring up a voice only
robot on a smaller board or to bench test the language stack without a
camera present. The ONNX Runtime is the CPU build; there is no ORT CUDA
package installed on the Pi, because the GPU node is where CUDA lives.`,

  `Model artefacts are acquired by a small download step, not baked into
the repository. The wake model, the Silero VAD ONNX, the whisper.cpp
weights, the YOLO detector, the ArcFace and ECAPA models, and the Piper
voice each have a documented URL and a checksum. The scripts pull them
into a models/ tree under the repo root, and none of them exceeds a few
hundred megabytes on their own; the sum is well under two gigabytes even
with every fallback populated. Where the upstream ships in a format ONNX
Runtime cannot load directly, a small conversion step in
scripts/export_yolo_onnx.py or the equivalent produces the runtime
compatible file, and writes a sibling names JSON that the loader
auto detects. A missing model file does not crash the process; the factory
returns None for that faculty, and the loop runs without it.`,

  `Local device selection is the one calibration step the operator has to
do in person. The USB microphone that arrives from the factory does not
know which ALSA card it will land on, and the correct
audio.input_device value is whichever PortAudio index actually resolves
to the intended microphone on that Pi. A short interactive helper prints
the enumerated devices with their index, name, and default sample rate,
and the operator writes the chosen index into config.local.yaml. That
file is merged over config.yaml at startup so that per machine values
never diverge from the canonical config in the repository. The same file
carries the camera index, any input gain adjustment, and the GPU_HOST
that pi_tunnel.sh will connect to.`,

  SUB('II', 'The GPU Node'),

  `The GPU node build is a set of independent server installations that
share a card but not much else. Faster-Whisper large-v3-turbo, Orpheus's
vLLM server, the vision FastAPI application, Ollama with the gemma4
and nomic-embed-text models pulled, and optionally a self hosted
SearXNG container each install standalone from their upstream projects.
None of them depend on each other, and each of them can be started,
stopped, and rebuilt independently. The shared concern is memory
pressure: the card holds several models between them, and loading them
naively will evict one at random every time a call arrives. Two workarounds reduce that pressure. The KV cache is quantised
to q8_0 by the run_gpu_servers.sh script, which roughly halves the KV
footprint of the chat model, and the embedding model is a small (~275 MB)
dedicated model kept pinned with keep_alive=-1 so that recall never load
churns against the chat model. The card can handle it, but only with
those two settings on.`,

  `The launcher script is idempotent. It reads a list of expected
ports, checks whether each is already listening, and skips anything
that is up. Anything that is missing is started under setsid nohup so
that the launching shell can be closed without killing the servers. That
is the mode used during development. In production the same servers are
managed by systemd units carried in scripts/systemd/, so a restart of
the node comes up with everything running without an operator on the
box. Model artefacts on the GPU side live in a shared models directory
per server; nothing is packaged with the repository, and every server
has a documented pull step for its weights, with checksums where the
upstream provides them.`,

  SUB('III', 'The Tunnel and Systemd'),

  `The tunnel is opened by scripts/pi_tunnel.sh, which sets
StrictHostKeyChecking=accept-new so that a new host key is trusted on
first sight rather than hanging on a prompt (a systemd service has no one
to answer a prompt), enables ServerAlive keepalives at 15 seconds so that
a stalled tunnel drops within roughly 45 seconds and autossh rebuilds
it, and uses ExitOnForwardFailure so that a taken local port is a fast
fail rather than a partial forward. The full forward set is 11435 to
11434 for Ollama, 9000 to 9000 for Whisper, 9100 to 9100 for Orpheus,
8000 to 8000 for the vision stack, and 8080 to 8080 for SearXNG. In
production the tunnel runs as its own systemd unit,
zero-tunnel.service, so that a reboot brings it back automatically.`,

  `The ZERO process itself runs as zero.service. The unit is configured
to restart on failure with a short backoff, sends stdout and stderr to
journalctl (which is where every listening heartbeat and every
utterance summary lands, and which is where triage in Section 12 looks
first), and stops with SIGINT so that the loop can hand off to the
memory save thread before exit rather than being killed mid write.
The optional camera preview, controlled by the vision.preview block,
runs inside the same ZERO process on vision.preview_port (defaulting
to 8008) and is bound to vision.preview_host (defaulting to
127.0.0.1) unless the operator has deliberately opened it.`,

  SUB('IV', 'First Run Verification'),

  `The intended first run sequence is worth writing out because it is
the fastest way to catch a wiring mistake before it is buried under two
other layers of it. On the GPU node, bash scripts/run_gpu_servers.sh and
then curl each health endpoint; a JSON body with ok:true from each of
9000, 9100, 8000, and a tags list from 11434 tells you the node is
alive. On the Pi, bash scripts/pi_tunnel.sh and curl the same endpoints
via 127.0.0.1; success means the tunnel is up on all five ports. Then
systemctl start zero, curl 127.0.0.1:8090/health until ready flips true
(usually within ten seconds), and say the wake word. The first
"listening..." heartbeat in the log confirms audio is arriving. The
first "utterance: rms=" line confirms the VAD is engaging. The first
"endpoint: committed" confirms endpointing is deciding. Every following
diagnostic in Section 11 uses these same lines.`,

  FIG(12, 'here'),
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6  ·  TESTING AND VALIDATION
// ═══════════════════════════════════════════════════════════════════════════
section6: [
  H('Testing and Validation'),

  `Testing an interactive voice system honestly is harder than testing a
web backend, because the answer to "does this work" is not primarily a
matter of correct output for a given input. It is a matter of latency
budgets, of graceful degradation, and of the small courtesies of pacing,
turn taking, and self correction that a person notices immediately in
their absence. The test matrix therefore has three layers. The lowest
layer is unit and integration tests in the tests/ tree, which pin the
non negotiable contracts and the fixed bugs so that they cannot regress.
The middle layer is a small set of harnesses that exercise faculties end
to end against captured audio, video, and log fixtures. The top layer is
manual acceptance procedures for the behaviours no automated test can
credibly evaluate.`,

  SUB('I', 'Automated Coverage'),

  `The tests/ tree is small and focused. It covers the state machine's
legal transitions and its refusal to crash on illegal ones, the
Conversation history trim and compaction logic (append cheap, trim rare,
never dangle an assistant message), the recall activation ranking
including the recency floor and the frequency dampener, the endpointer's
tri state semantic hold in each of its three arms, the identity fuser
including the disagreement branch, the guest gates that stop Whisper
hallucinations minting phantom guests, the event bus overflow behaviour
that drops rather than blocks, the reward clamp, the surprise scorer's
Laplace smoothing, the visual question classifier's exclusions of bare
demonstratives and polysemous verbs, the stop phrase length gate, and
the audio pipeline's resampling behaviour when a device rejects 16 kHz.
Each test names the exact bug or contract it protects, so that a
failure points at what would break if the code were changed the wrong
way.`,

  `Integration is done by driving the process from the outside through
its actual doors. A test client posts to POST /zero/turn_text with a
canned utterance, awaits {ok:true, reply}, and asserts on the reply's
shape rather than its literal text (because the model's phrasing will
vary between runs). The same client can post a captured WAV to
POST /zero/turn and assert both that heard matches the ground truth
within a small edit distance and that reply is non empty. Those tests
run against a running instance in CI mode, where the language model is
pointed at a small local model rather than the production GPU so that
the tests do not depend on the shared card being available.`,

  SUB('II', 'Bench Harnesses'),

  `The interesting numbers are produced by bench harnesses that live
under scripts/. The detector harness (scripts/bench_detect.py) runs
held out frames through the detector and reports the labels and per
frame time, and it is how the perception team keeps the class list
honest as the scene vocabulary grows. The world benchmark
(scripts/bench_world.py) exercises the whole perception pipeline
against a captured session. The wake word scoring script
(scripts/score_wake.py) reads the near miss log and helps tune
wake.threshold from real accepts and near accepts recorded in the
target room. The audio device helper (scripts/list_audio_devices.py)
prints the PortAudio device index table so the operator can write the
right audio.input_device into config.local.yaml.`,

  FIG(13, 'here'),

  SUB('III', 'Acceptance Criteria'),

  `The subsystem is judged on the same set of properties every release,
and the numeric bars are set from an ordinary living room with the
shipped microphone rather than from a lab. Wake detection is judged on
false accepts per hour of quiet room time and false rejects on
deliberate wake attempts. Endpoint commit latency is measured from the
last speech frame to the endpoint fire, and the shipped
vad.silence_ms of 450 milliseconds is the baseline before the
adaptive doubling on short utterances is factored in. Transcription is
judged as word error rate on the confirmed final transcript against
the internal evaluation set, in English and in English/Swahili code
switched turns separately. The speculative transcript's reuse rate, its
verbatim prefix match against the final utterance, is tracked as a
second signal because it is what determines whether speculation is
actually shortening turn latency in practice. First token latency and
first audio latency are measured from endpoint commit and are the
numbers that most directly determine how the robot feels; the specific
targets are held in the internal benchmarks rather than repeated here
because they move as models and hardware change. Detection is judged by
mean average precision on the internal evaluation frames. Identity
precision is measured at the write score threshold on a held out multi
speaker recording, and the design bias is toward precision over recall:
a borderline turn simply not writing is preferable to a higher recall
that risks writing one person's fact into another person's record.`,

  SUB('IV', 'Manual Procedures'),

  `Two behaviours have no honest automated proxy, and both are checked
manually before every release. The first is barge in in its full form:
say the wake word, ask a question that will produce a long reply, and
then interrupt it mid sentence with a follow up question. Acceptance is
that the follow up is transcribed correctly, the model's next reply
addresses it (rather than resuming the interrupted answer), and only the
sentences the person actually heard end up in the conversation history.
The second is proactive quiet, which is the property that the robot
never speaks when it should not. A person enters the room and stays for
five minutes; the robot may greet on arrival (once) and may ask up to
one curiosity question during the linger, but it must not chatter, and
it must respect quiet hours and the per hour cap. Both procedures are
run against a fresh reboot rather than a warm process, because most
proactive bugs hide behind cached state.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7  ·  FAILURE MODES AND RECOVERY
// ═══════════════════════════════════════════════════════════════════════════
section7: [
  H('Failure Modes and Recovery'),

  `The philosophy of failure handling in this subsystem is that the loop
must not die and the robot must not lie about what is wrong. Everything
that follows is a specific application of those two rules. The loop
survives by catching the exceptions that would otherwise unwind through
it, and the robot stays honest by exposing degradation flags that the
persona prompt sees, so that "why are you slow" produces "my fast
hearing is down, I am transcribing locally" rather than an evasive
apology.`,

  FIG(14, 'here'),

  SUB('I', 'Tunnel Loss'),

  `The most consequential failure the subsystem has to handle is the
loss of the SSH tunnel, because it takes out five faculties at once:
Whisper, Orpheus, the vision server, Ollama, and SearXNG. autossh
detects the stall within roughly 45 seconds and rebuilds the tunnel,
but during those seconds the fallback wrappers must keep working. Each
of the six FallbackXxx classes catches the RuntimeError raised by its
remote client, sets its degraded flag, lazily builds its local
implementation if not already built, and dispatches to it for the
remaining calls that turn. The next turn tries the remote again, and if
that succeeds the flag clears and the local implementation stays in
memory ready for the next drop. The pattern is per call rather than per
session: a tunnel that flickers back within one turn does not force a
whole session to run on backup voices.`,

  `The self state narration described in Section 2.IV is what makes this
visible. On the transition into degraded, a short note is appended to
the persona prompt telling the model, for the STT case: "your fast
hearing is offline, you are transcribing on the Pi itself, that is why
you feel slower". On the transition out, a mirror note reports that the
fast path is back. The model reads these as facts about itself and can
say so when asked. The alternative, silent degradation, produces a
robot that seems arbitrarily worse, and a user who does not trust it.`,

  SUB('II', 'Camera Absent'),

  `A camera that fails to open at startup is detected by the factory
returning None for vision.enabled. The main loop reads that None and
never asks a vision question again for the life of the process. The
persona prompt is seeded with a note that the robot is blind for this
run, and the visual question classifier gates every downstream check on
vision availability. There is a small trap here: the presence answer,
"is there a person in front of you", must return an unambiguous "the
camera never came up" rather than the more general "no one is in
frame", because the model treats those two situations differently. The
distinction lives in the presence check itself and is one of the more
carefully tested behaviours in the loop.`,

  SUB('III', 'GPU Server Slow or Cold'),

  `A slow embedder is the most common performance regression and the
most misdiagnosed. The symptom is a first token latency in the low
seconds when the healthy target is under a second. The cause is
usually that the embedder has been evicted from GPU memory by another
model and has to reload on first call, which stalls the chat model in
turn because they share the card. Two mitigations sit inside the
subsystem. The embed timeout is 3 seconds; on breach, the memory
subsystem auto degrades to the hash fallback embedder for the rest of
the session, so that the reply path does not keep paying that cost. And
the recall thread is joined under a hard 300 ms budget, so that a slow
embed misses this turn's recall entirely rather than dragging the reply
along with it. Both of those decisions were made after
diagnosing multi second first token lags that turned out to be entirely
downstream of the shared card, not of ZERO's own code.`,

  `A cold Ollama is a separate failure and looks like a several second
first token latency on the first turn after a restart, followed by
normal latency thereafter. This is model load time, and quantising the
KV cache is what keeps it rare rather than eliminating it. The health
endpoint of the LLM client returns a ready flag distinct from the
control plane's own ready flag, and the warmup call is what flips it
true; the control plane's ready waits on the warmup completion, which
is why the first health probe after a Pi reboot may take up to ten
seconds to answer true.`,

  SUB('IV', 'Audio Device Loss'),

  `A USB microphone unplugged mid conversation raises inside PortAudio.
The loop catches this on the capture side and, rather than trying to
recover automatically (which would require re enumerating devices, a
step that can hang), it logs a clear diagnostic and exits. Systemd
restarts the process on failure, and the fresh process re-enumerates
cleanly. This is the one place the subsystem chooses to die rather than
degrade, because a persistent process holding a stale device handle is
worse behaviour than a fresh process holding a live one.`,

  SUB('V', 'Whisper Hallucinates Text'),

  `Whisper hallucinates plausible text from near silence, producing
outputs such as "Thank you for watching" and "Obrigado" that read as
real turns to any downstream consumer that trusts the text. Those
strings were minting phantom guests before three quality gates were
added: an utterance must have at least identity.guests.min_words words,
at least identity.guests.min_ms of audio, and at least
identity.guests.min_rms in level to be eligible for guest assignment.
Each of the three catches a different failure mode (a hallucinated
short greeting, a burst of background noise picked up as a phrase, a
distant television), and together they are the reason the corpus stays
clean enough to train against. This same set of gates is what
distinguishes an "utterance dropped, still listening" line in the log
from an "utterance committed" line.`,

  SUB('VI', 'Memory Database Lock'),

  `SQLite raises OperationalError when a writer holds the database
during a write. The subsystem's response is uniform across stores:
catch the exception, log it at info level with the operation name,
skip the write for that turn, and continue. The next turn retries. The
episode store runs in WAL mode specifically so that this contention is
rarer, and the memory store is written on turn boundaries rather than
in the middle of speech, so a missed write does not degrade the
conversation itself. A locked database is never a fatal error; a
corrupted database is, and it presents as a sqlite3.DatabaseError on
open, at which point the file is renamed with a timestamp and a fresh
empty database is created. The renamed file remains on disk for later
inspection.`,

  SUB('VII', 'Control Plane Exceptions'),

  `Every handler is wrapped in a try/except that catches everything,
logs the traceback, and returns HTTP 500 with the exception text
truncated to 200 characters. The truncation is a size guard against a
long stack trace being returned to a browser; the logged traceback is
the full text. The one class of exception the control plane treats
specially is a decode failure on POST /zero/turn, which returns HTTP
422 with the tail of ffmpeg's stderr, because a caller sending an
unsupported audio container needs to see why. Everything else is 500.`,

  SUB('VIII', 'Single Instance Discipline'),

  `Two ZERO processes on the same Pi are a failure mode in themselves:
they would both open the microphone, both post events onto their own
buses, and produce a robot that answers each turn twice. A single
instance lock, held on a well known file, prevents a second process
from starting; the second exits immediately with a clear message. This
is enforced at the process level rather than at the systemd level
because ad hoc runs (an operator launching a bench version by hand) are
the common cause, not the service manager. The lock is released on
clean exit and released by the kernel on unclean exit, so a crashed
process does not leave the machine unable to start a fresh one.`,

  SUB('IX', 'What Is Deliberately Not Handled'),

  `Not every exception is caught, and the ones that are not are named
here so that the choice is deliberate rather than accidental. A
KeyboardInterrupt propagates through everything, because that is the
only way to stop the process from a terminal cleanly. A SystemExit
propagates for the same reason. An out of memory error propagates
because the process cannot honestly continue after it, and letting
systemd restart is the correct response. A bug that raises during the
factory build is fatal, because the loop cannot start; that class of
error is caught during setup, not at runtime.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8  ·  CONFIGURATION AND CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════
section8: [
  H('Configuration and Calibration'),

  `The behaviour of the subsystem is set by a single YAML file at the
repository root, config.yaml, which is deep merged with an optional
config.local.yaml at load time so that per machine overrides never
collide with the canonical values kept in version control. Every key in
config.yaml is documented in place with a short prose note explaining
what it does and, more importantly, what the failure mode of setting it
wrong is. That inline documentation is authoritative; the walk in this
section explains the shape of the file and the small number of keys that
carry the most consequence.`,

  FIG(15, 'here'),

  SUB('I', 'The Top Level Blocks'),

  `The file is organised as one top level block per faculty, and the
blocks are the ones you would expect from Section 2: audio, privacy,
control, conversation, memory, learning, perception, proactive,
preferences, tools, identity, voiceid, wake, vad, stt, llm, tts, vision,
and world. A missing block is treated as an empty block, so a minimal
config file is a valid config file, and every default is taken from the
in code default. The intent is that a fresh install works with an
untouched config.yaml, and that customisation is additive: an operator
writes only the values they need to change in config.local.yaml, and
the rest is inherited.`,

  SUB('II', 'Audio Calibration'),

  `The audio block carries the values that determine whether the robot
can hear the person in the room at all. sample_rate is fixed at 16000
and there is no supported reason to change it. block_ms is 30 by
convention and matches every downstream consumer; changing it is a
system wide edit rather than a config change. input_device names the
PortAudio device by string ("pipewire") rather than by index, so that a
USB re-enumeration does not break the pipeline. input_gain is the
multiplier applied before int16 conversion, and its calibration is done
by watching the "utterance: rms=" log lines and adjusting until normal
speech lands in the range 2000 to 6000 with peaks well under 32767.
The shipped default of 6.0 is what a Logitech BRIO produces on a
typical desk setup; a headset microphone will need much less, and a
distant boom will need more. The consequence of getting this wrong is
either clipping (peaks at 32767, degrades transcription accuracy
noticeably) or under drive (rms in the low hundreds, the wake model
fails to fire even on a shouted wake word).`,

  `output_device selects the speaker sink via PipeWire's default, which
is the mechanism that lets a Bluetooth speaker or the 3.5 mm jack be
selected without touching this file. The acoustic echo cancellation
block, aec, ships disabled because it needs the wired output path
(Bluetooth's drifting latency defeats the filter) and it depends on
speexdsp being installed. When enabled, filter_ms controls the echo tail
the filter can absorb and should be raised in a live, reflective room.`,

  SUB('III', 'Voice Endpointing'),

  `The vad block is where a robot that feels responsive or unresponsive
is actually set. engine selects webrtc or silero, and the choice depends
on the acoustic environment more than on the microphone: webrtc is
softer and catches quiet speakers, Silero is sharper and rejects noise
better. silence_ms is the trailing silence in milliseconds that ends an
utterance and is the single knob most likely to be tuned in the field;
450 is a compromise that feels natural to most speakers. semantic_hold
is true by default and turns on the mid thought protection described in
Section 4.IV; the tri state hold requires semantic_hold_wait_ms as its
bound, which ships at 600 and should not be lowered under 300 without
also verifying that the STT round trip meets that number. energy_threshold
is the RMS floor a single frame must exceed to start an utterance, and
it is kept deliberately low (150 as shipped) because the VAD is the real
speech detector; making this higher rejected real speech in field tests.
min_utterance_rms is the proximity gate that drops a whole utterance
whose average level is too low, and it too is kept low (30 as shipped)
because sad or tired people speak quietly, and dropping their words is
a human feel bug.`,

  SUB('IV', 'Identity and Memory'),

  `The identity block carries the fusion weights, the fused threshold,
and the strict per channel thresholds for the disagreement branch. The
shipped values are w_face 0.55, w_voice 0.45, fused threshold 0.50,
face 0.42, voice 0.45. The one value that requires the most care during
tuning is identity.session.write_min_score, which ships at 0.55: it is
the bar a turn must clear to write into a person's durable memory. A
lower value produces more writes but more cross contamination between
speakers; a higher value produces fewer writes but the robot forgets
recognised people. The guests sub block carries the min_words, min_ms,
and min_rms gates that stop Whisper hallucinations minting phantom
guests, and those are shipped conservative on purpose: it is better to
leave a real guest unminted than to mint a phantom one, because a
missing guest is invisible and a phantom guest pollutes recall.`,

  `The memory block carries the retrieval budget, the top_k, the decay
parameter, and the forgetting settings. half_life_days is the name of
the decay parameter and it ships at 14; as noted in Section 4.VII and
14, this is used as a time constant, not a half life, so the true half
life is about 9.7 days. budget_ms is 300 and is the hard cap on the
recall thread; if a recall is not returned in that window, the turn
proceeds without the note. This is the value that prevents a slow
embedder from dragging first token latency into the multi second range.`,

  SUB('V', 'LLM, TTS, and Vision Selection'),

  `The llm block chooses the engine, the host URL, the model, and the
history sizes. host is the SSH tunnel local port, 127.0.0.1:11435 by
default, and it is the field most often edited wrongly during
development: pointing it at 11434 breaks the tunnel offset, and
pointing it at a remote address exposes the LLM to the network. model
is the Ollama model name and defaults to gemma4:latest. history_turns
and history_trim_at are 6 and 12 by default; they encode the cache
friendly trim policy of "grow to 12, trim to 6" that keeps most turns
append only and pays the full re-read cost rarely.`,

  `The tts block chooses orpheus, piper, or fish, with a fallback engine
named for the case where the primary fails. The orpheus sub block
carries the URL, voice, and prebuffer_ms; the last of these is a jitter
buffer that pre-accumulates 300 ms of audio before playback starts,
which is what prevents a brief generation hiccup from underrunning the
speaker mid word. The piper sub block names the binary and the voice
model, and length_scale sets the base speaking rate that the
orchestrator nudges around per segment.`,

  `The vision block carries the camera geometry, the detection cadence,
the detector model path, and the GPU vision server URLs. camera.mjpg is
true by default because most USB webcams cannot sustain raw YUYV at
640x480x30 and return no frames without it; MJPG is the reliable path.
detect_interval_s is the minimum time between detection calls and is
0.1 by default; that is fine on a healthy tunnel and is the value to
raise when the GPU is saturated. gpu.enabled controls whether the
grounded distance and bearing facts are computed at all; it ships off
because the multimodal LLM reads position straight from the frames and
distances-in-metres was reading clinical rather than helpful.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9  ·  PERFORMANCE AND RESOURCE BUDGET
// ═══════════════════════════════════════════════════════════════════════════
section9: [
  H('Performance and Resource Budget'),

  `Performance in this subsystem is dominated by three things: the CPU
budget on the Pi, the KV cache budget on the shared GPU, and the round
trip latency across the tunnel. Nothing else is close. The prose below
walks each of them and then reconstructs the end to end latency budget
that the whole design was written against.`,

  FIG(16, 'here'),

  SUB('I', 'Pi 5 Resources'),

  `The AF-1 head runs on a Raspberry Pi 5 with 8 GB of RAM. The ZERO
process, in its shipped configuration with the camera on and the
identity models loaded, holds the interpreter, the audio pipeline, the
ONNX Runtime working sets for the local models (open vocabulary YOLO,
ArcFace, ECAPA, Silero), and the numpy buffers for audio and frames.
The single largest transient allocation is the audio buffer for a
maximum length utterance at vad.max_utterance_ms of 15000
milliseconds, and it is bounded by that cap. CPU usage varies by
state, with idle dominated by the wake model and motion detection at
every frame, listening dominated by the endpointer during speech,
thinking dominated by identity and diarisation running alongside
recall, and speaking dominated by playback and the TTS producer
sharing cores. The 60 percent detector duty budget in
world.budgets.tier1_max_duty is the ceiling that prevents the
perception loop starving the frame grab under a busy scene.`,

  `Power draw is not measured from software, and thermal behaviour
depends on the physical head assembly rather than on this subsystem.
An un-heatsinked Pi 5 will throttle under sustained load in a warm
room, so the head assembly carries the passive cooling documented in
the mechanical assembly section of the Engineering Corpus. Measured
power and thermal numbers, when the head assembly is instrumented for
them, belong in that document rather than in this one.`,

  SUB('II', 'GPU Resources'),

  `The GPU node's card is shared by five models: the chat model
(gemma4 8B at fp16 or its published quantisation), Whisper
large-v3-turbo, Orpheus 3B, YOLO11x, and CLIP plus the face models on
the vision server. The KV cache on the chat model is where the pressure
manifests: at num_ctx 8192 the chat model's KV cache alone can push the
card into OOM and cause it to evict whichever model was loaded most
recently. Two settings turn that from a common failure into a rare
one. num_ctx is set explicitly to 4096, which halves the KV footprint
without materially reducing the effective context (persona plus memory
plus six turns fits comfortably). OLLAMA_KV_CACHE_TYPE is set to q8_0,
which halves the KV footprint again. With both in place, the chat
model, Whisper, Orpheus, and the embedder can all stay resident.`,

  `The dedicated embedding model (nomic-embed-text, roughly 275 MB) is
pinned with keep_alive=-1 so that recall never has to load-churn it
against the chat model. Before that pinning, a slow embed reliably
forced the chat model to reload on the next turn, and the observed
symptom was a first token latency of six or more seconds instead of
under one. This was the single largest first token regression the
subsystem has fixed.`,

  SUB('III', 'End to End Latency'),

  `The healthy latency chain from a spoken end of turn to the first
audible response is the shape of the design. The endpointer detects the
pause first, and about 180 milliseconds into the silence it calls back
into the speculative STT worker so that transcription starts inside
the silence rather than after it. When the silence reaches
vad.silence_ms and the semantic hold decides to commit, the endpoint
fires. The speculative transcript is reused whenever it is a verbatim
prefix of the final utterance, and in that common case the entire STT
round trip has already been paid inside the silence. Recall runs on
its own thread under the memory.retrieval.budget_ms cap so that a
slow embedder cannot drag the reply along with it; if the budget is
missed the turn proceeds without the recall note. The LLM stream
starts as soon as the prompt is ready, and the filler, when the
probability picks it and the grace window elapses without reply audio,
covers the small gap before first tokens arrive. The TTS producer
synthesises the first sentence into the playback queue as soon as the
first sentence boundary lands in the token stream, and playback begins
on that first sentence.`,

  `Every part of this chain assumes the tunnel is healthy, the chat
model is resident, and the embed backend is responding within budget.
A tunnel drop pushes STT onto whisper.cpp on the Pi CPU; the local
engine is meaningfully slower than the GPU one and the speculative
overlap no longer helps because the local engine cannot productively
run twice inside the endpoint window. An evicted chat model adds model
reload time before the first token arrives. A slow embed misses the
recall for that turn but does not otherwise slow anything down,
because of the budget cap; the log line for over budget recall is
what makes that invisible failure visible.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 10  ·  SECURITY, PRIVACY, AND DATA HANDLING
// ═══════════════════════════════════════════════════════════════════════════
section10: [
  H('Security, Privacy, and Data Handling'),

  `The subsystem holds personal data by design: it has to, in order to
recognise a person's face and voice, remember what they told it, and
respect their preferences. This section describes what is held, where
it is held, how long it stays, and how it can be removed. It also
describes the small attack surface the LAN bound control plane
presents, and the mitigations that are in place today.`,

  FIG(17, 'here'),

  SUB('I', 'What Is Held'),

  `Four categories of personal data are held on the Pi. First, face
embeddings, produced by the ArcFace model and stored as 512
dimensional float32 vectors in zero_identity.sqlite. These are
compressed representations of a face and cannot be inverted to
reproduce the original image. Second, voice embeddings, produced by the
ECAPA model and stored the same way. Third, transcripts of what has
been said, held in memory during a session and appended per speaker
per session as NDJSON lines to data/corpus/interactions.jsonl at
session end. Fourth, derived facts and preferences (name, likes,
things asked to be remembered), stored as rows in zero_memory.sqlite
with a person_id foreign key. Guest voices, when someone speaks who is
not enrolled, are held as embeddings under a negative person_id in
zero_guests.sqlite and are subject to the same erasure verbs as
enrolled persons.`,

  SUB('II', 'The Privacy Guard'),

  `The bystander_mode key in the privacy block selects one of three
policies: open, guarded (the default), and strict. In open mode, an
unrecognised voice is answered and remembered exactly as an enrolled
voice is, which is appropriate for a single user household where every
voice is welcome. In guarded mode, an unrecognised voice is answered
but nothing they say is stored long term; the session runs, memory is
not written, and the corpus append is skipped. In strict mode, an
unrecognised voice is not engaged at all; the wake word from an
unrecognised voice is ignored, and speculation is skipped for the same
reason. Enrolled persons always receive full service regardless of the
mode.`,

  `Erasure is always honoured, regardless of mode. Two verbs are
recognised before the model is consulted: "forget that" removes the
most recent memory entry, and "forget everything about me" removes the
speaker's enrolled identity, all their embeddings, all their memory
rows, and their entry in the guests store if present. Erasure is
immediate and does not wait for a background job. The recording
indicator is the visible pair to this: the LED (or the log line, when
no pin is configured) confirms per state whether the robot is
listening, thinking, or speaking, so that a person in the room always
knows when audio is being captured.`,

  SUB('III', 'The Attack Surface'),

  `The control plane binds 0.0.0.0 with wide open CORS and no
authentication, which means anything on the local network can make the
robot speak, run a text turn, or end a session. This is a deliberate
trade for a LAN only device: the intended deployment is a trusted
home network where the AF-1 application is the only client, and adding
a token would burden every developer with a token dance during
iteration. The mitigation posture is threefold. The subsystem is
firewalled at the network level (a home router does not forward 8090
inbound). The device is not intended for use on hostile networks. And
the response bodies are strictly bounded (500 characters truncated on
say, 1200 on turn_text) so that a malicious caller cannot use the
control plane as an amplification vector. If the deployment target
changes, this decision is one of the first that would revisit; a
signed token in the persona would be a small change to the handlers
and would let the bind stay wide.`,

  `Key material for the tunnel is a standard SSH key pair generated per
Pi during setup. The public key is installed on the GPU node's
authorised_keys for the account that owns the model servers, and the
private key sits in the standard SSH directory of the account that
runs the tunnel unit. There is no automatic key rotation, and none is
planned; operator rotation is preferable to another moving part in a
deployment of this size.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 11  ·  OBSERVABILITY AND DIAGNOSTICS
// ═══════════════════════════════════════════════════════════════════════════
section11: [
  H('Observability and Diagnostics'),

  `The subsystem is observed by exactly two things: its logs and its
health endpoint. There is no telemetry pipeline and no external
metrics store. That is deliberate for the LAN only deployment, and it
places a load on the log lines themselves to be useful during triage.
This section names the lines that matter and the way a whole session
is reconstructed from them.`,

  FIG(18, 'here'),

  SUB('I', 'The Log Setup'),

  `Logs are written to stdout through the Python logging module with a
per module logger named after the faculty (get_logger("vad") for the
endpointer, get_logger("stt") for the transcription clients, and so
on). Log levels are set globally by the ZERO_LOG_LEVEL environment
variable and default to INFO, with DEBUG available for the endpointer
and the STT clients when diagnosing an issue. Every log line carries
the module name as its prefix, so that a grep on "vad:" isolates the
audio pipeline and a grep on "llm:" isolates the language model, which
is how triage is done in practice.`,

  SUB('II', 'Load Bearing Log Lines'),

  `A short set of log lines are load bearing: they exist specifically
to support triage, and removing them would break the field diagnostic
procedures in Section 12. The endpointer emits a periodic "listening"
line while it waits for speech, tagged with the module name vad; if
this line ticks but no utterance is ever captured, the problem is on
the VAD or level side, and if it does not tick at all frames are not
arriving from PortAudio. The endpointer emits an utterance summary
line with the RMS and peak level of every completed utterance, and
those values are the input to microphone gain calibration. The
endpointer also emits a "dropped: too quiet" line when the proximity
gate rejects an utterance whose average RMS falls below the shipped
minimum; that line distinguishes a dropped background blip from a
committed real turn. The fallback wrappers emit a transition line
when a faculty switches to its local fallback and a mirror line when
the remote comes back, and those are the ground truth behind the
degradation flags reported by the /zero/status endpoint.`,

  SUB('III', 'The Health Endpoint and Status'),

  `GET /health answers immediately at startup with {ok, service, state,
ready}, and it is the one endpoint that must always work; if it does
not, the process is not running or is not on the port it claims. GET
/zero/status returns the state, the last external turn, and the
current degradation flags, and it is the endpoint the AF-1 application
polls to show whether the robot is running on its backup voice.
Together they answer the two questions a supervising system asks: "is
the process alive" and "is it healthy".`,

  SUB('IV', 'Reading a Session'),

  `A whole session can be reconstructed from the log stream by
following the sequence of module tags. The wake acceptance (or a
proactive event carrying open_conversation) opens the session and
moves the state from IDLE to LISTENING. The listening heartbeat stops
and an utterance summary appears when the endpointer commits. The
speculative STT tag reports whether the pause fired a speculative call
and whether its transcript was reused as the final one. Identity and
diarisation lines appear next with the fused score. Recall reports
whether its budget was met and how many facts came back. The language
model client reports the first token and the total token count. The
TTS producer reports when the first audio chunk lands, and playback
reports the sentence boundaries as they play. At session end, a
closing line records the reason (sleep timeout, stop phrase, barge in
end, or exception) and the memory save thread reports its writes to
the durable stores.`,

  `The camera preview served on vision.preview_port (defaulting to
8008 and bound to 127.0.0.1) is the other diagnostic surface. It
shows the current frame with the detection boxes overlaid, and it is
the fastest way to confirm that the camera is producing frames and
the detector is producing labels, both of which have to be true
before any visual behaviour can work at all.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 12  ·  OPERATIONS RUNBOOK
// ═══════════════════════════════════════════════════════════════════════════
section12: [
  H('Operations Runbook'),

  `This section is written for the operator who has to bring the system
up, take it down, update it, or triage it, and who wants the exact
commands rather than a description of them. Everything below assumes a
Debian based Pi and a Debian based GPU node; the commands transfer
directly to other distributions with the obvious substitutions.`,

  FIG(19, 'here'),

  SUB('I', 'Bringing the System Up'),

  `On the GPU node, run bash scripts/run_gpu_servers.sh. It is
idempotent: it will start whatever is not running and leave whatever is
alone. Verify that /health on 9000, 9100, and 8000 all return ok:true,
and that curl 127.0.0.1:11434/api/tags lists the expected models
(gemma4:latest and nomic-embed-text at minimum). On the Pi, run
GPU_HOST=user@gpu-box bash scripts/pi_tunnel.sh (or systemctl start
zero-tunnel), and verify the tunnel with curl on 127.0.0.1:9000/health,
127.0.0.1:11435/api/tags, and 127.0.0.1:8000/health. Then systemctl
start zero, poll 127.0.0.1:8090/health until ready:true, and say the
wake word to confirm the audio path. The whole procedure takes under
thirty seconds on a warm GPU node.`,

  SUB('II', 'Taking It Down'),

  `systemctl stop zero on the Pi is the graceful stop. The main loop
receives SIGINT, drops out of its current phase safely, and the
memory-save thread writes the session to disk before the process
exits. Do not use SIGKILL unless the process is actually stuck; a
SIGKILL loses the current session's memory write. On the GPU node,
each server is stopped by its systemd unit, or by killing the process
listening on its port if run manually.`,

  SUB('III', 'Updating and Rolling Back'),

  `An update is a git pull on the Pi. Because the zero package is
installed editable, the pull is the whole update; there is no build
step and no wheel install. After pulling, systemctl restart zero and
verify with the same health check sequence as bring up. A rollback is
git checkout to the previous known good commit and another restart.
Both operations are seconds, not minutes. Model artefact updates are
manual: download the new artefact, update its path in config.local.yaml
if the filename changed, restart the process. The models are not on the
git critical path because they are large and because the model server
side of the GPU node also holds its own copies.`,

  SUB('IV', 'Triage Table'),

  `The three most common triage cases are handled below by symptom.
"Robot does not reply to the wake word" resolves in this order: check
"listening..." heartbeat in the log (if missing, the microphone stream
is dead, check audio.input_device); check the log for any wake accept
line (if none, wake.threshold may be too high or the model may be
missing); if wake fires but no endpoint commit follows, check that the
VAD is receiving frames (a fragmentation bug where the mic was renamed
by ALSA has the same symptom). "Robot answers but very slowly": check
the degraded flag in /zero/status (if true, the tunnel is down, check
autossh and the GPU node); check the log for "recall: over budget"
(if present, the embed backend is slow, verify the embed model is
resident with ollama ps); check the log for LLM reload messages (if
present, the KV cache has evicted the chat model, verify
OLLAMA_KV_CACHE_TYPE is set). "Robot goes deaf after its first reply":
this is the barge in ordering bug fixed in Section 4.VIII; if it
recurs after a code change, check that the monitor thread is joined
before the microphone pause on the shutdown path.`,

  SUB('V', 'Backup and Restore'),

  `The subsystem's durable state is small enough to back up with cp.
The files that matter are zero_memory.sqlite, zero_identity.sqlite,
zero_guests.sqlite, zero_episodes.sqlite, zero_curiosity.sqlite,
zero_objects.sqlite, voiceprint.npy, and the data/corpus/ directory.
Together they are a few tens of megabytes. Backup is a stop of the
zero service, a cp of the files (or a sqlite3 .backup to be safe on
WAL journaled files), and a start. Restore is the mirror: stop,
replace the files, start, and verify by asking the robot to recall a
fact that was known before the backup was taken.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 13  ·  DEPENDENCIES AND LICENCES
// ═══════════════════════════════════════════════════════════════════════════
section13: [
  H('Dependencies and Licences'),

  `The subsystem depends on a small tree of third party code and a
larger tree of third party model weights, and the licensing landscape
is uneven across them. This section names the important pieces, with
the versions currently in use, the licences under which they are
distributed, and a short note on any supply chain risk worth
recording.`,

  FIG(20, 'here'),

  SUB('I', 'Code Dependencies'),

  `On the Pi side, the notable dependencies are the Python interpreter
(3.11 or newer), ONNX Runtime (1.19 as tested, the CPU build), NumPy,
OpenCV, PortAudio bindings, webrtcvad, gpiozero, and the ollama
Python client. All of these are Apache 2.0, BSD, or MIT, with no
copyleft ties, and all are pinned by exact version in requirements.txt
so that a rebuild produces the same tree. On the GPU node side, the
notable dependencies are CUDA (the driver and runtime), PyTorch and
vLLM for the Orpheus server, faster-whisper for the STT server,
FastAPI for the vision application, and Ollama itself. Licences here
are more varied: PyTorch is BSD, vLLM is Apache 2.0, Ollama is MIT.`,

  SUB('II', 'Model Weights'),

  `The model weights are where the interesting supply chain notes
live. openWakeWord ships pretrained models under a permissive licence
and a custom "hey zero" model, if trained, would inherit whatever
licence the training pipeline documents. Silero VAD v5 is a permissive
release and the ONNX export in models/vad/silero_vad.onnx is a direct
export from the upstream release. Whisper large-v3-turbo is
distributed under MIT by OpenAI, and faster-whisper's wrapper does
not change that. The Orpheus 3B weights carry their own terms which
should be reviewed against the intended deployment. Piper voices are
individually licensed; the shipped en_US-amy-medium voice is under a
permissive licence, but voice sets should be verified case by case.
YOLO11n, YOLO11x, and YOLOv8-worldv2 are AGPL when redistributed as
weights; that is one of the more consequential licence facts in the
tree and is the reason a productised deployment might choose a
differently licensed detector. ArcFace and ECAPA follow their upstream
research code licences. Depth Anything V2 is permissive. gemma4 is
distributed under Google's Gemma terms of use, which permit self
hosted commercial use with the usual caveats; those terms should be
reviewed against the intended deployment.`,

  `The general shape of the supply chain risk here is that the code
tree is boring and safe, and the model tree is where attention has to
be paid. Weights are large, versioned externally, and often
distributed without a signed release; a defensive posture is to pin
weights by checksum in the acquisition scripts and to reject any
mismatch at build time. That posture is followed in the acquisition
scripts for the models where an upstream checksum is available, and
noted as an open item for those where it is not.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 14  ·  OPEN ITEMS AND DECISIONS LOG
// ═══════════════════════════════════════════════════════════════════════════
section14: [
  H('Open Items and Decisions Log'),

  `This section names the known gaps and the design decisions worth
recording, so that a future engineer reading the code does not have to
reconstruct why a particular choice was made from the diff alone.
Items are stated with their current status, so an "open" item is
something to do, and a "decided" item is something not to change
casually.`,

  FIG(21, 'here'),

  SUB('I', 'Open Items'),

  `Schema drift between zero/vision/schemas.py and
server/vision/shared/schemas.py, described in Section 3.II, is open.
The two files have diverged and the sync script referenced in comments
does not exist in the repository. The fix is either to write the
sync_schemas.py mentioned in the docstring or to move both files to a
shared package installable in both trees. The current benign state
should not be relied on beyond the next edit to either file.`,

  `The half_life_days misnomer, described in Section 4.VII and 8.IV,
is open. The parameter name promises half life semantics but the
implementation is a time constant, so tuning under the current name is
off by a factor of ln(2). The fix is to either rename the parameter
(a breaking config change) or to divide by ln(2) on read (a silent
semantic change). Neither has been done; the correction is documented
here and in the config comment.`,

  `The filler pre synthesis fragility, described in Section 4.VIII, is
open. Two consecutive failures abort pre synthesis, and there is no
metric that surfaces that state; a robot that has quietly stopped
pre synthesising fillers can still ship a reply, but its pacing on
slow first tokens becomes noticeably worse. The fix is a small
counter exposed by the status endpoint.`,

  `Control plane authentication, described in Section 10.III, is open
but low priority for the current deployment. The wide open bind is a
deliberate trade for a LAN only device; the fix, if the deployment
target ever leaves the LAN, is a signed token in the header set that
the handlers verify, and a matching change on the application side.`,

  `The vad engine default versus shipped mismatch, described in
Section 4.III, is a minor open item. The factory defaults to Silero
and config.yaml overrides to webrtc; a reader of the factory in
isolation will draw the wrong conclusion. The fix is to change the
factory default to match the shipped value.`,

  `Episode consolidation, referenced in Section 4.XI and by the
episodes block in Section 8, is scaffolded but not turned on by
default. The nightly job runs when configured and populates the
consolidated_at column, but the training pipeline that consumes those
rows is documented separately and its win over ordinary memory writes
has not been measured on this subsystem yet.`,

  SUB('II', 'Decisions Worth Recording'),

  `Sessions are owned by voice rather than by face, because voice is
what actually produces a turn's transcript, and because attributing a
turn to a face when the voice is unfamiliar is exactly the kind of
cross contamination the write score threshold exists to prevent. This
is worth naming because the natural alternative, using the most
confident channel per turn, was tried and produced multi speaker
sessions writing into the wrong person's memory.`,

  `Perception runs continuously rather than on demand, because a
person asking about the room does not want to wait while the camera
first looks, and because Tier 0 motion detection is cheap enough to
absorb without notice. The alternative, cold vision, was tried and
produced visible pauses whenever a visual question landed.`,

  `The control plane lives inside the ZERO process rather than beside
it, because a network turn and a spoken turn share every faculty (the
Conversation object, the memory database, the voice, the speaker), and
running two processes forces every one of those to be synchronised
across a boundary that has no reason to exist. This is one of the
system's larger design commitments and is unlikely to change.`,

  `Speculation is skipped while STT is degraded to the local engine,
because the local engine cannot productively run the speculative and
the final transcription in the small overlap window that the semantic
hold allows. The alternative, always speculate, was tried and produced
a slower first token during degradation than not speculating at all,
because two whisper.cpp runs at 1.5 seconds each does not fit under
the endpoint window.`,

  `Guests are minted only after a transcript proves the turn was real,
because Whisper hallucinates plausible text from near silence and the
resulting phantom guests polluted the training corpus in the early
runs. The three quality gates (min words, min ms, min rms) each catch
a different failure mode, and together they are the reason the corpus
is clean enough to train against.`,
],

// ═══════════════════════════════════════════════════════════════════════════
// APPENDICES
// ═══════════════════════════════════════════════════════════════════════════
appendices: [
  H('Appendices'),

  SUB('A', 'Glossary'),

  `IDLE, LISTENING, THINKING, SPEAKING are the four states of the
conversation loop, defined in zero/state.py. Endpointer is the module
that decides when a person has stopped speaking. VAD is voice activity
detection. Semantic hold is the tri state decision, based on the last
word of the speculative transcript, about whether to commit an
endpoint or wait one more silence window. Speculative transcript is
the STT run at 180 ms into a silence, used both to overlap the STT
round trip with the silence wait and to inform the semantic hold.
Barge in is the ability of a person to interrupt the robot mid speech
by wake word or sustained speech. Filler is a short pre synthesised
line played to cover the gap while the model is generating. WorldState
is the shared, published perception object read by every consumer that
needs to know what the room contains. Tier 0, 1, 2 are motion,
detection, and narration, respectively. Fusion is the identity
decision combining face and voice cosines. Guest is a provisional
identity minted for an unfamiliar voice, kept separate under a
negative person_id.`,

  SUB('B', 'On Disk Schema'),

  `The full SQLite schema for each of the six stores is documented in
Section 3.V, and the DDL as executed on first open is in the store
constructors under zero/memory/, zero/identity/, and
zero/learning/episodes.py. The convention that every embedding is
accompanied by a dim column, and every person_id follows the sign
convention (positive enrolled, negative guest, null anonymous), is
uniform across every store.`,

  SUB('C', 'The Complete Default config.yaml'),

  `The shipped config.yaml is the authoritative default and is
documented in place with prose notes. Rather than reproduce it here in
full, this section names each top level block and its role, with a
pointer to the section that discusses its keys in depth: audio and vad
in Section 8.II and 8.III, identity and memory in Section 8.IV, llm
and tts and vision in Section 8.V, and the perception, proactive,
world, and learning blocks in Section 2 through 4 where their
behaviour is described.`,

  SUB('D', 'Port and Device Map'),

  `The tunnel forwards five local ports to their counterparts on the
GPU node: 9000 to 9000 for Whisper, 9100 to 9100 for Orpheus, 8000 to
8000 for the vision stack, 8080 to 8080 for SearXNG, and 11435 to
11434 for Ollama. The control plane listens on 8090 on the Pi, bound
to 0.0.0.0. The camera preview listens on 8008 on the Pi, bound to
127.0.0.1 by default.`,

  SUB('E', 'Detailed Changelog'),

  `Revision 0.1 is the first public draft of this document. Prior
revisions of the codebase are documented in git history; the notable
changes referenced by this document are the addition of the tri state
semantic hold, the addition of the identity write score threshold,
the addition of the three guest quality gates, the switch of the KV
cache to q8_0 quantisation, the pinning of the embedding model with
keep_alive minus one, and the fix to the barge in shutdown ordering
that had produced the deaf after first reply symptom.`,
],

};
