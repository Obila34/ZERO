# 1 Purpose & Scope

ZERO is the conversational intelligence subsystem of the AF-1. It is the part of
the robot a person actually talks to. Everything it does resolves into one
user-visible behaviour: somebody speaks near the robot, and the robot answers out
loud in its own voice, promptly, knowing who is speaking, remembering what has
been said before, and aware of what is currently in front of its camera. Every
module described in this document exists to make that single behaviour happen and
to keep it happening when parts of the system fail.

The subsystem runs as one Python process on the Raspberry Pi 5 in the AF-1 head,
started as `python -m zero.main`. Inside that process a four-state loop turns
continuously. The robot sits in IDLE with the microphone open, listening only for
its wake word. On a wake detection it moves to LISTENING and captures the
utterance until a voice-activity endpointer decides the person has finished
speaking. It then moves to THINKING, where transcription, speaker identification,
emotional read, memory recall and language-model generation all run, several of
them in parallel. It finishes in SPEAKING, where the reply is synthesised and
played sentence by sentence, and then returns to LISTENING for the next turn
without requiring the wake word again. The conversation stays open until the
person says a stop phrase or falls silent for the configured sleep timeout, at
which point the robot returns to IDLE and writes the session to memory in the
background. The legal moves between those four states are declared in
`zero/state.py` and asserted on every transition, so a wiring error surfaces as a
logged illegal-transition warning rather than as silent misbehaviour.

Two design commitments shape the whole subsystem and are worth stating before any
architecture is discussed. The first is that heavy models do not run on the Pi.
Speech recognition, the language model, expressive speech synthesis, depth
estimation and the vision-language model all execute on a GPU node reached over a
single SSH tunnel, while the Pi keeps the real-time work: microphone capture,
wake-word detection, endpointing, object detection, tracking, playback and the
loop itself. The second commitment is that every remote faculty holds a local
fallback. If the tunnel drops, transcription falls back to whisper.cpp on the Pi
and the voice falls back to Piper. The robot keeps talking, more slowly and less
expressively, and it is told about its own degradation so it can say so honestly
instead of behaving as though nothing changed.

The subsystem is also responsible for the faculties that make the robot feel
present rather than transactional, and these are in scope precisely because they
are inseparable from the conversation loop. Identity fuses a face embedding from
the current camera frame with a voice embedding from the current utterance, so
the robot knows who it is talking to and, critically, whose memory a turn is
allowed to be written into. Memory is a SQLite store of durable facts, per-person
preferences, rolling session summaries and semantic embeddings for relevance
recall. Vision runs continuously from startup rather than on demand, so the scene
is already perceived by the time anyone asks about it, and perception is never on
the critical path of a reply. Tool use gives the model timers, reminders, web
search and explicit remember and recall verbs. A privacy guard decides, per turn,
whether an unrecognised voice is answered at all and whether their words are
allowed to be stored. A proactive layer lets the robot open a conversation by
itself, greeting a person it recognises walking in or asking a question about
something new it has noticed. A learning loop turns each exchange into a
reward-tagged episode and appends the session to a training corpus for offline
fine-tuning.

Finally, the subsystem exposes a small HTTP control plane on port 8090, running
as a daemon thread inside the same process. This is the fusion surface the AF-1
application uses. It matters that it lives inside the process rather than beside
it: a push-to-talk turn arriving over HTTP drives the same conversation history,
the same memory store, the same tool registry, the same voice and the same
speaker as a turn spoken into the microphone. There is one brain, and the network
is simply another way into it.

## 1.1 In Scope

This document covers the ZERO process in full: the four-state conversation loop
and its transition rules; wake-word detection, voice-activity endpointing and the
semantic hold that keeps the robot listening through a pause in the middle of a
thought; speech-to-text in both its remote and local forms, including the
speculative transcription that starts work at the first pause rather than waiting
for the endpoint; the language model, its persona prompt, its history window and
the rolling compaction that keeps long sessions bounded; text-to-speech across
the Piper, Fish and Orpheus engines together with the orchestrator that
translates one shared cue vocabulary into whatever the active engine understands;
barge-in, which lets a person interrupt the robot mid-sentence and have the
interrupting words carried into the next turn so they are never repeated.

It also covers the perception and state faculties: the always-on camera loop,
open-vocabulary and closed-vocabulary object detection, tracking, colour naming,
scene phrasing, taught objects, the live world-state model and its surprise gate;
face and voice identity, provisional guest clustering for unfamiliar speakers,
and diarisation across a multi-speaker conversation; affect estimation and the
cross-turn mood it feeds; the SQLite memory schema, embeddings, preferences,
consolidation and erasure on request; the tool registry and router; the privacy
guard and its visible indicator; the proactive policy, its triggers and its
adaptive cooldowns; the episode and reward machinery and the corpus export that
feeds fine-tuning. On the GPU side it covers the servers this subsystem calls and
owns the client contract for: the Whisper server, the Orpheus voice server and
the vision server with its depth and vision-language components. It covers
`config.yaml` in full, the `config.local.yaml` override mechanism, the systemd
units, the tunnel and health-check scripts, and the HTTP control plane contract.

## 1.2 Out of Scope

Nothing below the neck belongs to this document. Locomotion, balance, the arms
and end effectors, and any motion planning or safety interlock are covered by the
relevant motion-control document and are not reachable from this subsystem. The
chest Pi 4B and the Nano compute nodes are named here only where ZERO exchanges
messages with them; their own processes, boot order and firmware are documented
separately. Power delivery, thermal design, mechanical assembly and teardown are
outside this document entirely and live in the Engineering Corpus and teardown
volumes.

The AF-1 application is out of scope as a product. This document defines and owns
the HTTP contract the application calls, and stops at that boundary. The
application's own interface, its voice picker, its state handling and its
packaging are documented by the application team. Provisioning of the GPU node is
likewise out of scope: this document specifies what the servers must expose and
how ZERO behaves when they are unreachable, but the machine's operating system,
drivers and physical hosting are not described here. Model training is out of
scope beyond the point where this subsystem writes the corpus; the fine-tuning
pipeline that consumes that corpus is documented on its own. Cloud or hosted
language-model providers are out of scope by design, because the subsystem is
built to run without internet access apart from the optional web-search tool.

## 1.3 Intended Readers

This document is written for four readers, and each of them should be able to
stop at a different depth. An engineer joining the voice stack should be able to
read sections 1 through 4 and understand the loop well enough to trace a single
utterance from microphone to speaker, then use sections 5 and 8 to get a working
machine. An operator or field technician should be able to work almost entirely
from sections 7, 11 and 12, which cover what breaks, what the logs are saying and
exactly which commands restart which box. An integration engineer building
against the robot from the AF-1 application needs section 3, which is the
authoritative interface contract, and section 7 for the failure semantics they
must handle. A technical reviewer conducting due diligence should find sections
9, 10, 13 and 14 sufficient: what the system costs in compute and power, what it
does with personal data, what it depends on and under which licences, and what is
still open. Everything in this document is written against the repository and
commit named on the cover page. Where behaviour is configurable, the exact
`config.yaml` key is given rather than described, so that a claim in the prose can
always be checked against a value in the file.

---

## Figures for this section

Both figures are built and live in `figures/`. Insert at 7.17" wide, wrap Top and
Bottom. What follows is the text that goes into the document around each one.

---

### FIGURE 1 · ZERO SCOPE BOUNDARY

**File:** `figures/fig-01-scope-boundary.svg`

**Place:** at the end of 1.2 Out of Scope, after the paragraph ending "apart from
the optional web-search tool."

**Caption below the image** (Courier, 8 pt, all caps, letterspaced, `#8A8172`):

> FIGURE 1 · ZERO SCOPE BOUNDARY

**Figure note under the caption** (serif, 9.5 pt, `#8A8172`):

> Full line weight marks what this document specifies. The ghosted blocks belong
> to other documents and are named here only where ZERO exchanges messages with
> them. The subsystem is reachable from outside at exactly one point, the control
> plane on port 8090.

**Sentence to add in the body, immediately before the image**, as the closing
line of 1.2 (serif, 10.5 pt, `#25231B`):

> Figure 1 draws the boundary as built: the two machines this document specifies,
> the single tunnel that joins them, the one port that admits anything from
> outside, and the neighbouring subsystems that are named but not described here.

---

### FIGURE 2 · CONVERSATION STATE CYCLE

**File:** `figures/fig-02-conversation-cycle.svg`

**Place:** in the purpose text, directly after the paragraph that begins "The
subsystem runs as one Python process" and ends "rather than as silent
misbehaviour."

**Caption below the image** (Courier, 8 pt, all caps, letterspaced, `#8A8172`):

> FIGURE 2 · CONVERSATION STATE CYCLE

**Figure note under the caption** (serif, 9.5 pt, `#8A8172`):

> The graduated outer band is IDLE, the state the robot rests in. The three inner
> arcs are the conversation cycle, which repeats turn after turn with no further
> wake word. Only the two gates cross between the band and the cycle.

**Sentence to add in the body, immediately before the image** (serif, 10.5 pt,
`#25231B`):

> Figure 2 shows why the loop is drawn as a cycle rather than a list. After the
> wake word admits the first turn, the robot moves between listening, thinking
> and speaking indefinitely without returning to idle, and only a stop phrase or
> a silence long enough to trip the sleep timeout puts it back on the outer band.

---

### Numbering note

As placed above, Figure 2 falls earlier on the page than Figure 1, because the
loop is explained in the purpose text while the boundary belongs at the end of
1.2. Either accept that, or swap the two figure numbers so page order and figure
order agree. The numbers are drawn into each title block, so a swap means editing
both SVGs and renaming both files.
