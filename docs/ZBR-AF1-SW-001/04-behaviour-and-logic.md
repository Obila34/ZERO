# 4 Behaviour & Logic

This section describes what the system actually decides, and on what evidence.
Two habits run through all of it and are worth naming before the detail. The
first is that every judgement the robot makes has an explicit numeric threshold
that lives in `config.yaml` rather than in the code, so behaviour can be tuned in
the field without a deployment. The second is that a faculty which cannot reach a
confident answer degrades to a weaker one rather than failing: an unrecognised
voice still gets a reply, it just does not get to write into anyone's memory; a
scene that cannot be re-detected still publishes its last detections rather than
reporting an empty room. Almost every rule below is shaped by one of those two
habits.

## 4.1 The State Machine

Four states exist: IDLE, LISTENING, THINKING and SPEAKING. The legal moves
between them are declared as data in `zero/state.py` and checked on every
transition. IDLE may only go to LISTENING. LISTENING may go to THINKING, or back
to IDLE when nothing was said. THINKING may go to SPEAKING, to IDLE, or back to
LISTENING, that last case covering an empty reply where the robot stays in the
conversation rather than ending it. SPEAKING may go to IDLE or back to LISTENING,
the latter being barge-in.

An illegal transition is logged as a warning and then performed anyway. That is a
deliberate choice: a wiring bug should be loud in the logs during development but
must never crash a conversation in front of a person. The transition helper also
returns early when the destination equals the current state, which is what allows
the filler and the reply to both request SPEAKING without churn.

The most important structural fact about the loop is not visible in the state
list. After the wake word admits the first turn, the cycle is LISTENING to
THINKING to SPEAKING and back to LISTENING, repeatedly, with no wake word between
turns. IDLE is left behind until a stop phrase or a silence long enough to trip
`conversation.sleep_timeout_ms` returns the robot to it. A reader who models this
as a four-state ring will get the behaviour wrong.

## 4.2 Waking

Wake detection runs openWakeWord over every 30 ms frame in IDLE, comparing its
score against `wake.threshold`. The detector is reset and the microphone drained
whenever the robot enters or re-enters the wait, which matters because the
announcement path can speak while IDLE and the system must not hear its own voice
as a wake attempt.

A conversation can also be opened without a wake word at all. When a proactive
event carries the `open_conversation` flag in its metadata, the idle wait returns
as though woken, and the spoken opener is seeded into the fresh history as the
first assistant turn so that the model knows it said it. This is what allows the
robot to greet someone walking in and then listen for an answer, rather than
speaking into the air and going back to sleep.

## 4.3 Deciding When a Person Has Finished

Endpointing is the single most behaviour-defining piece of logic in the system,
because getting it wrong is immediately obvious to a user: too eager and it
interrupts, too patient and it feels unresponsive.

Two backends exist and the distinction between the code's default and the shipped
configuration matters. The factory defaults to Silero VAD, but `config.yaml`
explicitly selects **webrtc** with an aggressiveness of 2, and that is what runs
on the deployed robot. The reason is recorded in the config itself: Silero is
sharper on clean audio but stricter, and it was missing speech from a quiet
microphone. Anyone reading the factory in isolation will conclude Silero is
active; it is not.

Silero remains available and is worth understanding because it is the fallback
plan for a noisy environment. It runs through ONNX Runtime rather than torch,
which is what keeps the Pi torch-free. Silero v5 requires exactly 512-sample
windows at 16 kHz while the audio pipeline delivers 480-sample frames, so the
endpointer buffers samples and feeds whole 512-sample windows in order, carrying
the model's recurrent state between them. Its ONNX interface is smoke-tested at
construction with a zero-filled window, so a version or API mismatch raises there
and the factory falls back to webrtcvad, rather than producing an endpointer that
silently never detects speech. That failure mode, a microphone that appears alive
but is functionally deaf, is the reason the smoke test exists. If Silero is
selected, `vad.silero_threshold` ships at 0.3 rather than the library default of
0.5, again to accommodate a quiet microphone.

Starting and continuing an utterance use deliberately different rules. To start,
a frame must pass the VAD **and** exceed `vad.energy_threshold` in RMS, which
rejects quiet background onsets and distant voices. To continue, only the VAD is
consulted. That asymmetry is the fix for a real fragmentation bug: quiet
syllables and short pauses inside a sentence were failing the energy test and
splitting one utterance into several. A short pre-roll of `vad.speech_pad_ms` is
retained and prepended when speech starts, so the first word is not clipped.

The endpoint fires after `vad.silence_ms` of trailing silence, expressed
internally as a count of blocks. Two modifiers sit on top. The first is an
adaptive endpoint: if the amount of speech collected so far is less than
`vad.min_speech_for_fast_end_ms`, the required silence is doubled. A slow starter
who says "um" and then thinks is not cut off, while a finished sentence still
commits at the normal speed. The second is the semantic hold described in 4.4.

Two outcomes are carefully distinguished. A true idle timeout, meaning no speech
at all for `conversation.sleep_timeout_ms`, returns nothing and ends the
conversation. An utterance that was captured but whose average RMS falls below
`vad.min_utterance_rms` is dropped and the endpointer keeps listening with a
fresh idle window. This is a proximity gate: the owner's voice close to the
microphone is loud, a conversation across the room is not, and a stray background
blip must not put the robot to sleep mid-conversation. There is also a hard
length cap at `vad.max_utterance_ms`.

While waiting for speech, the endpointer logs a heartbeat every five seconds.
That line is load-bearing for triage and should not be removed: if it ticks but
speech is never captured, the problem is VAD or level; if it never ticks at all,
frames are not arriving and the microphone stream itself has died. Those are
different faults with different fixes, and this is what tells them apart.

## 4.4 Speculation and the Semantic Hold

Roughly 180 ms into any silence run, well before the endpoint would fire, the
endpointer calls back with the audio captured so far and transcription begins on
a background thread. This speculative transcript does two jobs at once.

Its first job is latency. If that pause turns out to be the real end of the
utterance, the transcript is already in hand and the round trip to the GPU has
been spent inside the silence wait rather than added after it. The main loop
decides whether it may reuse the speculative result by two tests: the speculative
audio must be a verbatim prefix of the final utterance, and the amount of audio
that arrived afterwards must be within a slack of twice `vad.silence_ms` plus
`vad.speech_pad_ms` plus 400 ms. A longer tail means speech resumed or the length
cap fired, so the speculative text is only a prefix and the full transcription
runs instead.

Its second job is judgement. When the endpoint condition is met, the loop is
asked whether the speaker was mid-thought, and the answer is derived from the
last word of the speculative transcript. A transcript ending in a conjunction, a
preposition, an article or a filler is treated as unfinished, as is one ending in
a comma, a colon, a dash or a literal ellipsis, which is what Whisper writes when
speech trails off. The word list deliberately excludes words that legitimately
end sentences, such as pronouns.

The critical detail is that this check is tri-state rather than boolean. True
means hold for one more silence window. False means commit now. **None** means
the transcript is still in flight, and in that case the endpointer waits a bounded
`vad.semantic_hold_wait_ms` rather than committing. Without that third state the
mid-thought check raced the STT round trip and lost: the answer arrived just after
the endpoint had already committed, and half-sentences were being shipped to the
model. The hold applies at most once per pause; if speech resumes, the hold is
cleared and the next pause gets a fresh decision.

Speculation is skipped in three cases, each for a distinct reason. It is skipped
when voice ID is enabled, because nothing may be transcribed before the owner
check runs. It is skipped in strict privacy mode for the same reason applied to
bystanders. And it is skipped while STT is degraded to the local CPU engine,
because speculating there simply runs a slow job twice, once at the pause and
once at the endpoint, with no overlap to gain.

## 4.5 Deciding Who Is Speaking

Identity fuses two independent signals. When a face match and a voice match name
the same person, the score is a weighted sum, `w_face` times the face cosine plus
`w_voice` times the voice cosine, with the weights normalised to sum to one, and
the result must clear `identity.fusion.threshold`, which ships at 0.50. When the
two signals name different people, they do not average. Instead the stronger
weighted signal competes alone against its own single-channel threshold, 0.42 for
face and 0.45 for voice, both stricter than the fused threshold because a single
channel is inherently less reliable. This is what prevents two weak and
contradictory matches from manufacturing a confident identification.

Sessions are owned by voice, not by face. Under `identity.session.voice_only`,
which defaults true, the speaker's voice decides whose memories the turn belongs
to and the face is treated as perception only. There is a second, stricter gate
on top: a turn only credits durable memory when the identity score reaches
`identity.session.write_min_score`, which ships at 0.55. A borderline match still
gets a normal conversation, it simply does not write into anyone's permanent
record. That single threshold is what stops a multi-speaker session
cross-contaminating memory.

An unfamiliar voice is not immediately made into a guest. The voiceprint is held,
and only after the transcript proves the turn was real is a provisional guest
assigned. Three gates decide "real": at least `identity.guests.min_words` words,
at least `identity.guests.min_ms` of audio, and at least
`identity.guests.min_rms` in level. These exist because Whisper hallucinates
plausible text from near-silence, producing phantom guests and polluting the
training corpus. Guest identifiers are negative, and clustering an unfamiliar
voice against existing guests keeps different strangers separate.

Diarisation compares consecutive turns' voice embeddings and raises a
speaker-change note when the cosine falls below
`perception.diarize.change_threshold`. The note is ephemeral and is attached to
that turn only.

One rule about sight deserves emphasis because it is a behaviour users notice.
The robot may only claim to see someone when their face is in the current frame.
Recognising a voice off-camera produces a different note, explicitly saying it
recognises the voice but cannot see the face. Furthermore, the identity note is
attached only when recognition changes or when the user asks a visual question.
Repeating it every turn fed the model greeting-fodder and it kept re-greeting
people mid-topic.

## 4.6 Deciding What Is Being Asked About

Most turns carry only a cheap text hint about the scene. A turn classified as
visual additionally pulls recent keyframes so the multimodal model can actually
look. The classifier is deliberately tight in two directions.

Bare demonstratives such as "this", "that" and "there" are excluded, because they
appear in most ordinary sentences. Polysemous verbs such as "see", "look",
"watch" and "picture" are excluded as bare words, because in speech they are
usually non-visual: "I see", "looking forward to it", "picture this scenario".
Those words appear instead inside the phrase list in their genuinely visual
forms, such as "what do you see", "take a look" and "look around".

Beyond the fixed word and phrase lists, the classifier consults the live
detections. If the utterance names an object the camera can see right now, the
turn is visual. This is what makes "what colour is the cup" work without anyone
having to enumerate every possible object: matching is done against the full
label and its head noun, so "cell phone" also matches "phone", plus a naive
plural, on word boundaries. The label "person" is excluded from this test,
because a person is on screen almost always and the word turns up constantly in
abstract speech.

Presence is answered from the detector rather than from the model. Whether a
person is in frame is decided by YOLO and stated to the model as fact, because
without that grounding the model happily answers "yes, I see you" to an empty
room. Three cases are distinguished: the camera never came up, in which case the
model is told it is blind and instructed not to invent a scene; the camera is
working and there is nobody in frame; and there is a person in frame.

Spontaneous scene changes are surfaced as an optional note the model may mention
or ignore, but never on a turn classified as a question. Answering "which planet
did Thanos visit" with "is that a remote on the table?" reads as not listening.
Ungated, the changes stay queued and self-expire until a calmer turn.

## 4.7 Deciding What to Remember and What to Recall

Durable facts are injected once at conversation start rather than per turn, which
keeps the prompt prefix stable and the model's cache warm. Per-turn relevance
recall is separate and ephemeral.

Recall ranks candidate memories by an activation score with four factors:

> activation = relevance × (0.2 + 0.8 × recency) × (importance / 10) × frequency

Relevance is the cosine between the query embedding and the stored embedding when
both exist and their dimensions agree, floored at zero; it falls back to keyword
overlap when there is no embedder, and to 1.0 when there is no query at all,
which reduces the ranking to recency times importance. Recency decays
exponentially from last access rather than creation, so recalling something
refreshes it.

The decay parameter deserves a correction that anyone tuning it needs. It is
named `memory.retrieval.half_life_days` and ships at 14, but the implementation
computes `exp(-age / half_life_seconds)`, which makes it a **time constant, not a
half-life**. At an age equal to the configured value the recency factor has
fallen to 1/e, roughly 0.368, rather than to 0.5. The true half-life is the
configured value multiplied by the natural logarithm of 2, so the shipped setting
of 14 days is really a half-life of about 9.7 days. The name is misleading rather
than the behaviour being wrong, but a reader who sets this expecting half-life
semantics will get a decay about 30 percent faster than intended. This is
recorded as an open item in Section 14.

The `0.2 + 0.8 ×` term puts a floor under the decay so that an old but highly
relevant and important fact can still surface. Frequency is
`1 + 0.1 × ln(1 + access_count)`, a deliberately gentle logarithm so that
frequently accessed memories are favoured slightly without dominating.

Two constraints sit around it. Recall runs on its own thread under a hard budget
of `memory.retrieval.budget_ms`, defaulting to 300 ms; if it does not return in
time the turn proceeds without the note rather than becoming slower, because a
slow embedder in the reply path had been showing up as multi-second first-token
lag. And any hit whose text already appears in the durable block injected at
conversation start is dropped, so the same fact is never spent twice in one
prompt.

In-session compaction keeps a long conversation bounded. History is allowed to
grow to `llm.history_trim_at` turns and only then trimmed back to
`llm.history_turns`, which ships as 12 and 6. This asymmetry is a cache
optimisation: appending is cheap because the model reprocesses only the new
messages, whereas dropping the oldest message shifts the prefix and forces a full
re-read, so the expensive operation is made rare rather than constant. Trimmed
turns are not discarded; they are held pending and folded into a rolling summary
by a background thread, and the trim window is aligned to a user turn because a
history opening with a dangling assistant message reads as replying to nothing.
The summary installer carries a stale-apply guard: if the conversation was reset
while the summary was being computed, the covered turns no longer match and the
summary is dropped, so a dead session can never ghost into a fresh one.

## 4.8 Generating and Speaking

Generation starts on a background thread before the robot has decided whether to
play a filler, so that model prefill overlaps the filler rather than following
it. Handing back a bare generator would not do: a generator does not begin work
until first consumed, so the worker thread is what makes prefill actually start.

The filler races the reply. A filler is chosen with probability
`conversation.filler_probability` and matched to what was said: an utterance
ending in a question mark or opening with a question word gets "good question,
let me think"; a reply of two words or fewer gets a short acknowledgement;
everything else gets a neutral line. It is then played only if no reply audio has
arrived within `conversation.filler_grace_ms`. A fast answer is therefore never
delayed by a canned "let me think". Fillers are pre-synthesised at startup, and
that pre-synthesis aborts after two consecutive failures rather than hammering a
dead TTS at thirty seconds per call.

The reply is spoken sentence by sentence. A producer thread splits the streaming
text into sentences and pushes each sentence's audio pieces onto a bounded queue,
while a single gapless output stream plays them as they arrive, so there are no
inter-sentence pauses. Anything the model writes in parentheses is stripped
before synthesis and never enters history, because those are hallucinated stage
directions rather than speech.

Two failure modes in this path were fixed and both are worth recording. The
producer's completion sentinel must be delivered with a blocking put rather than
a non-blocking one: with `put_nowait` the sentinel was dropped whenever the queue
was full, which happens as soon as synthesis outpaces playback, and a dropped
sentinel left the consumer blocked forever, freezing the conversation in SPEAKING
so the robot went deaf after its first reply. Symmetrically, the producer's own
queue writes must time out and re-check the stop flag, because on a barge-in the
consumer stops draining and a plain blocking put would wedge the producer thread.

Barge-in runs for the whole time the robot is making sound, covering the filler
as well as the reply, and triggers on either the wake word or sustained user
speech over the reply. Speech-based interruption needs no wake word and is
echo-aware, learning the ambient level for
`conversation.barge_in_learn_ms` and requiring `conversation.barge_in_speech_ms`
of speech at `conversation.barge_in_ratio` above it. The interrupting words are
captured and fed into the next turn's transcription so the person never has to
repeat themselves, but only the trigger window plus a short lead-in is kept,
because a longer ring buffer's prefix is the robot's own reply echo and it
garbled the following turn.

After an interruption only what was actually spoken enters history, sliced by the
index of the last sentence whose audio reached the speaker. The model must not
"remember" saying sentences the person never heard. The language model stream is
also stopped and its HTTP connection closed, so the GPU stops generating a reply
nobody is listening to.

One ordering rule in the barge-in shutdown is subtle enough to note. The monitor
thread is joined while the microphone is still live, and only then is the
microphone paused. Pausing first would block the monitor forever inside the frame
iterator, and it would then wake up and steal the next turn's audio off the shared
queue, which is precisely how the robot used to go deaf after its first reply.

## 4.9 Command Paths That Bypass the Model

Several utterances are handled before the model is ever consulted, each with its
own parser, and each replying with a fixed line.

Stop phrases end the conversation. Matching is on word boundaries and only in
utterances of five words or fewer, so a sentence that merely mentions one, such
as asking about a film called Goodbye Lenin, does not put the robot to sleep.

Enrolment has two entry points, an explicit command such as "remember my face"
and a plain introduction such as "I'm David", and both run the same guided
multi-angle capture. Object teaching is distinguished from person introduction by
the article: "this is a french press" teaches an object, "this is Peter" does
not. Behavioural corrections such as "speak slower" or "keep it short" are stored
as standing preferences and, where an engine knob exists, applied immediately.
Erasure distinguishes forgetting the last item from forgetting a person
entirely, and the latter clears the face and voice registrations as well as the
stored facts.

## 4.10 Deciding When to Speak Unprompted

The hard part of proactivity is staying quiet, and every proactive utterance must
pass every gate. A person must be present, and an unrecognised person only counts
when `proactive.engage_unknown` is set. Quiet hours must not be active. The
per-kind cooldown must have elapsed. The person must not already have been
greeted during this arrival, where an arrival ends after
`proactive.presence_reset_s`. And a global cap of `proactive.max_per_hour`
utterances must not be exceeded. Timers and reminders bypass the presence gates,
because the user explicitly asked for those, but they still respect quiet hours
by being deferred rather than dropped.

On top of the fixed cooldowns sits a bandit-style adaptation. Each proactive kind
keeps an exponential moving average of how its utterances land, scored in the
range minus one to one, and that average scales the kind's base cooldown. An
average of zero or unknown leaves the cooldown unchanged. A positive average
shortens it, down to half at plus one. A negative average lengthens it, up to
three times at minus one. A kind of nudge that keeps falling flat therefore backs
off by itself.

## 4.11 Scoring Outcomes and Surprise

Every exchange becomes a reward-tagged episode, with the reward assembled from
three signals that already exist and cost nothing extra. Affect contributes the
speaker's valence and confidence while speaking, on the reasoning that tone is
itself feedback. Interaction contributes a negative for a barge-in, since being
cut off is a verdict, and a mild positive for the conversation continuing within
90 seconds. The strongest signal is an explicit verdict in the opening of the
next utterance, matched by two deliberately narrow regular expressions: generic
negativity such as "this weather is awful" is affect's job, not a judgement on the
robot.

Because a verdict usually arrives one utterance after the reply it judges,
tagging is retrospective: the next utterance writes its reward back onto the
episode it actually judges. A pending proactive utterance is resolved the same
way, with any reply at all counting as a partial success and an explicit verdict
overriding.

Surprise is scored separately and drives attention rather than reward. The world
state keeps online, Laplace-smoothed counts of each label and event kind, and
scores each new event by its rarity in bits, which is the negative base-two
logarithm of its smoothed historical share. First-ever events score highest and
daily routine decays toward zero. Two thresholds consume that score: events above
`world.surprise.remember_bits` become scene episodes for later consolidation, and
events above `world.surprise.narrate_bits` wake the Tier 2 narrator. The
statistics persist across runs as a JSON sidecar, so the robot's sense of what is
normal accumulates over its lifetime rather than resetting at every boot.

---

## Figures for this section

Three figures. Files in `figures/`. Insert at 7.17" wide, wrap Top and Bottom.

### FIGURE 9 · ENDPOINT DECISION LADDER

**File:** `figures/fig-09-endpoint-ladder.svg`
**Place:** at the end of 4.4.
**Caption:** FIGURE 9 · ENDPOINT DECISION LADDER
**Figure note:**
> One pause, drawn as a silence axis. The bars mark where each rule fires. The
> tri-state hold is the only branch that can extend the wait, and it is bounded,
> which is what stops the check racing the transcription it depends on.

### FIGURE 10 · IDENTITY DECISION SURFACE

**File:** `figures/fig-10-identity-surface.svg`
**Place:** at the end of 4.5.
**Caption:** FIGURE 10 · IDENTITY DECISION SURFACE
**Figure note:**
> Face cosine against voice cosine. The shaded regions are where each verdict is
> reached. The single-channel bands are stricter than the fused line, which is why
> two weak but agreeing signals accept while two weak and conflicting ones do not.

### FIGURE 11 · MEMORY ACTIVATION DECAY

**File:** `figures/fig-11-activation-decay.svg`
**Place:** at the end of the recall discussion in 4.7.
**Caption:** FIGURE 11 · MEMORY ACTIVATION DECAY
**Figure note:**
> Recency contribution against age, at the shipped 14 day half-life. The floor at
> 0.2 is what allows an old but important and relevant memory to still surface;
> without it, recall would be purely a function of how recently something was said.
