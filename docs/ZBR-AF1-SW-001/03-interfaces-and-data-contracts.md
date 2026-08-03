# 3 Interfaces & Data Contracts

An interface, for the purposes of this document, is anything that crosses the
boundary drawn in Figure 1. That includes the HTTP control plane the AF-1
application calls, every request ZERO sends up the SSH tunnel to a model server,
the physical devices the Pi opens, the GPIO pin that drives the recording
indicator, and every file the system writes to disk. Internal function calls
between modules are not interfaces and are not listed here; they are covered by
Section 4. The distinction matters because everything in this section is a
contract that something outside the subsystem depends on, and therefore cannot be
changed without changing something else at the same time.

Three conventions run through all of the contracts below and are worth stating
once rather than repeating in every row. First, all audio crossing any boundary
is mono, 16-bit signed PCM, wrapped in a self-describing WAV container, so the
sample rate travels with the data rather than being agreed out of band. The one
exception is speech coming back from Orpheus, which is 24 kHz because that is
what its vocoder produces, and which the Pi resamples on receipt. Second, all
images crossing any boundary are JPEG-encoded and then base64-encoded into a JSON
string field named `image_jpeg_b64`, at a quality set by `vision.gpu.jpeg_quality`
and defaulting to 80. Third, every embedding stored on disk is a float32 array
written as a raw little-endian blob, always accompanied by an integer `dim`
column in the same row. Readers compare `dim` against their own model's dimension
and skip mismatched rows rather than failing, which is what allows an embedding
model to be swapped without invalidating the database.

There is a fourth convention that is easy to miss and causes real confusion when
missed. Throughout the system, a person identifier is a signed integer whose sign
carries meaning. A positive value is an enrolled person in the identity registry.
A negative value is a provisional guest, minted by clustering an unfamiliar voice
and deliberately kept separate so that one stranger's words never merge with
another's. A null value is an anonymous turn, which is what text mode produces and
what happens when identity is disabled or the speaker could not be placed. Every
store that carries `person_id`, and every corpus record, uses this convention, and
code that treats the field as an opaque key will silently mix guests with people.

## 3.1 The Control Plane

The control plane listens on TCP 8090, bound to `0.0.0.0` by `control.host`, and
is the only inbound interface the subsystem exposes. All responses are JSON
unless stated otherwise. A cross-origin preflight is answered with 204 and the
permissive header set described at the end of this subsection. Any request body
larger than 32 MB is rejected, which is generous for the intended use since a
minute of WebM Opus is well under it. Unhandled exceptions inside a handler are
caught, logged with a traceback, and returned as HTTP 500 with the exception text
truncated to 200 characters, so a malformed request can never take down the
process that is also holding the conversation.

`GET /health`, also reachable as `GET /zero/health`, returns
`{ok, service, state, ready}`. The `service` field is the constant string
`zero-control`. The `state` field is the current conversation state as its
lowercase value, one of `idle`, `listening`, `thinking` or `speaking`. The `ready`
field is the one that matters operationally: it is false from process start until
the language model has been pinned in memory by the warmup call, and true
afterwards. A client that treats a reachable-but-not-ready Pi as broken will
report false failures during the first few seconds after a restart.

`GET /zero/status` returns the state, the last external turn, and the current
degradation flags. This is the endpoint that lets the application show that the
robot is running on its backup voice without having to ask it.

`POST /zero/say` takes `{text, voice?}` and speaks the line on the Pi speaker
without involving the language model or writing anything to memory. The text is
truncated to 500 characters. An empty or missing `text` returns 400 with
`{ok:false, error:"empty-text"}`. The optional `voice` is an Orpheus speaker name
such as `leo`, `tara` or `draco`, and it overrides the configured default for
that request only, leaving ZERO's own default untouched.

`POST /zero/turn` is the push-to-talk path. The request body is raw audio bytes,
not JSON and not multipart, in any container ffmpeg can decode, which in practice
means WebM, OGG, WAV or MP4. Options travel as query parameters, `?voice=` and
`?person_id=`. The server shells out to ffmpeg to convert the body to mono float32
at the configured sample rate. A body that decodes cleanly runs a complete brain
turn, is spoken on the Pi speaker, and returns `{ok, heard, reply}`, where `heard`
is the transcript and `reply` is what was said. An empty body returns 400. A
decode failure returns 422 with the tail of ffmpeg's stderr in the error field,
which is deliberate: a caller sending an unsupported container needs to see why.

`POST /zero/turn_text` takes `{text, voice?, speak?, person_id?}` and runs the
same brain turn from typed input, with `text` truncated to 1200 characters and
`speak` defaulting to true. `POST /zero/control` accepts `{action:"sleep"}` and
ends the open conversation; any other action returns 400 naming the unknown
action. Anything else returns 404.

External turns are serialised against each other by a single lock and are held
off the native loop's think and speak phase, so the microphone and the network
cannot both drive the state machine at once. The `person_id` supplied by the
application defaults to `control.person_id`, which ships as 1, meaning AF-1 turns
are attributed to the logged-in operator rather than to an anonymous speaker.

CORS is deliberately unrestricted: `Access-Control-Allow-Origin: *`, methods
`GET, POST, OPTIONS`, headers `Content-Type`. The Tauri application fetches from
the Rust side and needs no CORS at all, but browser-based development does, and
no authentication is implemented at this layer. Combined with the `0.0.0.0` bind,
this means anything on the local network can make the robot speak. That is a
deliberate trade for a LAN-only device and it is assessed properly in Section 10,
not defended here.

## 3.2 The Model Server Contracts

Every one of these is reached through the SSH tunnel at a local port, so from
ZERO's point of view they are all on `127.0.0.1`. The port offset described in
Section 2.1 applies: Ollama alone is reached at local 11435.

The Whisper server accepts `POST /transcribe` with a raw WAV body and
`Content-Type: audio/wav`, and returns `{text}`. An optional `?language=` query
parameter is appended by the client when `stt.language` is set, accepting a
language code such as `sw`, or the literal `auto` to let Whisper detect, which is
the right setting for code-switched speech. The client raises rather than
returning an empty string on failure, and that distinction is load-bearing: the
fallback wrapper must be able to tell a dead tunnel from a silent room.

The Orpheus server accepts `POST /tts` with `{text, voice}` and returns a WAV at
24 kHz. A streaming variant at `/tts_stream` is derived by the client from the
configured URL by simple substring replacement, which means a non-standard path
in `tts.orpheus.url` will produce a stream URL that does not exist. `GET /health`
returns `{ok}` reflecting whether the model has loaded. The client keeps one HTTP
session alive across sentences rather than reconnecting, and retries a connection
failure once on a fresh connection so that a server restarted between turns
recovers rather than erroring. Before sending, the client rewrites ZERO's shared
cue vocabulary into Orpheus's native tags, mapping `[laughs]` to `<laugh>`,
`[sighs]` to `<sigh>` and so on, rendering `[hmm]` and `[pause]` as text the model
performs naturally, and stripping any cue it does not recognise.

The vision server exposes `GET /health`, which reports CUDA availability and VRAM
and is what proves the tunnel is alive; `POST /facts`, which takes an
`AnalyzeRequest` and returns an `AnalyzeResponse` whose `reply` is empty because
ZERO's own model writes the spoken answer; and `POST /analyze`, which runs the
same facts and then a vision-language model grounded in them, returning a
populated `reply`.

The shared wire types are Pydantic v2 models. A `Detection` carries `label` as a
string, `bbox` as exactly four floats in `[x, y, w, h]` in pixels of the frame
that was sent, `confidence` constrained to the range 0 to 1, and an optional
`color` string from HSV naming. An `AnalyzeRequest` carries `image_jpeg_b64`, a
list of `Detection`, a `question` string and a `history` list of `{role, content}`
dictionaries. A `SceneFact` carries `label`, optional `color`, optional
`distance_m` constrained to be non-negative, and an optional `bearing` which is
one of the coarse strings `left`, `center` or `right`. An `AnalyzeResponse`
carries a list of `SceneFact` and a `reply` string.

These models exist in two files, one on each side of the tunnel, because the two
nodes do not share a filesystem. **The two copies have drifted.** In the Pi copy
at `zero/vision/schemas.py`, both `AnalyzeRequest.question` and
`AnalyzeResponse.reply` have empty-string defaults and are therefore optional. In
the server copy at `server/vision/shared/schemas.py`, both are declared required.
Today this is benign, because the Pi serialises its own model and so always emits
both keys even when they are empty, but the contract is no longer single-sourced
and the next edit to either file can diverge further without anything failing
loudly. The server copy's docstring compounds this by instructing the reader to
edit a canonical file and run `python shared/sync_schemas.py`, and by naming a
mirror at `pi/shared/schemas.py`. Neither that script nor that path exists in this
repository. This is recorded as an open item in Section 14.

## 3.3 The Perception Offload

Separate from the vision endpoints above, the GPU node serves a set of embedding
and detection endpoints under `/perceive/` on the same port 8000, reached through
a client with its own timeouts: 5 seconds by default and 10 seconds for
detection. Every one of them raises a `RuntimeError` naming the path on any
failure, again so that the fallback wrappers can distinguish an unreachable
server from an empty result.

`POST /perceive/detect` takes `{image_jpeg_b64}` and returns `{detections: [...]}`
in the `Detection` shape. `POST /perceive/face` takes
`{image_jpeg_b64, max_faces}`, defaulting to three faces, and returns
`{faces: [{embedding: [float, ...], bbox: [...]}, ...]}`; the client discards any
face whose embedding array is empty. `POST /perceive/speaker` is the one endpoint
in this group that does not take JSON: the body is a raw WAV with
`Content-Type: audio/wav`, built from the utterance at the capture sample rate,
and the response is `{embedding: [float, ...]}`. `POST /perceive/embed_object`
takes `{image_jpeg_b64}` holding a crop rather than a full frame and returns
`{embedding: [float, ...]}`.

## 3.4 Local Device Interfaces

The microphone is opened through PortAudio at the index given by
`audio.input_device`, and the pipeline contract downstream of it is fixed: mono,
16-bit, 16 kHz, in blocks of `audio.block_ms` milliseconds, which ships as 30 and
therefore yields 480 samples per block. Two adaptations sit inside that contract.
A software gain from `audio.input_gain` multiplies each block before int16
conversion, because some USB and webcam microphones capture too quietly for the
wake model to fire. And when a device rejects 16 kHz outright, which is common
for cheap USB dongles that only support 44.1 or 48 kHz, the stream is opened at
the device's native rate and each block is resampled down, so every consumer
still sees the same frames it would otherwise.

The speaker is opened at `audio.output_device`. The camera is opened at
`vision.camera.index` and configured for 640 by 480 at a requested 30 frames per
second in MJPG, which is deliberately small so that detection keeps up on the Pi.

The recording indicator is a single GPIO pin named by
`privacy.indicator_gpio_pin`, driven through gpiozero. It is lit for the
`listening`, `thinking` and `speaking` states and dark for `idle`. With no pin
configured, or with gpiozero unavailable, it degrades to state-transition log
lines so that the same information is still observable through journalctl. This
is the only physical output the subsystem drives.

The optional camera preview serves HTTP on `vision.preview_port`, which ships as
8008, bound to `vision.preview_host`. That default is `127.0.0.1` on purpose: the
preview is an unauthenticated video stream, and publishing it to the network
requires an explicit change to `0.0.0.0`.

## 3.5 On-Disk Contracts

Six SQLite databases and two flat files make up the persistent state. All of them
are created on first use, all of them are opened with `check_same_thread=False`
because several threads write, and the episode store additionally runs in WAL
mode so that writers never block readers.

`zero_memory.sqlite` holds one table, `memories`, with columns `id`, `person_id`,
`layer`, `key`, `value`, `importance` defaulting to 5.0, `emb` as a float32 blob,
`emb_dim`, `created_at`, `last_access`, `access_count` and `protected`. There is
an index on `(layer, person_id)`. The `protected` flag marks rows that the storage
cap must never prune, which is what keeps the durable per-person
last-conversation record alive when ordinary facts are being aged out. This store
also carries a one-time legacy migration that folds rows from an older flat
schema, with tables named `memory` and `episodes`, into the current shape and then
drops them, so an old database upgrades in place on first open.

`zero_identity.sqlite` holds `people`, with `id`, a `name` that is unique and
case-insensitive, and `created_at`; and `embeddings`, with `id`, `person_id`,
`kind`, `dim`, a `vec` blob and `created_at`, with a foreign key to `people`. The
`kind` column is what allows face and voice vectors to live in one table.

`zero_guests.sqlite` holds `guest_samples`, with `id`, `guest`, `dim`, `vec` and
`created_at`. Guest identifiers are negative by construction: a new guest takes
the minimum of minus one and one below the lowest existing identifier. The store
caps samples per guest and total guests, dropping the least recently heard.

`zero_episodes.sqlite` holds `episodes`, with `id`, `v` recording the schema
version at write time, `ts`, `kind` drawn from `turn`, `scene`, `proactive` and
`action`, `person_id`, a `payload` JSON string defaulting to `{}`, `reward` where
null means untagged, `surprise`, and `consolidated_at`. It carries two indexes,
one on `(kind, ts)` and a partial index on unconsolidated rows. Alone among the
stores it uses a real migration mechanism driven by `PRAGMA user_version`, applying
each migration step in its own transaction. Rewards are clamped to the range
minus one to one on write.

`zero_curiosity.sqlite` holds `questions`, with `id`, a unique `source_key` that
prevents the same observation queuing twice, `text`, `priority`, `person_id`,
`created_at` and `asked_at`. `zero_objects.sqlite` holds `objects`, with `id`, a
case-insensitive `name`, `person_id`, `dim`, `vec` and `created_at`, which is
where a taught object binds a name to an embedding.

The interaction corpus is newline-delimited JSON at `data/corpus/interactions.jsonl`,
appended under a lock so records never interleave. One record is written per
speaker per session, holding that speaker's turns, their identifier under the sign
convention described above, and a timestamp. Because the session has already been
split by speaker before it is written, one person's speech never contaminates
another's training data. The enrolled voiceprint is a NumPy array at the path in
`voiceid.profile_path`, and the surprise predictor keeps its running statistics as
JSON at `world.surprise.stats_path`.

## 3.6 The Internal Event Bus

The event bus is the one internal mechanism documented here, because it is how
anything that is not the main loop gets the robot to speak, and because its
overflow behaviour is a contract rather than an implementation detail. An `Event`
carries `kind`, `text` already phrased for speech, `created_at`, an optional
`person_id`, and a `meta` dictionary. The queue holds 64 events. Posting to a full
queue drops the event and returns false rather than blocking, on the reasoning
that a lost nudge is better than a wedged timer thread. The main loop drains the
bus only at safe moments, meaning the idle wake-wait and turn boundaries, so an
announcement can never talk over a reply in progress. One `meta` key is
load-bearing: `open_conversation`, which tells the loop that the announcement
expects an answer and that it should begin listening rather than returning to
idle.

---

## Tables for this section

Two tables. Styled Word versions matching the template's markup are in
`tables/`, alongside Table 1.

**TABLE 2 · INTERFACE CONTRACTS** at `tables/table-02-interface-contracts.docx`.
Place after 3.4, so it summarises every crossing described in 3.1 through 3.4.

**TABLE 3 · PERSISTENT STORES** at `tables/table-03-persistent-stores.docx`.
Place at the end of 3.5.

---

## Figures for this section

### FIGURE 6 · TUNNEL PORT MAP

**File:** `figures/fig-06-port-map.svg`

**Place:** at the start of 3.2, before the paragraph beginning "The Whisper
server accepts".

Drawn as a patch panel: two labelled connector blocks, Pi side and node side,
with numbered ports and a cable per forward. The Ollama forward is drawn in the
accent colour because it is the one that changes number in transit.

**Caption:**

> FIGURE 6 · TUNNEL PORT MAP

**Figure note:**

> Five forwards on one tunnel. Four are straight through. Ollama alone shifts
> from local 11435 to remote 11434, which leaves 11434 free on the Pi for a local
> fallback model.

**Body sentence before the image:**

> Figure 6 lays the five forwards out as a panel, because the single asymmetry in
> the map is the thing most likely to be misconfigured and the easiest to see when
> drawn rather than described.

---

### FIGURE 7 · PERSISTENT STORE MAP

**File:** `figures/fig-07-store-map.svg`

**Place:** at the end of 3.5, after the paragraph ending "as JSON at
`world.surprise.stats_path`."

The six databases and two flat files drawn as labelled storage blocks, each
listing its tables and key columns, with the blob-bearing columns marked. A
sign-convention key sits alongside, showing positive, negative and null
`person_id`.

**Caption:**

> FIGURE 7 · PERSISTENT STORE MAP

**Figure note:**

> Every store that carries a person identifier obeys the same sign convention.
> Every embedding column is float32 little-endian with its dimension in the
> adjacent column, which is what allows an embedding model to change without
> invalidating the file.

**Body sentence before the image:**

> Figure 7 shows the eight files together with the columns that carry the
> conventions, since it is the conventions rather than the individual tables that
> a reader has to hold in mind.
