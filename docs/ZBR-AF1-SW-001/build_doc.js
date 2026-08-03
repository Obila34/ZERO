const docx = require('docx');
const fs = require('fs');
const path = require('path');

const {
    Document, Packer, Paragraph, TextRun, ImageRun,
    HeadingLevel, AlignmentType, TabStopPosition,
    TableRow, TableCell, Table, WidthType, BorderStyle,
    ShadingType, PageBreak, convertInchesToTwip,
    Header, Footer, PageNumber, NumberFormat
} = docx;

const BASE = __dirname;
const FIG = path.join(BASE, 'figures');

// ── Colours ──────────────────────────────────────────────────────────────────
const DARK   = '25231B';
const MID    = '8A8172';
const RULE_C = 'D9D4C7';
const BG     = 'F5F2EC';

// ── Helpers ──────────────────────────────────────────────────────────────────
function bodyPara(text, opts = {}) {
    return new Paragraph({
        spacing: { after: 160, line: 276 },
        alignment: AlignmentType.JUSTIFIED,
        ...opts,
        children: [
            new TextRun({
                text,
                font: 'Constantia',
                size: 21,          // 10.5 pt
                color: DARK,
            }),
        ],
    });
}

function bodyParaRuns(runs, opts = {}) {
    return new Paragraph({
        spacing: { after: 160, line: 276 },
        alignment: AlignmentType.JUSTIFIED,
        ...opts,
        children: runs,
    });
}

function sectionHeading(num, title) {
    return new Paragraph({
        spacing: { before: 480, after: 200 },
        children: [
            new TextRun({
                text: `${num}  ${title.toUpperCase()}`,
                font: 'Franklin Gothic Heavy',
                size: 28,        // 14 pt
                color: DARK,
                bold: true,
            }),
        ],
    });
}

function subHeading(num, title) {
    return new Paragraph({
        spacing: { before: 360, after: 160 },
        children: [
            new TextRun({
                text: `${num}  ${title}`,
                font: 'Franklin Gothic Heavy',
                size: 22,        // 11 pt
                color: DARK,
                bold: true,
            }),
        ],
    });
}

function figCaption(text) {
    return new Paragraph({
        spacing: { before: 120, after: 40 },
        alignment: AlignmentType.CENTER,
        children: [
            new TextRun({
                text,
                font: 'Courier New',
                size: 16,   // 8 pt
                color: MID,
                characterSpacing: 60,
            }),
        ],
    });
}

function figNote(text) {
    return new Paragraph({
        spacing: { after: 200 },
        alignment: AlignmentType.CENTER,
        children: [
            new TextRun({
                text,
                font: 'Constantia',
                size: 19,   // 9.5 pt
                color: MID,
                italics: true,
            }),
        ],
    });
}

function insertFigure(num, caption, note, widthInches = 7.17) {
    const pad = String(num).padStart(2, '0');
    const pngPath = path.join(FIG, `fig-${pad}.png`);
    if (!fs.existsSync(pngPath)) {
        console.warn(`WARNING: ${pngPath} missing, skipping figure ${num}`);
        return [bodyPara(`[FIGURE ${num} IMAGE MISSING]`)];
    }
    const imgData = fs.readFileSync(pngPath);
    const widthPx = 1800;
    // Estimate aspect ratio from PNG header
    // PNG: bytes 16-19 = width, 20-23 = height (big endian)
    const pngW = imgData.readUInt32BE(16);
    const pngH = imgData.readUInt32BE(20);
    const aspect = pngH / pngW;
    const wEmu = Math.round(widthInches * 914400);
    const hEmu = Math.round(wEmu * aspect);
    return [
        new Paragraph({
            spacing: { before: 200, after: 0 },
            alignment: AlignmentType.CENTER,
            children: [
                new ImageRun({
                    data: imgData,
                    transformation: {
                        width: Math.round(widthInches * 96),
                        height: Math.round(widthInches * 96 * aspect),
                    },
                    type: 'png',
                }),
            ],
        }),
        figCaption(caption),
        figNote(note),
    ];
}

function hrule() {
    return new Paragraph({
        spacing: { before: 120, after: 120 },
        border: {
            bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE_C },
        },
        children: [],
    });
}

// ── Split prose into paragraphs ─────────────────────────────────────────────
function prose(text) {
    return text
        .split(/\n\n+/)
        .map(p => p.replace(/\n/g, ' ').trim())
        .filter(p => p.length > 0)
        .map(p => bodyPara(p));
}

// ── Inline code run ─────────────────────────────────────────────────────────
function codeRun(text) {
    return new TextRun({
        text,
        font: 'Courier New',
        size: 19,
        color: DARK,
    });
}

function bodyRun(text) {
    return new TextRun({
        text,
        font: 'Constantia',
        size: 21,
        color: DARK,
    });
}

// ── Build a styled table matching the template ──────────────────────────────
function buildTable(title, headers, rows) {
    const colCount = headers.length;
    const totalWidth = 10315;
    const colW = Math.floor(totalWidth / colCount);
    const colWidths = headers.map((_, i) => i === colCount - 1 ? totalWidth - colW * (colCount - 1) : colW);

    const cellBorders = {
        top: { style: BorderStyle.SINGLE, size: 4, color: RULE_C },
        bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE_C },
        left: { style: BorderStyle.NONE },
        right: { style: BorderStyle.NONE },
    };

    const titleRow = new Paragraph({
        spacing: { before: 360, after: 120 },
        children: [
            new TextRun({
                text: title,
                font: 'Courier New',
                size: 16,
                color: MID,
                characterSpacing: 60,
            }),
        ],
    });

    const headerRow = new TableRow({
        tableHeader: true,
        children: headers.map((h, i) =>
            new TableCell({
                width: { size: colWidths[i], type: WidthType.DXA },
                shading: { type: ShadingType.CLEAR, fill: DARK },
                borders: cellBorders,
                children: [
                    new Paragraph({
                        spacing: { before: 40, after: 40 },
                        children: [
                            new TextRun({
                                text: h,
                                font: 'Courier New',
                                size: 16,
                                color: 'FFFFFF',
                                bold: true,
                            }),
                        ],
                    }),
                ],
            })
        ),
    });

    const dataRows = rows.map((row, ri) =>
        new TableRow({
            children: row.map((cell, ci) =>
                new TableCell({
                    width: { size: colWidths[ci], type: WidthType.DXA },
                    shading: { type: ShadingType.CLEAR, fill: ri % 2 === 0 ? 'FFFFFF' : BG },
                    borders: cellBorders,
                    children: [
                        new Paragraph({
                            spacing: { before: 30, after: 30 },
                            children: [
                                new TextRun({
                                    text: cell,
                                    font: 'Courier New',
                                    size: 15,
                                    color: DARK,
                                }),
                            ],
                        }),
                    ],
                })
            ),
        })
    );

    return [
        titleRow,
        new Table({
            width: { size: totalWidth, type: WidthType.DXA },
            columnWidths: colWidths,
            rows: [headerRow, ...dataRows],
        }),
    ];
}

// ═══════════════════════════════════════════════════════════════════════════
// TABLE DATA
// ═══════════════════════════════════════════════════════════════════════════
const TABLE1_HEADERS = ['PROCESS / THREAD', 'RUNS ON', 'FUNCTION', 'RATE'];
const TABLE1_ROWS = [
    ['MainThread', 'Pi 5', 'The four-state conversation loop', 'Event driven'],
    ['PortAudio callback', 'Pi 5', 'Microphone capture, gain, resample, echo pause', '30 ms frames'],
    ['CameraStream', 'Pi 5', 'Frame grab only', '640x480, 30 fps requested'],
    ['Eyes', 'Pi 5', 'Tiered perception, colour, tracking, scene diff', 'Per frame, gated'],
    ['Narrator', 'Pi 5', 'Tier 2 scene narration via the VLM', '3.0 s, budget capped'],
    ['SurpriseGate', 'Pi 5', 'Prediction-error scoring into episodes', 'On world change'],
    ['proactive', 'Pi 5', 'Presence, curiosity, idle consolidation', '3.0 s tick'],
    ['zero-control', 'Pi 5', 'HTTP control plane', 'Request driven, port 8090'],
    ['EyesWebPreview', 'Pi 5', 'Optional camera preview', 'Request driven, port 8008'],
    ['stt-spec', 'Pi 5', 'Speculative transcription at first pause', 'Once per pause'],
    ['stt', 'Pi 5', 'Final transcription', 'Once per turn'],
    ['recall', 'Pi 5', 'Relevance search in memory', 'Once per turn, 300 ms cap'],
    ['llm-stream', 'Pi 5', 'Background generation', 'Once per turn'],
    ['tts-producer', 'Pi 5', 'Synthesis ahead of playback', 'Once per turn'],
    ['bargein', 'Pi 5', 'Interrupt monitor while speaking', 'Once per turn'],
    ['compaction', 'Pi 5', 'Rolling summary of trimmed turns', 'On trim, single flight'],
    ['memory-save', 'Pi 5', 'Per-speaker durable write, corpus append', 'Once per session'],
    ['timer-<id>', 'Pi 5', 'One per pending timer or reminder', 'Until fired'],
    ['Whisper server', 'GPU node', 'Speech to text, large-v3-turbo', 'Port 9000'],
    ['Orpheus server', 'GPU node', 'Expressive speech, 3B via vLLM', 'Port 9100'],
    ['Vision server', 'GPU node', 'Depth, scene facts, VLM, embeddings', 'Port 8000'],
    ['Ollama', 'GPU node', 'Conversational model, gemma4 8B', 'Port 11434'],
    ['SearXNG', 'GPU node', 'Web search backend', 'Port 8080'],
];

const TABLE2_HEADERS = ['INTERFACE', 'DIRECTION', 'TRANSPORT', 'CONTRACT / SCHEMA'];
const TABLE2_ROWS = [
    ['GET /health', 'App to Pi', 'HTTP 8090', '{ok, service, state, ready}'],
    ['GET /zero/status', 'App to Pi', 'HTTP 8090', '{state, last_turn, degradation}'],
    ['POST /zero/say', 'App to Pi', 'HTTP 8090', '{text, voice?} JSON'],
    ['POST /zero/turn', 'App to Pi', 'HTTP 8090', 'Raw audio body (any ffmpeg container)'],
    ['POST /zero/turn_text', 'App to Pi', 'HTTP 8090', '{text, voice?, speak?, person_id?}'],
    ['POST /zero/control', 'App to Pi', 'HTTP 8090', '{action:"sleep"}'],
    ['POST /transcribe', 'Pi to GPU', 'HTTP 9000', 'WAV body, returns {text}'],
    ['POST /tts', 'Pi to GPU', 'HTTP 9100', '{text, voice} JSON, returns WAV 24 kHz'],
    ['POST /tts_stream', 'Pi to GPU', 'HTTP 9100', 'Streaming variant of /tts'],
    ['GET /health (vision)', 'Pi to GPU', 'HTTP 8000', '{ok, cuda, vram}'],
    ['POST /facts', 'Pi to GPU', 'HTTP 8000', 'AnalyzeRequest, returns SceneFact[]'],
    ['POST /analyze', 'Pi to GPU', 'HTTP 8000', 'AnalyzeRequest, returns AnalyzeResponse'],
    ['POST /perceive/detect', 'Pi to GPU', 'HTTP 8000', '{image_jpeg_b64}, returns {detections}'],
    ['POST /perceive/face', 'Pi to GPU', 'HTTP 8000', '{image_jpeg_b64, max_faces}'],
    ['POST /perceive/speaker', 'Pi to GPU', 'HTTP 8000', 'WAV body, returns {embedding}'],
    ['POST /perceive/embed_object', 'Pi to GPU', 'HTTP 8000', '{image_jpeg_b64}, returns {embedding}'],
    ['Ollama /api/chat', 'Pi to GPU', 'HTTP 11435', 'Ollama chat API, streaming'],
    ['SearXNG /search', 'Pi to GPU', 'HTTP 8080', 'Standard SearXNG JSON API'],
    ['Microphone', 'Device to Pi', 'PortAudio', 'Mono 16-bit 16 kHz, 30 ms blocks'],
    ['Speaker', 'Pi to device', 'PortAudio', 'Float32 playback'],
    ['Camera', 'Device to Pi', 'OpenCV', '640x480 MJPG, 30 fps requested'],
    ['GPIO indicator', 'Pi to LED', 'gpiozero', 'High = recording, Low = idle'],
    ['Preview server', 'Pi to browser', 'HTTP 8008', 'MJPG stream, localhost only'],
];

const TABLE3_HEADERS = ['STORE', 'TABLE', 'KEY COLUMNS', 'NOTES'];
const TABLE3_ROWS = [
    ['zero_memory.sqlite', 'memories', 'id, person_id, layer, key', 'emb blob + emb_dim; protected flag'],
    ['zero_identity.sqlite', 'people', 'id, name (unique CI)', 'FK to embeddings'],
    ['zero_identity.sqlite', 'embeddings', 'id, person_id, kind', 'face/voice vec blob + dim'],
    ['zero_guests.sqlite', 'guest_samples', 'id, guest (negative)', 'Capped per guest and total'],
    ['zero_episodes.sqlite', 'episodes', 'id, v, ts, kind', 'WAL mode; reward clamped [-1,1]'],
    ['zero_curiosity.sqlite', 'questions', 'id, source_key (unique)', 'Priority queue, asked_at nullable'],
    ['zero_objects.sqlite', 'objects', 'id, name (CI)', 'Taught object embeddings'],
    ['interactions.jsonl', '(flat file)', 'timestamp, person_id', 'NDJSON, one record per speaker per session'],
    ['voiceprint .npy', '(flat file)', 'person_id', 'NumPy float32 array'],
    ['surprise stats.json', '(flat file)', 'label/event counts', 'Laplace-smoothed, persists across boots'],
];

// ═══════════════════════════════════════════════════════════════════════════
// SECTION CONTENT
// ═══════════════════════════════════════════════════════════════════════════

function section1() {
    const items = [];
    items.push(sectionHeading('1', 'Purpose & Scope'));

    // First paragraph
    items.push(bodyPara(
        'ZERO is the conversational intelligence subsystem of the AF-1. It is the part of the robot a person actually talks to. Everything it does resolves into one user-visible behaviour: somebody speaks near the robot, and the robot answers out loud in its own voice, promptly, knowing who is speaking, remembering what has been said before, and aware of what is currently in front of its camera. Every module described in this document exists to make that single behaviour happen and to keep it happening when parts of the system fail.'
    ));

    items.push(bodyPara(
        'The subsystem runs as one Python process on the Raspberry Pi 5 in the AF-1 head, started as python -m zero.main. Inside that process a four-state loop turns continuously. The robot sits in IDLE with the microphone open, listening only for its wake word. On a wake detection it moves to LISTENING and captures the utterance until a voice-activity endpointer decides the person has finished speaking. It then moves to THINKING, where transcription, speaker identification, emotional read, memory recall and language-model generation all run, several of them in parallel. It finishes in SPEAKING, where the reply is synthesised and played sentence by sentence, and then returns to LISTENING for the next turn without requiring the wake word again. The conversation stays open until the person says a stop phrase or falls silent for the configured sleep timeout, at which point the robot returns to IDLE and writes the session to memory in the background. The legal moves between those four states are declared in zero/state.py and asserted on every transition, so a wiring error surfaces as a logged illegal-transition warning rather than as silent misbehaviour.'
    ));

    // Figure 2 goes here (conversation cycle)
    items.push(bodyPara(
        'Figure 2 shows why the loop is drawn as a cycle rather than a list. After the wake word admits the first turn, the robot moves between listening, thinking and speaking indefinitely without returning to idle, and only a stop phrase or a silence long enough to trip the sleep timeout puts it back on the outer band.'
    ));
    items.push(...insertFigure(2,
        'FIGURE 2 · CONVERSATION STATE CYCLE',
        'The graduated outer band is IDLE, the state the robot rests in. The three inner arcs are the conversation cycle, which repeats turn after turn with no further wake word. Only the two gates cross between the band and the cycle.'
    ));

    items.push(bodyPara(
        'Two design commitments shape the whole subsystem and are worth stating before any architecture is discussed. The first is that heavy models do not run on the Pi. Speech recognition, the language model, expressive speech synthesis, depth estimation and the vision-language model all execute on a GPU node reached over a single SSH tunnel, while the Pi keeps the real-time work: microphone capture, wake-word detection, endpointing, object detection, tracking, playback and the loop itself. The second commitment is that every remote faculty holds a local fallback. If the tunnel drops, transcription falls back to whisper.cpp on the Pi and the voice falls back to Piper. The robot keeps talking, more slowly and less expressively, and it is told about its own degradation so it can say so honestly instead of behaving as though nothing changed.'
    ));

    items.push(bodyPara(
        'The subsystem is also responsible for the faculties that make the robot feel present rather than transactional, and these are in scope precisely because they are inseparable from the conversation loop. Identity fuses a face embedding from the current camera frame with a voice embedding from the current utterance, so the robot knows who it is talking to and, critically, whose memory a turn is allowed to be written into. Memory is a SQLite store of durable facts, per-person preferences, rolling session summaries and semantic embeddings for relevance recall. Vision runs continuously from startup rather than on demand, so the scene is already perceived by the time anyone asks about it, and perception is never on the critical path of a reply. Tool use gives the model timers, reminders, web search and explicit remember and recall verbs. A privacy guard decides, per turn, whether an unrecognised voice is answered at all and whether their words are allowed to be stored. A proactive layer lets the robot open a conversation by itself, greeting a person it recognises walking in or asking a question about something new it has noticed. A learning loop turns each exchange into a reward-tagged episode and appends the session to a training corpus for offline fine-tuning.'
    ));

    items.push(bodyPara(
        'Finally, the subsystem exposes a small HTTP control plane on port 8090, running as a daemon thread inside the same process. This is the fusion surface the AF-1 application uses. It matters that it lives inside the process rather than beside it: a push-to-talk turn arriving over HTTP drives the same conversation history, the same memory store, the same tool registry, the same voice and the same speaker as a turn spoken into the microphone. There is one brain, and the network is simply another way into it.'
    ));

    // 1.1
    items.push(subHeading('1.1', 'In Scope'));
    items.push(bodyPara(
        'This document covers the ZERO process in full: the four-state conversation loop and its transition rules; wake-word detection, voice-activity endpointing and the semantic hold that keeps the robot listening through a pause in the middle of a thought; speech-to-text in both its remote and local forms, including the speculative transcription that starts work at the first pause rather than waiting for the endpoint; the language model, its persona prompt, its history window and the rolling compaction that keeps long sessions bounded; text-to-speech across the Piper, Fish and Orpheus engines together with the orchestrator that translates one shared cue vocabulary into whatever the active engine understands; barge-in, which lets a person interrupt the robot mid-sentence and have the interrupting words carried into the next turn so they are never repeated.'
    ));
    items.push(bodyPara(
        'It also covers the perception and state faculties: the always-on camera loop, open-vocabulary and closed-vocabulary object detection, tracking, colour naming, scene phrasing, taught objects, the live world-state model and its surprise gate; face and voice identity, provisional guest clustering for unfamiliar speakers, and diarisation across a multi-speaker conversation; affect estimation and the cross-turn mood it feeds; the SQLite memory schema, embeddings, preferences, consolidation and erasure on request; the tool registry and router; the privacy guard and its visible indicator; the proactive policy, its triggers and its adaptive cooldowns; the episode and reward machinery and the corpus export that feeds fine-tuning. On the GPU side it covers the servers this subsystem calls and owns the client contract for: the Whisper server, the Orpheus voice server and the vision server with its depth and vision-language components. It covers config.yaml in full, the config.local.yaml override mechanism, the systemd units, the tunnel and health-check scripts, and the HTTP control plane contract.'
    ));

    // 1.2
    items.push(subHeading('1.2', 'Out of Scope'));
    items.push(bodyPara(
        'Nothing below the neck belongs to this document. Locomotion, balance, the arms and end effectors, and any motion planning or safety interlock are covered by the relevant motion-control document and are not reachable from this subsystem. The chest Pi 4B and the Nano compute nodes are named here only where ZERO exchanges messages with them; their own processes, boot order and firmware are documented separately. Power delivery, thermal design, mechanical assembly and teardown are outside this document entirely and live in the Engineering Corpus and teardown volumes.'
    ));
    items.push(bodyPara(
        'The AF-1 application is out of scope as a product. This document defines and owns the HTTP contract the application calls, and stops at that boundary. The application\'s own interface, its voice picker, its state handling and its packaging are documented by the application team. Provisioning of the GPU node is likewise out of scope: this document specifies what the servers must expose and how ZERO behaves when they are unreachable, but the machine\'s operating system, drivers and physical hosting are not described here. Model training is out of scope beyond the point where this subsystem writes the corpus; the fine-tuning pipeline that consumes that corpus is documented on its own. Cloud or hosted language-model providers are out of scope by design, because the subsystem is built to run without internet access apart from the optional web-search tool.'
    ));

    // Figure 1 (scope boundary)
    items.push(bodyPara(
        'Figure 1 draws the boundary as built: the two machines this document specifies, the single tunnel that joins them, the one port that admits anything from outside, and the neighbouring subsystems that are named but not described here.'
    ));
    items.push(...insertFigure(1,
        'FIGURE 1 · ZERO SCOPE BOUNDARY',
        'Full line weight marks what this document specifies. The ghosted blocks belong to other documents and are named here only where ZERO exchanges messages with them. The subsystem is reachable from outside at exactly one point, the control plane on port 8090.'
    ));

    // 1.3
    items.push(subHeading('1.3', 'Intended Readers'));
    items.push(bodyPara(
        'This document is written for four readers, and each of them should be able to stop at a different depth. An engineer joining the voice stack should be able to read sections 1 through 4 and understand the loop well enough to trace a single utterance from microphone to speaker, then use sections 5 and 8 to get a working machine. An operator or field technician should be able to work almost entirely from sections 7, 11 and 12, which cover what breaks, what the logs are saying and exactly which commands restart which box. An integration engineer building against the robot from the AF-1 application needs section 3, which is the authoritative interface contract, and section 7 for the failure semantics they must handle. A technical reviewer conducting due diligence should find sections 9, 10, 13 and 14 sufficient: what the system costs in compute and power, what it does with personal data, what it depends on and under which licences, and what is still open. Everything in this document is written against the repository and commit named on the cover page. Where behaviour is configurable, the exact config.yaml key is given rather than described, so that a claim in the prose can always be checked against a value in the file.'
    ));

    return items;
}

function section2() {
    const items = [];
    items.push(new Paragraph({ children: [new PageBreak()] }));
    items.push(sectionHeading('2', 'Architecture'));

    items.push(bodyPara(
        'ZERO is arranged around a single organising rule: there is one conversation loop, it runs on one thread, and nothing is allowed to block it. Every faculty that could take an unpredictable amount of time, which in practice means every faculty that touches a model, a network socket or a disk, runs on a thread of its own and either publishes its result to a place the loop can read at leisure or is joined under an explicit timeout. When one of them is slow, the loop does not wait for it. When one of them fails, the loop does not stop. This is why the architecture looks the way it does, and almost every structural decision in this section follows from it.'
    ));
    items.push(bodyPara(
        'The system spans two machines. The Raspberry Pi 5 in the AF-1 head runs the ZERO process itself and keeps all the work that is real-time or physically bound to the robot: capturing the microphone, detecting the wake word, deciding when a person has stopped speaking, reading frames off the camera, detecting objects, tracking them, and pushing audio to the speaker. A separate GPU node runs the model servers: speech recognition, expressive speech synthesis, the language model, depth estimation, the vision-language model, and the embedding services for faces, voices and objects. The two are joined by exactly one SSH tunnel that carries five forwarded ports. Nothing is exposed to the internet, and there is no second link to fall back to, which is why the failure behaviour of that tunnel is treated as an architectural concern rather than an operational one.'
    ));

    // Figure 8 (system data flow) - master sheet at start of section
    items.push(bodyPara(
        'Figure 8 is the whole subsystem on one sheet, and the rest of this section works through it a zone at a time: the two machines and the boundary between them, the threads that populate the middle zone, the tiering that keeps the SEE row affordable, and the fallback arrangement that decides which side of the tunnel boundary any given faculty is actually running on.'
    ));
    items.push(...insertFigure(8,
        'FIGURE 8 · SYSTEM DATA FLOW',
        'Two dashed boundaries divide the sheet: above the upper one is the GPU node, below the lower one is disk, and between them is the single Pi process. The SEE row runs continuously and independently of the others, which is why a question about the room can be answered without a pause. Tapered paths carry one-way data; accent cables are round trips across the tunnel.'
    ));

    // 2.1
    items.push(subHeading('2.1', 'The Two Machines'));
    items.push(bodyPara(
        'On the Pi, everything lives inside one Python process started as python -m zero.main. That process holds the state machine, the audio devices, the camera, the memory database, the tool registry, the identity registry and the HTTP control plane. It is deliberately a single process rather than a set of cooperating services, because every faculty in the system needs cheap access to the same conversation history, the same memory store and the same speaker identity, and paying inter-process serialisation costs on the reply path would show up directly as latency the user can hear.'
    ));
    items.push(bodyPara(
        'On the GPU node the arrangement is the opposite: several independent server processes, each owning one model, each addressable over HTTP, each restartable without disturbing the others. The Whisper server on port 9000 runs large-v3-turbo through faster-whisper and answers a POSTed WAV with a transcript. The Orpheus server on port 9100 runs a 3B speech model through vLLM and answers text with 24 kHz audio. The vision server on port 8000 is a FastAPI application exposing a health probe, a /facts endpoint that turns a frame plus its detections into per-object distance and bearing using Depth Anything V2 with camera intrinsics, and an /analyze endpoint that runs those facts and then a vision-language model grounded in them. Ollama on port 11434 serves the conversational model, which as configured is gemma4:latest, an 8B model at roughly 100 tokens per second on that hardware. A SearXNG instance on port 8080 backs the web-search tool.'
    ));
    items.push(bodyPara(
        'The tunnel is opened from the Pi by scripts/pi_tunnel.sh and maintained by autossh, which rebuilds it within roughly 45 seconds of a stall. It forwards local 9000, 9100, 8000 and 8080 straight through to the same ports on the node. The one asymmetry is Ollama: local port 11435 forwards to remote 11434, so that the Pi\'s own Ollama, if one is installed for local fallback, can keep 11434 for itself without a collision. The config reflects this, with llm.host pointing at http://127.0.0.1:11435. It is worth knowing that this offset exists, because a tunnel that appears healthy on 9000 while llm.host has been edited to 11434 produces a system that transcribes perfectly and then cannot think.'
    ));
    items.push(bodyPara(
        'Both boxes are intended to run under systemd in production. The units in scripts/systemd/ cover the tunnel and the ZERO process on the Pi and the model servers on the node. The shell script scripts/run_gpu_servers.sh exists for interactive use and is idempotent: it checks whether each port is already listening and skips anything that is up, launching what is missing under setsid nohup so it survives the shell that started it. That script also sets OLLAMA_FLASH_ATTENTION and OLLAMA_KV_CACHE_TYPE=q8_0, which halves the KV cache footprint of the language model. This is a memory-pressure decision rather than a speed one. The card is shared by the conversational model, Whisper, Orpheus, the detector, CLIP and the face models, and when it fills, the chat model gets evicted and the next reply pays a multi-second reload. Quantising the cache is what keeps it resident.'
    ));

    // 2.2
    items.push(subHeading('2.2', 'The Process Set on the Pi'));
    items.push(bodyPara(
        'The Pi process is best understood as one loop plus two kinds of thread. Persistent threads start once and run for the life of the process. Transient threads are created for a single turn, or a single background job, and are gone within seconds. Every thread in the system is a daemon thread, so nothing can keep the process alive after the main loop exits.'
    ));
    items.push(bodyPara(
        'The persistent set begins with the PortAudio callback thread inside MicCapture, which is the only thread that touches the microphone. It delivers 30 ms frames of 16 kHz mono audio onto a bounded queue, and every downstream audio consumer, the wake word and the endpointer alike, pulls from that one queue. Having a single capture rather than one per consumer is what stops the stages fighting over the device. That thread also carries two pieces of resilience worth naming here: a software gain multiplier, because some USB microphones capture too quietly for the wake model to fire, and an automatic resample path for devices that refuse 16 kHz and can only open at 44.1 or 48 kHz. The queue is also where the echo guard lives: while ZERO is thinking or speaking, capture is paused rather than drained downstream, so the robot cannot transcribe its own voice off the speaker.'
    ));
    items.push(bodyPara(
        'The camera has the same shape. CameraStream runs its own thread doing nothing but pulling frames, and the Eyes thread consumes them. Keeping the grab separate from the perception work means a slow detection pass cannot cause frames to back up inside the driver. Above Eyes sit three more persistent threads, each of which exists to keep expensive work off the loop: the Narrator, which periodically asks the vision-language model to describe the scene; the SurpriseGate, which scores world events by prediction error and turns the unexpected into stored episodes; and the proactive TriggerSource, which watches for a recognised person arriving, a curiosity question worth asking, or an idle moment suitable for memory consolidation. The control server contributes one more persistent thread, a ThreadingHTTPServer on port 8090, and the optional camera web preview contributes another on 8008, bound to localhost by default so an unauthenticated video stream is never published to the network by accident.'
    ));
    items.push(bodyPara(
        'The transient threads are where the latency work happens, and they are the part of the architecture most worth understanding. When the endpointer notices the person has paused, well before it is willing to declare the utterance finished, it starts a stt-spec thread that transcribes the audio captured so far. That speculative transcript serves two purposes. If the pause turns out to be the real end of the utterance, the transcript is already in hand and the round trip to the GPU has been overlapped with the silence wait rather than added after it. If the pause turns out to be mid-sentence, the transcript is still useful, because the endpointer inspects its last word to decide whether the speaker had actually finished a thought. Either way the work is not wasted. Once the utterance is confirmed, a stt thread runs the final transcription in parallel with identity and diarisation, which need only the audio and not the text. A recall thread searches memory for anything relevant to what was just said, and is joined under a hard 300 ms budget, after which the turn simply proceeds without the recall note rather than becoming slower. An llm-stream thread starts generation in the background so that the model\'s prefill overlaps the spoken filler, and a tts-producer thread synthesises the next sentence while the current one is still playing. A bargein thread watches the microphone for the entire time ZERO is making sound, so speech or a wake word can interrupt it. Two more transient threads run outside the reply path: compaction, which folds trimmed-away turns into a rolling summary, and memory-save, which writes the session to disk after the conversation ends so the robot returns to listening for the wake word immediately instead of going deaf for several seconds. Each active timer or reminder holds a thread of its own until it fires.'
    ));

    // Table 1
    items.push(...buildTable('TABLE 1 · PROCESS SET', TABLE1_HEADERS, TABLE1_ROWS));

    // Figure 3 (turn timing)
    items.push(bodyPara(
        'Figure 3 puts the whole set on one time axis for a single turn, and the shape of the architecture is easier to read there than in any list: almost nothing in the reply path happens after the thing it depends on, because almost everything has been started before it was needed.'
    ));
    items.push(...insertFigure(3,
        'FIGURE 3 · TURN TIMING CHART',
        'Lanes are threads, the axis is one turn. Every overlap in this chart is deliberate. The work that would otherwise sit end to end on the reply path has been moved sideways into the silence, the filler and the preceding sentence.'
    ));

    // 2.3
    items.push(subHeading('2.3', 'Tiered Perception'));
    items.push(bodyPara(
        'Vision is the one faculty that would happily consume the entire machine if it were allowed to, so it is organised into three tiers with a strict cost ceiling on each. The tiering is not an optimisation added later; it is the reason vision can be always-on at all, and always-on is what allows the robot to answer a question about the room without a visible pause while it looks.'
    ));
    items.push(bodyPara(
        'Tier 0 is motion detection, and it runs on every single frame. It is deliberately crude: the frame is downscaled to 160 pixels wide, differenced against the last, and the fraction of changed pixels compared against a threshold of 0.02. It costs almost nothing, it publishes a motion level to the shared WorldState, and its real job is to gate the tier above it. Motion is considered to have stopped only after 15 consecutive still frames, which prevents the gate flapping on noise.'
    ));
    items.push(bodyPara(
        'Tier 1 is object detection with colour naming and tracking. While the scene is moving it runs on every frame. Once the scene goes still it drops to a keepalive pass every 2 seconds, after a 3 second linger at full cadence so that a brief pause in movement does not immediately starve the detector. Above the motion gate sits a second, harder constraint: a duty budget that caps the detector at 60% of loop time regardless of what the motion gate says. The budget wins. This matters because motion and cost are not the same thing; a busy scene can ask for more detection than the Pi can afford, and without the ceiling the perception loop would starve the frame grab and the whole camera would stutter. When a frame is gated out, the previous detections are held as current but the fresh frame is still published, so identity, keyframes and the preview all keep working and the system never looks blind just because it chose not to re-detect.'
    ));
    items.push(bodyPara(
        'Tier 2 is narration, where the vision-language model is asked for a sentence about the scene. It runs at most every 3 seconds and is additionally capped at 20 inferences per minute, and it self-skips while nothing has changed. This is the only vision tier that reaches the GPU on a schedule rather than on demand.'
    ));
    items.push(bodyPara(
        'All three tiers publish into one WorldState object, and that is the read surface everything else uses. The main loop, the proactive watcher and the surprise gate do not call the camera or the detector; they read a snapshot, or wait for a change. Keeping a single published state rather than letting consumers pull from the perception stack directly is what allows the tiers to be re-rated, gated or disabled entirely without any consumer noticing.'
    ));

    // Figure 4
    items.push(bodyPara(
        'Figure 4 shows the three cadences against one another over a period in which the room goes from moving to still, which is the case that makes the gate and the budget do visibly different jobs.'
    ));
    items.push(...insertFigure(4,
        'FIGURE 4 · PERCEPTION TIER CADENCE',
        'The shaded band is motion. Tier 1 follows it, then decays to keepalive after the linger expires. The duty ceiling is independent of motion and overrides it, which is what protects the frame grab when the scene is busier than the Pi can afford.'
    ));

    // 2.4
    items.push(subHeading('2.4', 'The Factory and the Fallback Chain'));
    items.push(bodyPara(
        'Every engine in the system is constructed in exactly one file, zero/factory.py. It maps the engine names in config.yaml to concrete classes and nothing else in the codebase does that mapping, which is what makes an engine swap a configuration change rather than a code change. The factory also owns a convention that matters more than it first appears: a faculty that is disabled, or whose model files are missing, or whose optional dependencies are not installed, returns None rather than raising. The main loop is written to expect None from every optional faculty. This is why a Pi with no camera runs voice-only, a Pi with no face model runs voice-only identity, and a machine with neither identity model runs anonymously, all without a code path dedicated to each case.'
    ));
    items.push(bodyPara(
        'Layered on top of that is the fallback chain, which is the architectural expression of the local-fallback commitment stated in Section 1. Each faculty that prefers the GPU is wrapped in a fallback object that holds the remote implementation and a builder for the local one. The pattern appears six times: FallbackSTT over the remote Whisper client with whisper.cpp behind it, FallbackTTS over Orpheus with Piper behind it, FallbackDetector over the server-side detector with local YOLO behind it, and FallbackSpeaker, FallbackFace and FallbackObjectEmbedder over their server-side embedders with local models behind them. In every case the local implementation is built lazily, on the first remote failure, rather than at startup. That laziness is deliberate: loading whisper.cpp and a Piper voice into an 8 GB Pi that is also holding the detector and the identity models would cost memory that is only needed if the tunnel ever drops.'
    ));
    items.push(bodyPara(
        'The last piece of the chain is that degradation is visible rather than silent. Each fallback wrapper exposes a degraded flag, and the main loop polls those flags once per turn. On the transition into or out of degradation it appends a note to the outgoing prompt telling the model, in effect, that its fast hearing is down and it is transcribing locally, or that it is speaking through its backup voice. The robot can then say so when asked why it is slow, instead of behaving as though nothing has changed. Treating self-knowledge of failure as part of the architecture, rather than as a logging concern, is what keeps a degraded system honest to the person in front of it.'
    ));

    // Figure 5
    items.push(bodyPara(
        'Figure 5 draws the pattern once rather than six times, since the six faculties differ only in which models sit on either side of the switch.'
    ));
    items.push(...insertFigure(5,
        'FIGURE 5 · FALLBACK CHANGEOVER',
        'Six faculties, one pattern. The local contact is drawn dashed because it does not exist until the first remote failure builds it. The flag bus is what lets the robot tell the person it is running on backup rather than simply behaving worse.'
    ));

    // 2.5
    items.push(subHeading('2.5', 'The Control Plane Inside the Process'));
    items.push(bodyPara(
        'The HTTP control plane deserves its own subsection because its placement is the single most consequential integration decision in the system. It runs as a daemon thread inside the ZERO process, not as a sibling service. The consequence is that a push-to-talk turn arriving from the AF-1 application over HTTP is handled by the same Conversation object, the same SQLite memory, the same tool registry, the same voice and the same physical speaker as a turn spoken into the microphone. There is no synchronisation problem between a network brain and a local brain, because there is only one brain.'
    ));
    items.push(bodyPara(
        'The server exposes a health probe that answers immediately at startup, before the model is warm, with a ready flag that flips true once the language model has been pinned in memory by the warmup call. This lets the application distinguish a Pi that is booting from a Pi that is broken. Beyond health it offers a status endpoint reporting state, the last external turn and the degradation flags; a say endpoint that speaks a line on the Pi speaker without involving the model; a turn endpoint that accepts raw audio in any container ffmpeg can decode and runs a complete brain turn; a text turn endpoint for typed input; and a control endpoint whose only action is to end an open conversation. External turns are serialised against each other by one lock and kept off the native loop\'s think and speak phase, so the microphone and the network cannot both be driving the state machine at once.'
    ));
    items.push(bodyPara(
        'CORS is deliberately wide open, because the Tauri application fetches from the Rust side and needs none while browser development needs a wildcard. The server binds 0.0.0.0 by design, which means it is reachable by anything on the local network. That is a real attack surface and is treated as such in Section 10 rather than hidden here.'
    ));

    return items;
}

function section3() {
    const items = [];
    items.push(new Paragraph({ children: [new PageBreak()] }));
    items.push(sectionHeading('3', 'Interfaces & Data Contracts'));

    items.push(bodyPara(
        'An interface, for the purposes of this document, is anything that crosses the boundary drawn in Figure 1. That includes the HTTP control plane the AF-1 application calls, every request ZERO sends up the SSH tunnel to a model server, the physical devices the Pi opens, the GPIO pin that drives the recording indicator, and every file the system writes to disk. Internal function calls between modules are not interfaces and are not listed here; they are covered by Section 4. The distinction matters because everything in this section is a contract that something outside the subsystem depends on, and therefore cannot be changed without changing something else at the same time.'
    ));
    items.push(bodyPara(
        'Three conventions run through all of the contracts below and are worth stating once rather than repeating in every row. First, all audio crossing any boundary is mono, 16-bit signed PCM, wrapped in a self-describing WAV container, so the sample rate travels with the data rather than being agreed out of band. The one exception is speech coming back from Orpheus, which is 24 kHz because that is what its vocoder produces, and which the Pi resamples on receipt. Second, all images crossing any boundary are JPEG-encoded and then base64-encoded into a JSON string field named image_jpeg_b64, at a quality set by vision.gpu.jpeg_quality and defaulting to 80. Third, every embedding stored on disk is a float32 array written as a raw little-endian blob, always accompanied by an integer dim column in the same row. Readers compare dim against their own model\'s dimension and skip mismatched rows rather than failing, which is what allows an embedding model to be swapped without invalidating the database.'
    ));
    items.push(bodyPara(
        'There is a fourth convention that is easy to miss and causes real confusion when missed. Throughout the system, a person identifier is a signed integer whose sign carries meaning. A positive value is an enrolled person in the identity registry. A negative value is a provisional guest, minted by clustering an unfamiliar voice and deliberately kept separate so that one stranger\'s words never merge with another\'s. A null value is an anonymous turn, which is what text mode produces and what happens when identity is disabled or the speaker could not be placed. Every store that carries person_id, and every corpus record, uses this convention, and code that treats the field as an opaque key will silently mix guests with people.'
    ));

    // 3.1
    items.push(subHeading('3.1', 'The Control Plane'));
    items.push(bodyPara(
        'The control plane listens on TCP 8090, bound to 0.0.0.0 by control.host, and is the only inbound interface the subsystem exposes. All responses are JSON unless stated otherwise. A cross-origin preflight is answered with 204 and the permissive header set described at the end of this subsection. Any request body larger than 32 MB is rejected, which is generous for the intended use since a minute of WebM Opus is well under it. Unhandled exceptions inside a handler are caught, logged with a traceback, and returned as HTTP 500 with the exception text truncated to 200 characters, so a malformed request can never take down the process that is also holding the conversation.'
    ));
    items.push(bodyPara(
        'GET /health, also reachable as GET /zero/health, returns {ok, service, state, ready}. The service field is the constant string zero-control. The state field is the current conversation state as its lowercase value, one of idle, listening, thinking or speaking. The ready field is the one that matters operationally: it is false from process start until the language model has been pinned in memory by the warmup call, and true afterwards. A client that treats a reachable-but-not-ready Pi as broken will report false failures during the first few seconds after a restart.'
    ));
    items.push(bodyPara(
        'GET /zero/status returns the state, the last external turn, and the current degradation flags. This is the endpoint that lets the application show that the robot is running on its backup voice without having to ask it.'
    ));
    items.push(bodyPara(
        'POST /zero/say takes {text, voice?} and speaks the line on the Pi speaker without involving the language model or writing anything to memory. The text is truncated to 500 characters. An empty or missing text returns 400 with {ok:false, error:"empty-text"}. The optional voice is an Orpheus speaker name such as leo, tara or draco, and it overrides the configured default for that request only, leaving ZERO\'s own default untouched.'
    ));
    items.push(bodyPara(
        'POST /zero/turn is the push-to-talk path. The request body is raw audio bytes, not JSON and not multipart, in any container ffmpeg can decode, which in practice means WebM, OGG, WAV or MP4. Options travel as query parameters, ?voice= and ?person_id=. The server shells out to ffmpeg to convert the body to mono float32 at the configured sample rate. A body that decodes cleanly runs a complete brain turn, is spoken on the Pi speaker, and returns {ok, heard, reply}, where heard is the transcript and reply is what was said. An empty body returns 400. A decode failure returns 422 with the tail of ffmpeg\'s stderr in the error field, which is deliberate: a caller sending an unsupported container needs to see why.'
    ));
    items.push(bodyPara(
        'POST /zero/turn_text takes {text, voice?, speak?, person_id?} and runs the same brain turn from typed input, with text truncated to 1200 characters and speak defaulting to true. POST /zero/control accepts {action:"sleep"} and ends the open conversation; any other action returns 400 naming the unknown action. Anything else returns 404.'
    ));
    items.push(bodyPara(
        'External turns are serialised against each other by a single lock and are held off the native loop\'s think and speak phase, so the microphone and the network cannot both drive the state machine at once. The person_id supplied by the application defaults to control.person_id, which ships as 1, meaning AF-1 turns are attributed to the logged-in operator rather than to an anonymous speaker.'
    ));
    items.push(bodyPara(
        'CORS is deliberately unrestricted: Access-Control-Allow-Origin: *, methods GET, POST, OPTIONS, headers Content-Type. The Tauri application fetches from the Rust side and needs no CORS at all, but browser-based development does, and no authentication is implemented at this layer. Combined with the 0.0.0.0 bind, this means anything on the local network can make the robot speak. That is a deliberate trade for a LAN-only device and it is assessed properly in Section 10, not defended here.'
    ));

    // 3.2
    items.push(subHeading('3.2', 'The Model Server Contracts'));

    // Figure 6 (port map)
    items.push(bodyPara(
        'Figure 6 lays the five forwards out as a panel, because the single asymmetry in the map is the thing most likely to be misconfigured and the easiest to see when drawn rather than described.'
    ));
    items.push(...insertFigure(6,
        'FIGURE 6 · TUNNEL PORT MAP',
        'Five forwards on one tunnel. Four are straight through. Ollama alone shifts from local 11435 to remote 11434, which leaves 11434 free on the Pi for a local fallback model.'
    ));

    items.push(bodyPara(
        'Every one of these is reached through the SSH tunnel at a local port, so from ZERO\'s point of view they are all on 127.0.0.1. The port offset described in Section 2.1 applies: Ollama alone is reached at local 11435.'
    ));
    items.push(bodyPara(
        'The Whisper server accepts POST /transcribe with a raw WAV body and Content-Type: audio/wav, and returns {text}. An optional ?language= query parameter is appended by the client when stt.language is set, accepting a language code such as sw, or the literal auto to let Whisper detect, which is the right setting for code-switched speech. The client raises rather than returning an empty string on failure, and that distinction is load-bearing: the fallback wrapper must be able to tell a dead tunnel from a silent room.'
    ));
    items.push(bodyPara(
        'The Orpheus server accepts POST /tts with {text, voice} and returns a WAV at 24 kHz. A streaming variant at /tts_stream is derived by the client from the configured URL by simple substring replacement, which means a non-standard path in tts.orpheus.url will produce a stream URL that does not exist. GET /health returns {ok} reflecting whether the model has loaded. The client keeps one HTTP session alive across sentences rather than reconnecting, and retries a connection failure once on a fresh connection so that a server restarted between turns recovers rather than erroring. Before sending, the client rewrites ZERO\'s shared cue vocabulary into Orpheus\'s native tags, mapping [laughs] to <laugh>, [sighs] to <sigh> and so on, rendering [hmm] and [pause] as text the model performs naturally, and stripping any cue it does not recognise.'
    ));
    items.push(bodyPara(
        'The vision server exposes GET /health, which reports CUDA availability and VRAM and is what proves the tunnel is alive; POST /facts, which takes an AnalyzeRequest and returns an AnalyzeResponse whose reply is empty because ZERO\'s own model writes the spoken answer; and POST /analyze, which runs the same facts and then a vision-language model grounded in them, returning a populated reply.'
    ));
    items.push(bodyPara(
        'The shared wire types are Pydantic v2 models. A Detection carries label as a string, bbox as exactly four floats in [x, y, w, h] in pixels of the frame that was sent, confidence constrained to the range 0 to 1, and an optional color string from HSV naming. An AnalyzeRequest carries image_jpeg_b64, a list of Detection, a question string and a history list of {role, content} dictionaries. A SceneFact carries label, optional color, optional distance_m constrained to be non-negative, and an optional bearing which is one of the coarse strings left, center or right. An AnalyzeResponse carries a list of SceneFact and a reply string.'
    ));
    items.push(bodyPara(
        'These models exist in two files, one on each side of the tunnel, because the two nodes do not share a filesystem. The two copies have drifted. In the Pi copy at zero/vision/schemas.py, both AnalyzeRequest.question and AnalyzeResponse.reply have empty-string defaults and are therefore optional. In the server copy at server/vision/shared/schemas.py, both are declared required. Today this is benign, because the Pi serialises its own model and so always emits both keys even when they are empty, but the contract is no longer single-sourced and the next edit to either file can diverge further without anything failing loudly. The server copy\'s docstring compounds this by instructing the reader to edit a canonical file and run python shared/sync_schemas.py, and by naming a mirror at pi/shared/schemas.py. Neither that script nor that path exists in this repository. This is recorded as an open item in Section 14.'
    ));

    // 3.3
    items.push(subHeading('3.3', 'The Perception Offload'));
    items.push(bodyPara(
        'Separate from the vision endpoints above, the GPU node serves a set of embedding and detection endpoints under /perceive/ on the same port 8000, reached through a client with its own timeouts: 5 seconds by default and 10 seconds for detection. Every one of them raises a RuntimeError naming the path on any failure, again so that the fallback wrappers can distinguish an unreachable server from an empty result.'
    ));
    items.push(bodyPara(
        'POST /perceive/detect takes {image_jpeg_b64} and returns {detections: [...]} in the Detection shape. POST /perceive/face takes {image_jpeg_b64, max_faces}, defaulting to three faces, and returns {faces: [{embedding: [float, ...], bbox: [...]}, ...]}; the client discards any face whose embedding array is empty. POST /perceive/speaker is the one endpoint in this group that does not take JSON: the body is a raw WAV with Content-Type: audio/wav, built from the utterance at the capture sample rate, and the response is {embedding: [float, ...]}. POST /perceive/embed_object takes {image_jpeg_b64} holding a crop rather than a full frame and returns {embedding: [float, ...]}.'
    ));

    // 3.4
    items.push(subHeading('3.4', 'Local Device Interfaces'));
    items.push(bodyPara(
        'The microphone is opened through PortAudio at the index given by audio.input_device, and the pipeline contract downstream of it is fixed: mono, 16-bit, 16 kHz, in blocks of audio.block_ms milliseconds, which ships as 30 and therefore yields 480 samples per block. Two adaptations sit inside that contract. A software gain from audio.input_gain multiplies each block before int16 conversion, because some USB and webcam microphones capture too quietly for the wake model to fire. And when a device rejects 16 kHz outright, which is common for cheap USB dongles that only support 44.1 or 48 kHz, the stream is opened at the device\'s native rate and each block is resampled down, so every consumer still sees the same frames it would otherwise.'
    ));
    items.push(bodyPara(
        'The speaker is opened at audio.output_device. The camera is opened at vision.camera.index and configured for 640 by 480 at a requested 30 frames per second in MJPG, which is deliberately small so that detection keeps up on the Pi.'
    ));
    items.push(bodyPara(
        'The recording indicator is a single GPIO pin named by privacy.indicator_gpio_pin, driven through gpiozero. It is lit for the listening, thinking and speaking states and dark for idle. With no pin configured, or with gpiozero unavailable, it degrades to state-transition log lines so that the same information is still observable through journalctl. This is the only physical output the subsystem drives.'
    ));
    items.push(bodyPara(
        'The optional camera preview serves HTTP on vision.preview_port, which ships as 8008, bound to vision.preview_host. That default is 127.0.0.1 on purpose: the preview is an unauthenticated video stream, and publishing it to the network requires an explicit change to 0.0.0.0.'
    ));

    // Table 2
    items.push(...buildTable('TABLE 2 · INTERFACE CONTRACTS', TABLE2_HEADERS, TABLE2_ROWS));

    // 3.5
    items.push(subHeading('3.5', 'On-Disk Contracts'));
    items.push(bodyPara(
        'Six SQLite databases and two flat files make up the persistent state. All of them are created on first use, all of them are opened with check_same_thread=False because several threads write, and the episode store additionally runs in WAL mode so that writers never block readers.'
    ));
    items.push(bodyPara(
        'zero_memory.sqlite holds one table, memories, with columns id, person_id, layer, key, value, importance defaulting to 5.0, emb as a float32 blob, emb_dim, created_at, last_access, access_count and protected. There is an index on (layer, person_id). The protected flag marks rows that the storage cap must never prune, which is what keeps the durable per-person last-conversation record alive when ordinary facts are being aged out. This store also carries a one-time legacy migration that folds rows from an older flat schema, with tables named memory and episodes, into the current shape and then drops them, so an old database upgrades in place on first open.'
    ));
    items.push(bodyPara(
        'zero_identity.sqlite holds people, with id, a name that is unique and case-insensitive, and created_at; and embeddings, with id, person_id, kind, dim, a vec blob and created_at, with a foreign key to people. The kind column is what allows face and voice vectors to live in one table.'
    ));
    items.push(bodyPara(
        'zero_guests.sqlite holds guest_samples, with id, guest, dim, vec and created_at. Guest identifiers are negative by construction: a new guest takes the minimum of minus one and one below the lowest existing identifier. The store caps samples per guest and total guests, dropping the least recently heard.'
    ));
    items.push(bodyPara(
        'zero_episodes.sqlite holds episodes, with id, v recording the schema version at write time, ts, kind drawn from turn, scene, proactive and action, person_id, a payload JSON string defaulting to {}, reward where null means untagged, surprise, and consolidated_at. It carries two indexes, one on (kind, ts) and a partial index on unconsolidated rows. Alone among the stores it uses a real migration mechanism driven by PRAGMA user_version, applying each migration step in its own transaction. Rewards are clamped to the range minus one to one on write.'
    ));
    items.push(bodyPara(
        'zero_curiosity.sqlite holds questions, with id, a unique source_key that prevents the same observation queuing twice, text, priority, person_id, created_at and asked_at. zero_objects.sqlite holds objects, with id, a case-insensitive name, person_id, dim, vec and created_at, which is where a taught object binds a name to an embedding.'
    ));
    items.push(bodyPara(
        'The interaction corpus is newline-delimited JSON at data/corpus/interactions.jsonl, appended under a lock so records never interleave. One record is written per speaker per session, holding that speaker\'s turns, their identifier under the sign convention described above, and a timestamp. Because the session has already been split by speaker before it is written, one person\'s speech never contaminates another\'s training data. The enrolled voiceprint is a NumPy array at the path in voiceid.profile_path, and the surprise predictor keeps its running statistics as JSON at world.surprise.stats_path.'
    ));

    // Figure 7 (store map)
    items.push(bodyPara(
        'Figure 7 shows the eight files together with the columns that carry the conventions, since it is the conventions rather than the individual tables that a reader has to hold in mind.'
    ));
    items.push(...insertFigure(7,
        'FIGURE 7 · PERSISTENT STORE MAP',
        'Every store that carries a person identifier obeys the same sign convention. Every embedding column is float32 little-endian with its dimension in the adjacent column, which is what allows an embedding model to change without invalidating the file.'
    ));

    // Table 3
    items.push(...buildTable('TABLE 3 · PERSISTENT STORES', TABLE3_HEADERS, TABLE3_ROWS));

    // 3.6
    items.push(subHeading('3.6', 'The Internal Event Bus'));
    items.push(bodyPara(
        'The event bus is the one internal mechanism documented here, because it is how anything that is not the main loop gets the robot to speak, and because its overflow behaviour is a contract rather than an implementation detail. An Event carries kind, text already phrased for speech, created_at, an optional person_id, and a meta dictionary. The queue holds 64 events. Posting to a full queue drops the event and returns false rather than blocking, on the reasoning that a lost nudge is better than a wedged timer thread. The main loop drains the bus only at safe moments, meaning the idle wake-wait and turn boundaries, so an announcement can never talk over a reply in progress. One meta key is load-bearing: open_conversation, which tells the loop that the announcement expects an answer and that it should begin listening rather than returning to idle.'
    ));

    return items;
}

function section4() {
    const items = [];
    items.push(new Paragraph({ children: [new PageBreak()] }));
    items.push(sectionHeading('4', 'Behaviour & Logic'));

    items.push(bodyPara(
        'This section describes what the system actually decides, and on what evidence. Two habits run through all of it and are worth naming before the detail. The first is that every judgement the robot makes has an explicit numeric threshold that lives in config.yaml rather than in the code, so behaviour can be tuned in the field without a deployment. The second is that a faculty which cannot reach a confident answer degrades to a weaker one rather than failing: an unrecognised voice still gets a reply, it just does not get to write into anyone\'s memory; a scene that cannot be re-detected still publishes its last detections rather than reporting an empty room. Almost every rule below is shaped by one of those two habits.'
    ));

    // 4.1
    items.push(subHeading('4.1', 'The State Machine'));
    items.push(bodyPara(
        'Four states exist: IDLE, LISTENING, THINKING and SPEAKING. The legal moves between them are declared as data in zero/state.py and checked on every transition. IDLE may only go to LISTENING. LISTENING may go to THINKING, or back to IDLE when nothing was said. THINKING may go to SPEAKING, to IDLE, or back to LISTENING, that last case covering an empty reply where the robot stays in the conversation rather than ending it. SPEAKING may go to IDLE or back to LISTENING, the latter being barge-in.'
    ));
    items.push(bodyPara(
        'An illegal transition is logged as a warning and then performed anyway. That is a deliberate choice: a wiring bug should be loud in the logs during development but must never crash a conversation in front of a person. The transition helper also returns early when the destination equals the current state, which is what allows the filler and the reply to both request SPEAKING without churn.'
    ));
    items.push(bodyPara(
        'The most important structural fact about the loop is not visible in the state list. After the wake word admits the first turn, the cycle is LISTENING to THINKING to SPEAKING and back to LISTENING, repeatedly, with no wake word between turns. IDLE is left behind until a stop phrase or a silence long enough to trip conversation.sleep_timeout_ms returns the robot to it. A reader who models this as a four-state ring will get the behaviour wrong.'
    ));

    // 4.2
    items.push(subHeading('4.2', 'Waking'));
    items.push(bodyPara(
        'Wake detection runs openWakeWord over every 30 ms frame in IDLE, comparing its score against wake.threshold. The detector is reset and the microphone drained whenever the robot enters or re-enters the wait, which matters because the announcement path can speak while IDLE and the system must not hear its own voice as a wake attempt.'
    ));
    items.push(bodyPara(
        'A conversation can also be opened without a wake word at all. When a proactive event carries the open_conversation flag in its metadata, the idle wait returns as though woken, and the spoken opener is seeded into the fresh history as the first assistant turn so that the model knows it said it. This is what allows the robot to greet someone walking in and then listen for an answer, rather than speaking into the air and going back to sleep.'
    ));

    // 4.3
    items.push(subHeading('4.3', 'Deciding When a Person Has Finished'));
    items.push(bodyPara(
        'Endpointing is the single most behaviour-defining piece of logic in the system, because getting it wrong is immediately obvious to a user: too eager and it interrupts, too patient and it feels unresponsive.'
    ));
    items.push(bodyPara(
        'Two backends exist and the distinction between the code\'s default and the shipped configuration matters. The factory defaults to Silero VAD, but config.yaml explicitly selects webrtc with an aggressiveness of 2, and that is what runs on the deployed robot. The reason is recorded in the config itself: Silero is sharper on clean audio but stricter, and it was missing speech from a quiet microphone. Anyone reading the factory in isolation will conclude Silero is active; it is not.'
    ));
    items.push(bodyPara(
        'Silero remains available and is worth understanding because it is the fallback plan for a noisy environment. It runs through ONNX Runtime rather than torch, which is what keeps the Pi torch-free. Silero v5 requires exactly 512-sample windows at 16 kHz while the audio pipeline delivers 480-sample frames, so the endpointer buffers samples and feeds whole 512-sample windows in order, carrying the model\'s recurrent state between them. Its ONNX interface is smoke-tested at construction with a zero-filled window, so a version or API mismatch raises there and the factory falls back to webrtcvad, rather than producing an endpointer that silently never detects speech. That failure mode, a microphone that appears alive but is functionally deaf, is the reason the smoke test exists. If Silero is selected, vad.silero_threshold ships at 0.3 rather than the library default of 0.5, again to accommodate a quiet microphone.'
    ));
    items.push(bodyPara(
        'Starting and continuing an utterance use deliberately different rules. To start, a frame must pass the VAD and exceed vad.energy_threshold in RMS, which rejects quiet background onsets and distant voices. To continue, only the VAD is consulted. That asymmetry is the fix for a real fragmentation bug: quiet syllables and short pauses inside a sentence were failing the energy test and splitting one utterance into several. A short pre-roll of vad.speech_pad_ms is retained and prepended when speech starts, so the first word is not clipped.'
    ));
    items.push(bodyPara(
        'The endpoint fires after vad.silence_ms of trailing silence, expressed internally as a count of blocks. Two modifiers sit on top. The first is an adaptive endpoint: if the amount of speech collected so far is less than vad.min_speech_for_fast_end_ms, the required silence is doubled. A slow starter who says "um" and then thinks is not cut off, while a finished sentence still commits at the normal speed. The second is the semantic hold described in 4.4.'
    ));
    items.push(bodyPara(
        'Two outcomes are carefully distinguished. A true idle timeout, meaning no speech at all for conversation.sleep_timeout_ms, returns nothing and ends the conversation. An utterance that was captured but whose average RMS falls below vad.min_utterance_rms is dropped and the endpointer keeps listening with a fresh idle window. This is a proximity gate: the owner\'s voice close to the microphone is loud, a conversation across the room is not, and a stray background blip must not put the robot to sleep mid-conversation. There is also a hard length cap at vad.max_utterance_ms.'
    ));
    items.push(bodyPara(
        'While waiting for speech, the endpointer logs a heartbeat every five seconds. That line is load-bearing for triage and should not be removed: if it ticks but speech is never captured, the problem is VAD or level; if it never ticks at all, frames are not arriving and the microphone stream itself has died. Those are different faults with different fixes, and this is what tells them apart.'
    ));

    // 4.4
    items.push(subHeading('4.4', 'Speculation and the Semantic Hold'));
    items.push(bodyPara(
        'Roughly 180 ms into any silence run, well before the endpoint would fire, the endpointer calls back with the audio captured so far and transcription begins on a background thread. This speculative transcript does two jobs at once.'
    ));
    items.push(bodyPara(
        'Its first job is latency. If that pause turns out to be the real end of the utterance, the transcript is already in hand and the round trip to the GPU has been spent inside the silence wait rather than added after it. The main loop decides whether it may reuse the speculative result by two tests: the speculative audio must be a verbatim prefix of the final utterance, and the amount of audio that arrived afterwards must be within a slack of twice vad.silence_ms plus vad.speech_pad_ms plus 400 ms. A longer tail means speech resumed or the length cap fired, so the speculative text is only a prefix and the full transcription runs instead.'
    ));
    items.push(bodyPara(
        'Its second job is judgement. When the endpoint condition is met, the loop is asked whether the speaker was mid-thought, and the answer is derived from the last word of the speculative transcript. A transcript ending in a conjunction, a preposition, an article or a filler is treated as unfinished, as is one ending in a comma, a colon, a dash or a literal ellipsis, which is what Whisper writes when speech trails off. The word list deliberately excludes words that legitimately end sentences, such as pronouns.'
    ));
    items.push(bodyPara(
        'The critical detail is that this check is tri-state rather than boolean. True means hold for one more silence window. False means commit now. None means the transcript is still in flight, and in that case the endpointer waits a bounded vad.semantic_hold_wait_ms rather than committing. Without that third state the mid-thought check raced the STT round trip and lost: the answer arrived just after the endpoint had already committed, and half-sentences were being shipped to the model. The hold applies at most once per pause; if speech resumes, the hold is cleared and the next pause gets a fresh decision.'
    ));
    items.push(bodyPara(
        'Speculation is skipped in three cases, each for a distinct reason. It is skipped when voice ID is enabled, because nothing may be transcribed before the owner check runs. It is skipped in strict privacy mode for the same reason applied to bystanders. And it is skipped while STT is degraded to the local CPU engine, because speculating there simply runs a slow job twice, once at the pause and once at the endpoint, with no overlap to gain.'
    ));

    // Figure 9 (endpoint ladder)
    items.push(bodyPara(
        'Figure 9 puts these timing rules on a single silence axis, so the order in which each gate fires and the gap between them is immediately visible.'
    ));
    items.push(...insertFigure(9,
        'FIGURE 9 · ENDPOINT DECISION LADDER',
        'One pause, drawn as a silence axis. The bars mark where each rule fires. The tri-state hold is the only branch that can extend the wait, and it is bounded, which is what stops the check racing the transcription it depends on.'
    ));

    // 4.5
    items.push(subHeading('4.5', 'Deciding Who Is Speaking'));
    items.push(bodyPara(
        'Identity fuses two independent signals. When a face match and a voice match name the same person, the score is a weighted sum, w_face times the face cosine plus w_voice times the voice cosine, with the weights normalised to sum to one, and the result must clear identity.fusion.threshold, which ships at 0.50. When the two signals name different people, they do not average. Instead the stronger weighted signal competes alone against its own single-channel threshold, 0.42 for face and 0.45 for voice, both stricter than the fused threshold because a single channel is inherently less reliable. This is what prevents two weak and contradictory matches from manufacturing a confident identification.'
    ));
    items.push(bodyPara(
        'Sessions are owned by voice, not by face. Under identity.session.voice_only, which defaults true, the speaker\'s voice decides whose memories the turn belongs to and the face is treated as perception only. There is a second, stricter gate on top: a turn only credits durable memory when the identity score reaches identity.session.write_min_score, which ships at 0.55. A borderline match still gets a normal conversation, it simply does not write into anyone\'s permanent record. That single threshold is what stops a multi-speaker session cross-contaminating memory.'
    ));
    items.push(bodyPara(
        'An unfamiliar voice is not immediately made into a guest. The voiceprint is held, and only after the transcript proves the turn was real is a provisional guest assigned. Three gates decide "real": at least identity.guests.min_words words, at least identity.guests.min_ms of audio, and at least identity.guests.min_rms in level. These exist because Whisper hallucinates plausible text from near-silence, producing phantom guests and polluting the training corpus. Guest identifiers are negative, and clustering an unfamiliar voice against existing guests keeps different strangers separate.'
    ));
    items.push(bodyPara(
        'Diarisation compares consecutive turns\' voice embeddings and raises a speaker-change note when the cosine falls below perception.diarize.change_threshold. The note is ephemeral and is attached to that turn only.'
    ));
    items.push(bodyPara(
        'One rule about sight deserves emphasis because it is a behaviour users notice. The robot may only claim to see someone when their face is in the current frame. Recognising a voice off-camera produces a different note, explicitly saying it recognises the voice but cannot see the face. Furthermore, the identity note is attached only when recognition changes or when the user asks a visual question. Repeating it every turn fed the model greeting-fodder and it kept re-greeting people mid-topic.'
    ));

    // Figure 10 (identity surface)
    items.push(...insertFigure(10,
        'FIGURE 10 · IDENTITY DECISION SURFACE',
        'Face cosine against voice cosine. The shaded regions are where each verdict is reached. The single-channel bands are stricter than the fused line, which is why two weak but agreeing signals accept while two weak and conflicting ones do not.'
    ));

    // 4.6
    items.push(subHeading('4.6', 'Deciding What Is Being Asked About'));
    items.push(bodyPara(
        'Most turns carry only a cheap text hint about the scene. A turn classified as visual additionally pulls recent keyframes so the multimodal model can actually look. The classifier is deliberately tight in two directions.'
    ));
    items.push(bodyPara(
        'Bare demonstratives such as "this", "that" and "there" are excluded, because they appear in most ordinary sentences. Polysemous verbs such as "see", "look", "watch" and "picture" are excluded as bare words, because in speech they are usually non-visual: "I see", "looking forward to it", "picture this scenario". Those words appear instead inside the phrase list in their genuinely visual forms, such as "what do you see", "take a look" and "look around".'
    ));
    items.push(bodyPara(
        'Beyond the fixed word and phrase lists, the classifier consults the live detections. If the utterance names an object the camera can see right now, the turn is visual. This is what makes "what colour is the cup" work without anyone having to enumerate every possible object: matching is done against the full label and its head noun, so "cell phone" also matches "phone", plus a naive plural, on word boundaries. The label "person" is excluded from this test, because a person is on screen almost always and the word turns up constantly in abstract speech.'
    ));
    items.push(bodyPara(
        'Presence is answered from the detector rather than from the model. Whether a person is in frame is decided by YOLO and stated to the model as fact, because without that grounding the model happily answers "yes, I see you" to an empty room. Three cases are distinguished: the camera never came up, in which case the model is told it is blind and instructed not to invent a scene; the camera is working and there is nobody in frame; and there is a person in frame.'
    ));
    items.push(bodyPara(
        'Spontaneous scene changes are surfaced as an optional note the model may mention or ignore, but never on a turn classified as a question. Answering "which planet did Thanos visit" with "is that a remote on the table?" reads as not listening. Ungated, the changes stay queued and self-expire until a calmer turn.'
    ));

    // 4.7
    items.push(subHeading('4.7', 'Deciding What to Remember and What to Recall'));
    items.push(bodyPara(
        'Durable facts are injected once at conversation start rather than per turn, which keeps the prompt prefix stable and the model\'s cache warm. Per-turn relevance recall is separate and ephemeral.'
    ));
    items.push(bodyPara(
        'Recall ranks candidate memories by an activation score with four factors: activation = relevance x (0.2 + 0.8 x recency) x (importance / 10) x frequency. Relevance is the cosine between the query embedding and the stored embedding when both exist and their dimensions agree, floored at zero; it falls back to keyword overlap when there is no embedder, and to 1.0 when there is no query at all, which reduces the ranking to recency times importance. Recency decays exponentially from last access rather than creation, so recalling something refreshes it.'
    ));
    items.push(bodyPara(
        'The decay parameter deserves a correction that anyone tuning it needs. It is named memory.retrieval.half_life_days and ships at 14, but the implementation computes exp(-age / half_life_seconds), which makes it a time constant, not a half-life. At an age equal to the configured value the recency factor has fallen to 1/e, roughly 0.368, rather than to 0.5. The true half-life is the configured value multiplied by the natural logarithm of 2, so the shipped setting of 14 days is really a half-life of about 9.7 days. The name is misleading rather than the behaviour being wrong, but a reader who sets this expecting half-life semantics will get a decay about 30 percent faster than intended. This is recorded as an open item in Section 14.'
    ));
    items.push(bodyPara(
        'The 0.2 + 0.8 x term puts a floor under the decay so that an old but highly relevant and important fact can still surface. Frequency is 1 + 0.1 x ln(1 + access_count), a deliberately gentle logarithm so that frequently accessed memories are favoured slightly without dominating.'
    ));
    items.push(bodyPara(
        'Two constraints sit around it. Recall runs on its own thread under a hard budget of memory.retrieval.budget_ms, defaulting to 300 ms; if it does not return in time the turn proceeds without the note rather than becoming slower, because a slow embedder in the reply path had been showing up as multi-second first-token lag. And any hit whose text already appears in the durable block injected at conversation start is dropped, so the same fact is never spent twice in one prompt.'
    ));
    items.push(bodyPara(
        'In-session compaction keeps a long conversation bounded. History is allowed to grow to llm.history_trim_at turns and only then trimmed back to llm.history_turns, which ships as 12 and 6. This asymmetry is a cache optimisation: appending is cheap because the model reprocesses only the new messages, whereas dropping the oldest message shifts the prefix and forces a full re-read, so the expensive operation is made rare rather than constant. Trimmed turns are not discarded; they are held pending and folded into a rolling summary by a background thread, and the trim window is aligned to a user turn because a history opening with a dangling assistant message reads as replying to nothing. The summary installer carries a stale-apply guard: if the conversation was reset while the summary was being computed, the covered turns no longer match and the summary is dropped, so a dead session can never ghost into a fresh one.'
    ));

    // Figure 11 (activation decay)
    items.push(...insertFigure(11,
        'FIGURE 11 · MEMORY ACTIVATION DECAY',
        'Recency contribution against age, at the shipped 14 day half-life. The floor at 0.2 is what allows an old but important and relevant memory to still surface; without it, recall would be purely a function of how recently something was said.'
    ));

    // 4.8
    items.push(subHeading('4.8', 'Generating and Speaking'));
    items.push(bodyPara(
        'Generation starts on a background thread before the robot has decided whether to play a filler, so that model prefill overlaps the filler rather than following it. Handing back a bare generator would not do: a generator does not begin work until first consumed, so the worker thread is what makes prefill actually start.'
    ));
    items.push(bodyPara(
        'The filler races the reply. A filler is chosen with probability conversation.filler_probability and matched to what was said: an utterance ending in a question mark or opening with a question word gets "good question, let me think"; a reply of two words or fewer gets a short acknowledgement; everything else gets a neutral line. It is then played only if no reply audio has arrived within conversation.filler_grace_ms. A fast answer is therefore never delayed by a canned "let me think". Fillers are pre-synthesised at startup, and that pre-synthesis aborts after two consecutive failures rather than hammering a dead TTS at thirty seconds per call.'
    ));
    items.push(bodyPara(
        'The reply is spoken sentence by sentence. A producer thread splits the streaming text into sentences and pushes each sentence\'s audio pieces onto a bounded queue, while a single gapless output stream plays them as they arrive, so there are no inter-sentence pauses. Anything the model writes in parentheses is stripped before synthesis and never enters history, because those are hallucinated stage directions rather than speech.'
    ));
    items.push(bodyPara(
        'Two failure modes in this path were fixed and both are worth recording. The producer\'s completion sentinel must be delivered with a blocking put rather than a non-blocking one: with put_nowait the sentinel was dropped whenever the queue was full, which happens as soon as synthesis outpaces playback, and a dropped sentinel left the consumer blocked forever, freezing the conversation in SPEAKING so the robot went deaf after its first reply. Symmetrically, the producer\'s own queue writes must time out and re-check the stop flag, because on a barge-in the consumer stops draining and a plain blocking put would wedge the producer thread.'
    ));
    items.push(bodyPara(
        'Barge-in runs for the whole time the robot is making sound, covering the filler as well as the reply, and triggers on either the wake word or sustained user speech over the reply. Speech-based interruption needs no wake word and is echo-aware, learning the ambient level for conversation.barge_in_learn_ms and requiring conversation.barge_in_speech_ms of speech at conversation.barge_in_ratio above it. The interrupting words are captured and fed into the next turn\'s transcription so the person never has to repeat themselves, but only the trigger window plus a short lead-in is kept, because a longer ring buffer\'s prefix is the robot\'s own reply echo and it garbled the following turn.'
    ));
    items.push(bodyPara(
        'After an interruption only what was actually spoken enters history, sliced by the index of the last sentence whose audio reached the speaker. The model must not "remember" saying sentences the person never heard. The language model stream is also stopped and its HTTP connection closed, so the GPU stops generating a reply nobody is listening to.'
    ));
    items.push(bodyPara(
        'One ordering rule in the barge-in shutdown is subtle enough to note. The monitor thread is joined while the microphone is still live, and only then is the microphone paused. Pausing first would block the monitor forever inside the frame iterator, and it would then wake up and steal the next turn\'s audio off the shared queue, which is precisely how the robot used to go deaf after its first reply.'
    ));

    // 4.9
    items.push(subHeading('4.9', 'Command Paths That Bypass the Model'));
    items.push(bodyPara(
        'Several utterances are handled before the model is ever consulted, each with its own parser, and each replying with a fixed line.'
    ));
    items.push(bodyPara(
        'Stop phrases end the conversation. Matching is on word boundaries and only in utterances of five words or fewer, so a sentence that merely mentions one, such as asking about a film called Goodbye Lenin, does not put the robot to sleep.'
    ));
    items.push(bodyPara(
        'Enrolment has two entry points, an explicit command such as "remember my face" and a plain introduction such as "I\'m David", and both run the same guided multi-angle capture. Object teaching is distinguished from person introduction by the article: "this is a french press" teaches an object, "this is Peter" does not. Behavioural corrections such as "speak slower" or "keep it short" are stored as standing preferences and, where an engine knob exists, applied immediately. Erasure distinguishes forgetting the last item from forgetting a person entirely, and the latter clears the face and voice registrations as well as the stored facts.'
    ));

    // 4.10
    items.push(subHeading('4.10', 'Deciding When to Speak Unprompted'));
    items.push(bodyPara(
        'The hard part of proactivity is staying quiet, and every proactive utterance must pass every gate. A person must be present, and an unrecognised person only counts when proactive.engage_unknown is set. Quiet hours must not be active. The per-kind cooldown must have elapsed. The person must not already have been greeted during this arrival, where an arrival ends after proactive.presence_reset_s. And a global cap of proactive.max_per_hour utterances must not be exceeded. Timers and reminders bypass the presence gates, because the user explicitly asked for those, but they still respect quiet hours by being deferred rather than dropped.'
    ));
    items.push(bodyPara(
        'On top of the fixed cooldowns sits a bandit-style adaptation. Each proactive kind keeps an exponential moving average of how its utterances land, scored in the range minus one to one, and that average scales the kind\'s base cooldown. An average of zero or unknown leaves the cooldown unchanged. A positive average shortens it, down to half at plus one. A negative average lengthens it, up to three times at minus one. A kind of nudge that keeps falling flat therefore backs off by itself.'
    ));

    // 4.11
    items.push(subHeading('4.11', 'Scoring Outcomes and Surprise'));
    items.push(bodyPara(
        'Every exchange becomes a reward-tagged episode, with the reward assembled from three signals that already exist and cost nothing extra. Affect contributes the speaker\'s valence and confidence while speaking, on the reasoning that tone is itself feedback. Interaction contributes a negative for a barge-in, since being cut off is a verdict, and a mild positive for the conversation continuing within 90 seconds. The strongest signal is an explicit verdict in the opening of the next utterance, matched by two deliberately narrow regular expressions: generic negativity such as "this weather is awful" is affect\'s job, not a judgement on the robot.'
    ));
    items.push(bodyPara(
        'Because a verdict usually arrives one utterance after the reply it judges, tagging is retrospective: the next utterance writes its reward back onto the episode it actually judges. A pending proactive utterance is resolved the same way, with any reply at all counting as a partial success and an explicit verdict overriding.'
    ));
    items.push(bodyPara(
        'Surprise is scored separately and drives attention rather than reward. The world state keeps online, Laplace-smoothed counts of each label and event kind, and scores each new event by its rarity in bits, which is the negative base-two logarithm of its smoothed historical share. First-ever events score highest and daily routine decays toward zero. Two thresholds consume that score: events above world.surprise.remember_bits become scene episodes for later consolidation, and events above world.surprise.narrate_bits wake the Tier 2 narrator. The statistics persist across runs as a JSON sidecar, so the robot\'s sense of what is normal accumulates over its lifetime rather than resetting at every boot.'
    ));

    return items;
}

// ═══════════════════════════════════════════════════════════════════════════
// BUILD
// ═══════════════════════════════════════════════════════════════════════════
async function main() {
    const children = [
        ...section1(),
        ...section2(),
        ...section3(),
        ...section4(),
    ];

    const doc = new Document({
        sections: [{
            properties: {
                page: {
                    size: {
                        width: convertInchesToTwip(8.27),  // A4
                        height: convertInchesToTwip(11.69),
                    },
                    margin: {
                        top: convertInchesToTwip(0.8),
                        bottom: convertInchesToTwip(0.8),
                        left: convertInchesToTwip(0.75),
                        right: convertInchesToTwip(0.75),
                    },
                },
            },
            children,
        }],
    });

    const buf = await Packer.toBuffer(doc);
    const outPath = path.join(BASE, 'ZBR-AF1-SW-001_Sections1-4.docx');
    fs.writeFileSync(outPath, buf);
    console.log(`Written: ${outPath} (${buf.length} bytes)`);
}

main().catch(err => { console.error(err); process.exit(1); });
