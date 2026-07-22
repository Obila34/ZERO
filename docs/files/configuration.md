# ZERO — Configuration & environment reference

ZERO is **YAML-configured, not environment-configured**. There is no `.env`, no
`.env.example`, and no `os.getenv` call anywhere in the `zero/` package except
for the log level and a display probe. The eleven environment variables that do
exist belong to shell scripts and GPU servers.

Configuration comes from two files, deep-merged:

1. `config.yaml` at the project root — checked in, the shipped defaults.
2. `config.local.yaml` at the project root — **gitignored**, deep-merged on top
   for per-device overrides (mic index, gain, paths).

`load_config(path)` accepts an explicit path as the first positional CLI
argument (`python -m zero.main /path/to/other.yaml`); `config.local.yaml` is
merged on top regardless. Access is dotted: `cfg.get("tts.engine", "piper")`.
`cfg.resolve_path("stt.model_path")` resolves relative values against the
project root, so the process's working directory never matters.

The GPU vision service has a **separate** config at `server/vision/config.yaml`,
loaded relative to its own file and `lru_cache`d.

---

## 1. Environment variables (the complete list)

| Variable | Read by | Type | Default | Effect |
|---|---|---|---|---|
| `ZERO_LOG` | `zero/utils/logging.py` | string | `INFO` | Root log level for the whole Pi app |
| `DISPLAY` / `WAYLAND_DISPLAY` | `zero/vision/preview.py::_has_display` | presence | unset | `vision.preview_mode: auto` picks a cv2 window if either is set, else the MJPEG web sink |
| `GPU_HOST` | `scripts/pi_tunnel.sh`, `zero-tunnel.service` | `user@host` | none — script exits 2 | SSH target for the tunnel. The unit ships a Tailscale IP so it works off-LAN |
| `AUTOSSH_GATETIME` | `zero-tunnel.service` | int | `0` | autossh: don't require a successful first session before restarting |
| `ZERO_LOG_DIR` | `scripts/run_gpu_servers.sh` | path | `$HOME/zero_logs` | Where detached GPU server logs go |
| `ZERO_VENV` | `scripts/run_gpu_servers.sh` | path | `<repo>/.venv` | venv to activate before launching GPU servers |
| `ORPHEUS_FP16` | `scripts/run_gpu_servers.sh` | `0`/`1` | `0` | `1` starts the fp16 vLLM Orpheus; `0` starts the quantized GGUF build |
| `LD_LIBRARY_PATH` | `server/whisper_server.py`, `server/orpheus_cpp_server.py` | path list | inherited | **Set by the process itself**, then `os.execv` re-exec — see below |
| `_ZB_CUDA_RELAUNCHED` | same two files | `1` | unset | Internal guard so the CUDA-library re-exec happens exactly once |

### The CUDA re-exec

Both GPU servers call `_ensure_cuda_libs()` at import time. It globs
`site-packages/nvidia/**` for `libcublas`/`libcudnn` (whisper) or every `*.so*`
(orpheus), prepends those directories to `LD_LIBRARY_PATH`, sets
`_ZB_CUDA_RELAUNCHED=1`, and re-executes the interpreter with the same argv.
This is why you never set `LD_LIBRARY_PATH` by hand — and why heavy imports
(`uvicorn`, `faster_whisper`, `llama_cpp`) sit *after* the call with `# noqa: E402`.

---

## 2. `config.yaml` — master key reference

Consumed by `zero/factory.py` unless noted. "Consumer" names the module that
actually reads the key.

### `audio`

| Key | Type | Shipped value | Consumer | Notes |
|---|---|---|---|---|
| `sample_rate` | int | 16000 | everywhere | 16 kHz mono is assumed end-to-end |
| `block_ms` | int | 30 | capture, VAD, barge-in | frame size |
| `input_device` | str/int | `hw:3,0` | `MicCapture` | PortAudio device; auto-resamples if the device rejects 16 kHz |
| `input_gain` | float | 6.0 | `MicCapture` | Tune from the `utterance: rms=` log; target rms 2000–6000 |
| `output_device` | str/int | `pipewire` | `Speaker` | |
| `aec.enabled` | bool | `false` | `SpeexAEC` | Needs `speexdsp`; wired output only — BT latency drift defeats it |
| `aec.filter_ms` | int | 200 | `SpeexAEC` | Echo tail length |

### `privacy`

| Key | Type | Shipped | Consumer | Notes |
|---|---|---|---|---|
| `bystander_mode` | enum | `guarded` | `PrivacyGuard` | `open` = answer + remember · `guarded` = answer, don't store · `strict` = don't engage (and don't even transcribe) |
| `indicator_gpio_pin` | int? | blank | `ListeningIndicator` | Blank = log-only indicator |

### `control` — the AF1 fusion surface

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | Starts `ControlServer` in-process |
| `host` | str | `0.0.0.0` | **Unauthenticated. See `api.md` §1.** |
| `port` | int | 8090 | |
| `person_id` | int | 1 | Default memory owner for external turns |

### `conversation`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `sleep_timeout_ms` | int | 30000 | Silence before returning to wake-word mode |
| `multilingual` | bool | `true` | Appends a Swahili/English code-switch instruction to the system prompt |
| `barge_in` | bool | `true` | Master switch; `false` disables both mechanisms |
| `barge_in_on_speech` | bool | `true` | `false` = wake-word-only interruption |
| `barge_in_learn_ms` | int | 900 | Echo-floor learning window at reply start |
| `barge_in_speech_ms` | int | 300 | Sustained foreground speech needed to trigger |
| `barge_in_ratio` | float | 1.6 | Foreground must beat the learned echo floor by this factor |
| `barge_in_min_rms` | int | 250 | Absolute int16 RMS floor |
| `filler_probability` | float | 0.25 | Chance a filler is even considered |
| `filler_grace_ms` | int | 600 | Real audio arriving inside this window wins — a fast reply is never delayed |
| `fillers.{question,default,ack}` | str[] | short phrases | Pre-synthesised at startup; synthesis aborts after 2 consecutive misses |
| `compaction.enabled` | bool | `true` | Fold trimmed turns into a rolling summary |
| `compaction.max_summary_chars` | int | 600 | Hard cap on the injected summary |

### `memory`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | `false` → `build_memory` returns `None`, every call site no-ops |
| `path` | path | `zero_memory.sqlite` | |
| `max_facts` | int | 30 | Semantic facts injected at conversation start |
| `recent_episodes` | int | 3 | Episode summaries injected; 0 = off |
| `max_stored_facts` | int | 200 | Storage cap (oldest by `last_access` pruned) |
| `max_stored_episodes` | int | 100 | |
| `embeddings.backend` | enum | `auto` | `auto` (ollama if `llm.host` set, else hash) · `ollama` · `hash` · `off` |
| `embeddings.ollama_model` | str | `nomic-embed-text` | **Must be a real embedding model** — a chat model silently degrades recall |
| `embeddings.timeout_s` | float | 3.0 | Fail fast into hash rather than stall the reply path |
| `embeddings.slow_ms` | int | 800 | Consistently slower → auto-degrade to hash for the session |
| `embeddings.hash_dim` | int | 256 | Fallback embedder dimension |
| `retrieval.top_k` | int | 6 | Memories surfaced per recall |
| `retrieval.half_life_days` | float | 14 | Recency decay |
| `retrieval.min_activation` | float | 0.05 | Below this a memory never surfaces |
| `retrieval.budget_ms` | int | 300 | Hard per-turn recall budget; over budget = skip the note this turn |
| `forgetting.enabled` | bool | `true` | |
| `forgetting.max_age_days` | float | 90 | |
| `last_conversation.enabled` | bool | `true` | The protected per-person welcome-back record |

### `learning`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `objects.enabled` | bool | `true` | |
| `objects.db_path` | path | `zero_objects.sqlite` | |
| `objects.clip_model_path` | path? | blank | Blank = offline histogram embedder |
| `objects.match_threshold` | float | 0.80 | Local histogram embedder |
| `objects.match_threshold_clip` | float | 0.72 | Used instead when the remote CLIP embedder is active |
| `objects.max_samples_per_object` | int | 5 | |
| `objects.unknown_conf` | float | 0.45 | Below this and unlearned → curiosity queue |
| `curiosity.enabled` | bool | `true` | |
| `curiosity.db_path` | path | `zero_curiosity.sqlite` | |
| `curiosity.max_pending` | int | 20 | |
| `corpus.enabled` | bool | `true` | |
| `corpus.dir` | path | `data/corpus` | |

### `perception`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `remote.enabled` | bool | `true` | Master switch for `/perceive/*` offload; `false` = all-local engines |
| `remote.base_url` | url | `http://127.0.0.1:8000` | Through the tunnel |
| `remote.timeout_s` | float | 5.0 | face / speaker / CLIP |
| `remote.detect_timeout_s` | float | 10.0 | First call loads YOLO11x |
| `remote.jpeg_quality` | int | 80 | |
| `affect.enabled` | bool | `true` | Voice prosody + text valence (+ optional FER) |
| `affect.face_model_path` | path? | blank | Optional FER+ ONNX; blank = voice+text only |
| `diarize.enabled` | bool | `true` | |
| `diarize.change_threshold` | float | 0.40 | Consecutive-turn voice cosine below this = new speaker |

### `proactive`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | |
| `engage_unknown` | bool | `true` | Greet anyone seen, not only enrolled faces |
| `quiet_hours` | [int,int] | `[22, 7]` | `[]` disables |
| `greet_cooldown_s` | int | 14400 | Re-greet the same person at most every 4 h |
| `curiosity_cooldown_s` | int | 1800 | |
| `remark_cooldown_s` | int | 300 | |
| `max_per_hour` | int | 6 | Hard global ceiling |
| `presence_reset_s` | int | 1200 | Away this long → next arrival greets again |
| `check_interval_s` | float | 3.0 | Watcher tick |
| `linger_s` | float | 45 | Presence duration before a question/remark |
| `consolidate_interval_s` | int | *(not in `config.yaml`)* | Read by `build_proactive`, defaults to 1800 — an undocumented knob |

### `preferences`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | Enables the "speak slower" / "keep it short" parser |

### `tools`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | `false` returns the bare LLM — no router, no timers |
| `allow` | str[] | `[time, timer, reminder, remember, recall, web_search]` | **The safety boundary.** A tool not listed cannot be registered |
| `websearch.enabled` | bool | `true` | Also gates the "staying current" paragraph in the system prompt |
| `websearch.url` | url | `http://127.0.0.1:8080/search` | SearXNG JSON API |
| `websearch.timeout_s` | float | 8.0 | |
| `websearch.max_results` | int | 3 | |

### `identity`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | |
| `db_path` | path | `zero_identity.sqlite` | |
| `max_samples_per_kind` | int | 5 | Per person, per channel |
| `reinforce_threshold` | float | 0.70 | Confident joint matches refresh stored samples |
| `guests.enabled` | bool | `true` | |
| `guests.db_path` | path | `zero_guests.sqlite` | |
| `guests.match_threshold` | float | 0.55 | Same-guest voice cosine |
| `guests.max_guests` | int | 50 | |
| `guests.min_words` / `min_ms` / `min_rms` | 2 / 1200 / 150 | | Anti-hallucination quality gate before a guest is minted |
| `session.voice_only` | bool | `true` | `false` restores legacy face+voice session ownership |
| `session.write_min_score` | float | 0.55 | Stricter than the live accept — governs what gets *saved* |
| `voice.model_path` | path | `models/voiceid/voxceleb_ECAPA512_LM.onnx` | |
| `face.model_path` | path | `models/identity/arcface.onnx` | Blank/missing → voice-only |
| `face.min_size` | int | 60 | Smallest face in px worth matching |
| `fusion.w_face` / `w_voice` | 0.55 / 0.45 | | Normalised internally |
| `fusion.threshold` | float | 0.50 | Both-channel accept |
| `fusion.face_threshold` | float | 0.42 | Single-channel accepts are stricter by nature |
| `fusion.voice_threshold` | float | 0.45 | Also the live session-ownership bar |

### `voiceid` — "only my voice"

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `enabled` | bool | **`false`** | Off by default; enrol first with `python test_voiceid.py enroll` |
| `model_path` | path | `models/voiceid/voxceleb_ECAPA512_LM.onnx` | |
| `profile_path` | path | `voiceprint.npy` | Missing model *or* profile → auto-disabled with a warning |
| `threshold` | float | 0.45 | Owner ≈ 0.55–0.70, others ≈ 0.25–0.35 |

### `wake`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `engine` | enum | `openwakeword` | Only value implemented; anything else raises |
| `model` | str | `hey_jarvis` | A built-in, not a real "Hey Zero" — that needs a custom-trained model |
| `threshold` | float | 0.5 | |

### `vad`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `engine` | enum | `webrtc` | `webrtc` (energy-based, catches quiet mics) or `silero` (sharper but stricter) |
| `silero_model` | path | `models/vad/silero_vad.onnx` | |
| `silero_threshold` | float | 0.3 | Lower = more sensitive |
| `silence_ms` | int | 450 | Trailing silence that ends an utterance |
| `semantic_hold` | bool | `true` | Check the speculative transcript for a mid-thought ending before committing |
| `semantic_hold_wait_ms` | int | 600 | Bounded wait when the transcript is still in flight |
| `min_speech_for_fast_end_ms` | int | 600 | Short utterances wait **double** the silence; 0 = off |
| `aggressiveness` | int 0–3 | 2 | webrtc only |
| `energy_threshold` | int | 150 | Per-frame int16 RMS floor to *start* a capture |
| `min_utterance_rms` | int | 30 | Proximity gate — whole-utterance average; deliberately very low |
| `max_utterance_ms` | int | 15000 | Hard cap |
| `speech_pad_ms` | int | 200 | Pre/post-roll |

### `stt`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `engine` | enum | `remote` | `remote` (GPU turbo) or `whispercpp` (local) |
| `remote_url` | url | `http://127.0.0.1:9000/transcribe` | |
| `remote_timeout` | int | 30 | Also used as the STT thread join bound (+5 s) |
| `language` | str | `auto` | blank = server default · `sw` · `auto` (best for code-switching) |
| `fallback` | str? | `whispercpp` | Blank = no fallback; failures are logged and the utterance dropped |
| `model` | str | `base.en` | Local fallback only |
| `model_path` | path | `models/whisper/ggml-base.en.bin` | |
| `threads` | int | 4 | All Pi cores |
| `audio_ctx` | int | 768 | Caps the encoder window; 0 = full 30 s |
| `initial_prompt` | str | conversational bias | Trims single-phoneme mishears |

### `llm`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `engine` | enum | `ollama` | Only value implemented |
| `host` | url | `http://127.0.0.1:11435` | Tunnel port; also the embeddings host |
| `model` | str | `gemma4:latest` | |
| `temperature` | float | 0.6 | Honesty is enforced by prompt clauses, not temperature |
| `max_tokens` | int | 500 | → `num_predict` |
| `num_ctx` | int | 4096 | Halves KV-cache VRAM vs 8192 on a shared GPU. **Must match between `warmup()` and `stream()`** |
| `history_turns` | int | 6 | Size kept after a trim |
| `history_trim_at` | int | 12 | Let history grow to this before trimming — most turns stay append-only (prefix-cache hit) |

Not exposed in `config.yaml`: `OllamaLLM.keep_alive` is hardcoded to `-1`
(pin in VRAM forever).

### `tts`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `engine` | enum | `orpheus` | `orpheus` · `piper` · `fish` |
| `fallback` | str? | `piper` | Blank = go mute on remote failure |
| `orpheus.url` | url | `http://127.0.0.1:9100/tts` | `/tts_stream` is derived by string replacement |
| `orpheus.voice` | str | `leo` | male: leo dan zac · female: tara leah jess mia zoe |
| `orpheus.timeout` | int | 30 | |
| `orpheus.prebuffer_ms` | int | 300 | Jitter buffer — read by `Speaker`, not by the TTS engine |
| `piper.binary` | path | `piper/piper` | Relative paths with a `/` resolve against the repo root |
| `piper.voice` | path | `models/piper/en_US-amy-medium.onnx` | |
| `piper.length_scale` | float | 1.0 | Base rate; mood and preferences nudge it |
| `fish.model_dir` | path | `models/fish/openaudio-s1-mini` | |
| `fish.precision` | enum | `int8` | |
| `fish.infer_cmd` | str? | **`null`** | The Fish path is non-functional until this is set |

### `vision`

| Key | Type | Shipped | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | Needs `requirements-vision.txt`; import failure → voice-only, logged |
| `multimodal` | bool | `true` | **Only set true if `llm.model` is vision-capable** — verify with `ollama show` |
| `max_items` | int | 6 | Objects named in the detector hint |
| `change_note_cooldown_s` | float | 120 | Rate limit on spontaneous "oh, is that new?" remarks |
| `preview` | bool | `true` | |
| `preview_mode` | enum | `auto` | `auto` · `window` · `web` |
| `preview_host` | str | `127.0.0.1` | `0.0.0.0` exposes an **unauthenticated** camera stream on the LAN |
| `preview_port` | int | 8008 | |
| `preview_scale` | float | 1.0 | |
| `detect_interval_s` | float | 0.1 | Min seconds between detections (~10/s) |
| `frames_per_look` | int | 2 | Keyframes per visual turn; a person-crop replaces one wide frame |
| `look_window_s` | float | 1.0 | Sample those frames from the last N seconds |
| `camera.index` | int | 0 | BRIO exposes video0–3; only 0 (or sometimes 2) is a capture node |
| `camera.width/height` | int | 640 / 480 | |
| `camera.request_fps` | int | 30 | |
| `camera.mjpg` | bool | `true` | USB webcams can't sustain raw YUYV at this res/fps |
| `detect.model_path` | path | `yolo11s.onnx` | Checked into the repo, NMS baked into the graph |
| `detect.confidence` | float | 0.35 | |
| `detect.iou` | float | 0.45 | **Informational only** — NMS is inside the ONNX graph |
| `detect.device` | enum | `cpu` | `cuda` if an onnxruntime CUDA EP is present |
| `detect.imgsz` | int | 640 | Must match the export |
| `detect.classes` | str[] | `[]` | `[]` = keep all 80 COCO classes |
| `detect.names_path` | path? | blank | Blank = auto-load a sibling `<model>.names.json`, else COCO-80 |
| `color.top_n` | int | **0** | 0 skips pixel colour naming entirely — the multimodal LLM reads real colours off the frame |
| `color.center_crop` / `min_saturation` / `min_value` / `white_value` / `min_colorful_ratio` / `min_pixels` | | 0.6 / 50 / 40 / 200 / 0.10 / 50 | HSV naming tunables |
| `gpu.enabled` | bool | **`false`** | Depth/bearing facts are off by default (they read as clinical) |
| `gpu.url` | url | `http://127.0.0.1:8000` | |
| `gpu.health_path` / `facts_path` / `analyze_path` | | `/health` `/facts` `/analyze` | |
| `gpu.health_timeout_s` / `facts_timeout_s` / `analyze_timeout_s` | | 5.0 / 15.0 / 30.0 | |
| `gpu.jpeg_quality` | int | 80 | |
| `gpu.vlm_fallback` | bool | **`false`** | Route visual turns to the GPU VLM when the main LLM can't see. Ignored when `multimodal: true` |

### Removed / dead keys

The `persona` section no longer exists — the system prompt lives in
`zero/llm/persona.py::SYSTEM_TEMPLATE`. `config.yaml` documents the removal at
the bottom of the file. Any `persona:` block in a `config.local.yaml` is silently
ignored.

---

## 3. `server/vision/config.yaml` — GPU service

| Key | Shipped | Notes |
|---|---|---|
| `server.host` / `port` / `log_level` | `0.0.0.0` / 8000 / `info` | Bound by `app.main()`; the systemd unit passes host/port on the uvicorn command line instead |
| `depth.model` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | Metric checkpoint → `predicted_depth` is in meters |
| `depth.metric` | `true` | `false` = affine-invariant inverse depth, scaled by `relative_scale` |
| `depth.relative_scale` | 1.0 | Only used when `metric: false` |
| `depth.max_depth_m` | 20.0 | Clamps sky/edge noise |
| `facts.bbox_center_crop` | 0.6 | Central crop for the per-box depth median |
| `facts.center_bearing_deg` | 12 | \|angle\| below this → `center` |
| `facts.vertical_bearing` / `vertical_deg` | `false` / 18 | Optional high/low |
| `facts.min_valid_pixels` | 10 | Fewer valid samples → no distance |
| `camera.intrinsics_path` | `../calibration/intrinsics.json` | **This file does not exist in the repo** — the FOV fallback always applies |
| `camera.fallback_hfov_deg` | 70 | Used when intrinsics are absent; logs "APPROXIMATE intrinsics" |
| `vlm.model` | `Qwen/Qwen2-VL-2B-Instruct` | Or `vikhyatk/moondream2` |
| `vlm.backend` | `qwen` | `qwen` \| `moondream` |
| `vlm.quantization` | `none` | `4bit`/`8bit` need bitsandbytes + CUDA |
| `vlm.max_new_tokens` | 96 | |
| `vlm.temperature` | 0.2 | Low → stays close to the grounded facts |
| `vlm.max_history_turns` | 6 | |
| `vlm.send_image` | `true` | Facts are always included; this adds the frame |
| `runtime.device` | `auto` | |
| `runtime.cache_dir` | `.cache` | HuggingFace weights |
| `debug.enabled` / `dir` | `false` / `debug` | Saves a colorised depth map per `/facts` request |

Hardcoded in `server/vision/perception.py` rather than configured:
`YOLO_MODEL = "yolo11x.pt"`, `YOLO_CONF = 0.35`,
`CLIP_MODEL = "openai/clip-vit-base-patch32"`, and the two ONNX paths
`models/voiceid/voxceleb_ECAPA512_LM.onnx`, `models/identity/arcface.onnx`
(resolved against the repo root). Changing these requires a code edit.

---

## 4. Port map (both nodes)

| Port | Where | Service | Reached how |
|---|---|---|---|
| 8008 | Pi | Vision preview MJPEG | Browser, private by default |
| 8090 | Pi | Control plane | LAN, unauthenticated |
| 11435 → 11434 | Pi → GPU | Ollama | tunnel |
| 9000 | Pi → GPU | Whisper | tunnel |
| 9100 | Pi → GPU | Orpheus | tunnel |
| 8000 | Pi → GPU | Vision + `/perceive/*` | tunnel |
| 8080 | Pi → GPU | SearXNG | tunnel |

The Pi-side ports in `config.yaml` (`llm.host`, `stt.remote_url`,
`tts.orpheus.url`, `vision.gpu.url`, `perception.remote.base_url`,
`tools.websearch.url`) **must** match the `-L` forwards in
`scripts/pi_tunnel.sh`. They are duplicated in two places with no shared
constant — change one, change both.
