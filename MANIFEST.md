# Migration MANIFEST — zerolabs1

Node: `zerolabs1` (Ubuntu 24.04, kernel 6.8.0-136, driver 595.58.03, CUDA 13.2)
Started: 2026-08-03

## Baseline (Phase 0, before any change)

- GPU: 1x RTX 5060 Ti, 16311 MiB total, **12673 MiB used, 3169 MiB free**, compute cap 12.0 (sm_120)
- Disk `/`: 466G total, 349G used, **98G available** (79% full)

### Live services (all KEEP until cutover confirmed)

| Service | Port | VRAM | Unit |
|---|---|---|---|
| Ollama (`gemma4:latest` 8B Q4_K_M + `nomic-embed-text`) | 11434 | 4352 + 500 MiB | `ollama.service` |
| ZERO Vision (depth + scene facts) | 8000 | 1786 MiB | `zero-vision.service` |
| Whisper STT (`deepdml/faster-whisper-large-v3-turbo-ct2`) | 9000 | 2252 MiB | none — nohup, started by login script |
| Orpheus TTS (GGUF + SNAC) | 9100 | 2818 MiB | `zero-orpheus.service` |
| zb-face (InsightFace buffalo_l) | — | 888 MiB | `zb-face.service` |
| SearXNG (docker) | 8080 | — | docker container `searxng` |

## REMOVED

Safe-list sweep, 2026-08-03 (approved). Free space `/`: 98G → **175G** (+77G).

| Item | Size | Why |
|---|---|---|
| `~/.cache/pip` | 8.6G | safe list — package cache, re-downloadable |
| `/root/.cache/pip` | 5.0G | safe list — package cache, re-downloadable |
| `~/.cache/uv` | 4.0G | safe list — package cache, re-downloadable |
| apt cache (`apt-get clean`) | ~0.5G | safe list |
| `/root/.ollama` | 9.0G | orphan — ollama runs as user `ollama` from `/usr/share/ollama/.ollama`; stray duplicate `gemma4:latest` blob, referenced by nothing |
| ollama `gemma4:12b-it-q4_K_M` | 7.6G | safe list — not `gemma4:latest`, not loaded |
| ollama `gemma4:12b-it-qat` | 7.2G | safe list — not `gemma4:latest`, not loaded |
| 23 unused docker images | 39.2G | safe list — only `searxng` is running; it was untouched and is still up |

Verified after sweep: `ollama ps` still shows `gemma4:latest` + `nomic-embed-text` resident on GPU; `searxng` container up; no live service restarted.

## Fleet VRAM survey (2026-08-03)

| Node | GPU | Total | Free | Held by |
|---|---|---|---|---|
| zerolabs1 (this) | RTX 5060 Ti (sm_120) | 16311 MiB | **3169 MiB** | ollama, orpheus:9100, whisper:9000, vision:8000, zb-face |
| zerolabs0 (`100.95.210.94`) | RTX 5070 Ti (sm_120) | 16303 MiB | **5078 MiB** | S2-Pro fish-speech `api_server.py` on :8080 (compiled) |
| zerolabs2 (`100.100.95.12`) | RTX 5060 Ti | 16311 MiB | **7034 MiB** | `zero-whisper` + `zero-orpheus` ("zl2 direct-link"), zb-voice (Chatterbox), Blender wall |
| zerolabs3-1 (`100.76.252.64`) | Quadro P620 | 2048 MiB | 1045 MiB | negligible — unusable for this |

SSH: `maxwell@<tailscale-ip>`. zerolabs5 / zeroasus not accessible under this account.

## STOPPED (reversible — not deleted)

| What | Node | Freed | How to restore |
|---|---|---|---|
| S2-Pro `fish-speech/tools/api_server.py` (pid 429143) | zerolabs0 | **10762 MiB VRAM** | `ssh maxwell@100.95.210.94 'cd ~/s2lab && setsid nohup ./run_s2.sh >> api_server.log 2>&1 < /dev/null &'` — note `--compile` means a slow first-request warmup |
| `zero-whisper` + `zero-orpheus` (redundant duplicates) | zerolabs2 | **4074 MiB VRAM** | `sudo systemctl enable --now zero-whisper zero-orpheus` — now `disabled`, so they no longer return at reboot |
| `zb-orpheus-watchdog.timer` + `.service` | zerolabs2 | **2872 MiB VRAM** | `sudo systemctl enable --now zb-orpheus-watchdog.timer` |
| `zb-wall-live` (Blender AF1 wall) | zerolabs2 | **1436 MiB VRAM** | `sudo systemctl enable --now zb-wall-live` — user asked to disable, plans to restore later |

### Evidence the stopped items were unused

- **zerolabs2 whisper/orpheus**: the Pi's autossh tunnel terminates at `obilasam3@100.110.56.17` (**zerolabs1**), so the Pi can never reach zerolabs2's copies. No established connections on :9000/:9100; zero request activity in 6h.
- **`zb-orpheus-watchdog`**: fired every 2 min and restarted `zero-orpheus` on zerolabs2, silently undoing the disable and reclaiming 2.9GB. zerolabs1 runs its own separate `orpheus-watchdog.timer` for the authoritative instance. This would have evicted Kyutai TTS on a 2-minute cycle.
- **`zb-wall-live`**: NOT idle — `HDMI-0 connected primary 1920x1200`, a real attached monitor. Stopped only on explicit user instruction.

### Checked and deliberately left running

- **`zb-voice` Chatterbox** (3220 MiB) — loopback-only `127.0.0.1:8731`, no connections, zero requests in 24h, no journal entries. Strong idleness evidence but never confirmed by the user, so untouched.
- **`zb-compress`, `zb-scheduler`** — CPU-only, hold no VRAM.

Had no systemd unit; was launched by hand under `nohup`. Last served a TTS request 2026-08-03 10:07.
`s2_shim.py` (pid 442132, listens `0.0.0.0:9100`, no VRAM) was **left running** — it will error until S2-Pro is restarted.

zerolabs0 GPU after: **12 MiB used / 15828 MiB free**, no compute processes.

## INCIDENT — ZERO STT + Vision down after nightly reboot (2026-08-03 21:01 → 2026-08-04 07:2x UTC)

**Not caused by this migration.** zerolabs1 rebooted on schedule via the pre-existing `zb-nightly-reboot.timer`.
Post-reboot the following did NOT come back, leaving the Pi with no speech recognition and no vision for ~10h:

| Service | Port | Root cause |
|---|---|---|
| `zero-vision.service` | 8000 | unit is `disabled`; journal showed *no entries this boot* |
| `whisper_server.py` | 9000 | has **no systemd unit** — launched by a *login* script, so it only returns when someone SSHes in |

Survived the reboot: Orpheus (revived every 3 min by `orpheus-watchdog.timer`), ollama, zb-face.
`gemma4:latest` was also evicted — `zb-warm-gemma.service` ran fine at boot but requested `keep_alive: 30m`.

Restored 2026-08-04 (user-approved): `systemctl start zero-vision` + relaunched whisper as the login script does.
Verified after: 9000 whisper ready (2208 MiB), 8000 vision `/docs` 200, 9100 orpheus, 11434 ollama.

**Still open:** both will drop again at the next nightly reboot (21:00 UTC daily). Fixing that means
`systemctl enable zero-vision` and writing a real unit for whisper — deliberately NOT done, as it changes
live-node config beyond the migration's scope.

## Tunnel topology (verified 2026-08-03)

Pi (`zero-head`, 100.106.44.56) → `autossh ... obilasam3@100.110.56.17` = **zerolabs1**, forwarding:
`11435→11434` (ollama), `9000→9000` (whisper), `9100→9100` (orpheus), `8000→8000` (vision), `8080→8080`.
Unit: `zero-tunnel.service` → `/home/head/Mzee/ZERO/scripts/pi_tunnel.sh`.

**Consequence:** zerolabs1's services are authoritative. zerolabs2's `zero-whisper`/`zero-orpheus` are unreachable by the Pi — redundant duplicates, candidates for later cleanup (not touched).

## INSTALLED

### zerolabs0 (`100.95.210.94`, RTX 5070 Ti 16GB, sm_120) — vLLM host

| Item | Version / detail | Path |
|---|---|---|
| venv (Python 3.12.13) | system python is 3.14, unsupported by vLLM — pinned via `uv venv --python 3.12` | `~/vllm-stack/.venv` |
| vLLM | latest stable (installing) | `~/vllm-stack/.venv` |
| Model | `google/gemma-4-12B-it-qat-w4a16-ct` — 9.6GB on disk | `~/.cache/huggingface` |
| rustup / cargo | stable | `~/.cargo` |
| delayed-streams-modeling | git `--depth 1` | `~/vllm-stack/dsm` |

Planned port: **8001** (OpenAI-compatible).

### zerolabs2 (`100.100.95.12`, RTX 5060 Ti, ~7.0GB free) — Kyutai host

| Item | Version / detail | Path |
|---|---|---|
| rustup / cargo | stable | `~/.cargo` |
| CUDA toolkit | `cuda-toolkit-13-0` from NVIDIA apt repo (installing) — needed for `moshi-server --features cuda`; apt candidate 12.4 was too old for sm_120 | `/usr/local/cuda-13.0` |
| delayed-streams-modeling | git `--depth 1` | `~/kyutai/dsm` |

Dry-run verified before install: `0 upgraded, 78 newly installed, 0 to remove`, **no driver/dkms packages touched**. Live GPU services on the node confirmed unaffected.

| moshi-server | v0.6.4, built `--features cuda` (CUDA_COMPUTE_CAP=120) | `~/.cargo/bin/moshi-server` |
| Kyutai STT model | `kyutai/stt-1b-en_fr-candle` | `~/.cache/huggingface` |
| Kyutai TTS model | `kyutai/tts-1.6b-en_fr` + `kyutai/tts-voices` | `~/.cache/huggingface` |
| moshi python env | for the TTS `type = "Py"` module, imported by moshi-server via pyo3 | `~/kyutai/ttsenv` |

Ports: STT **8090** (live), TTS **8091**.

Local configs (stock configs OOM on this card):
- `configs/config-stt-zero.toml` — copy of `config-stt-en_fr-hf.toml` with `batch_size 64 → 8`
- `configs/config-tts-zero.toml` — copy of `config-tts.toml` with `batch_size 8 → 4`

## Build/runtime fixes applied (sm_120 gotchas)

| Node | Problem | Fix |
|---|---|---|
| zerolabs0 | vLLM system python is 3.14 (unsupported) | `uv venv --python 3.12` |
| zerolabs0 | flashinfer JIT: `Could not find nvcc, cuda_home='/usr/local/cuda' doesn't exist` | No system CUDA toolkit; the vLLM venv ships a full CUDA **13.3** tree — set `CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13` |
| zerolabs0 | flashinfer JIT: `FileNotFoundError: 'ninja'` | ninja is installed *in the venv*; must put `$VENV/bin` on `PATH` (invoking `.venv/bin/vllm` directly does not add it) |
| zerolabs2 | `cargo install moshi-server` → `openssl-sys`: system openssl not found | `apt-get install libssl-dev pkg-config` (openssl 3.0.13) |
| zerolabs2 | no nvcc; apt candidate was CUDA 12.4, too old for sm_120 | NVIDIA apt repo → `cuda-toolkit-13-0` (nvcc 13.0.88). Dry-run first confirmed no driver/dkms packages touched |
| zerolabs0 | flashinfer JIT: `CUDA compiler and CUDA toolkit headers are incompatible` (venv nvcc 13.3 vs flashinfer's bundled CCCL) | `VLLM_USE_FLASHINFER_SAMPLER=0` — falls back to the PyTorch-native sampler, no JIT needed |
| zerolabs0 | `torch.OutOfMemoryError` during CUDA-graph capture at `--gpu-memory-utilization 0.90` | dropped to `0.85`, `--max-model-len 8192`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. vLLM under-reserves with the multimodal towers loaded |
| zerolabs2 | HF `429 Too Many Requests` on `kyutai/tts-voices` (unauthenticated) | re-ran download; completed (EXIT=0) |
| zerolabs2 | `cargo install moshi-server` → `audiopus_sys`: build command not found | `apt-get install cmake clang libopus-dev build-essential` (cmake 3.28.3, clang 18.1.3, opus 1.4) |

Verified: `vllm 0.26.0`, `torch 2.11.0+cu130`, `torch.cuda.get_device_capability(0) == (12, 0)`, `cuda avail True`.

## Model selection reasoning

| Candidate | 4-bit size | Verdict |
|---|---|---|
| Gemma 4 31B Dense | ~17–18GB | Exceeds 15.8GB free. Rejected. |
| Gemma 4 26B-A4B MoE | NVFP4 ~16.5GB / FP8 ~26GB | Exceeds budget; also sm_120 has no native NVFP4 MoE kernels (vLLM issue #31085) → Marlin fallback. Rejected. |
| **Gemma 4 12B Unified QAT w4a16** | **9.6GB actual** | **Chosen.** Official Google compressed-tensors checkpoint built for vLLM; QAT beats post-hoc AWQ at 4-bit; 256K ctx; 140+ languages; multimodal image+audio, matching the live `gemma4:latest` capability set (8B → 12B upgrade). |

## PHASE 2 — VALIDATION RESULTS (2026-08-04)

### vLLM — `google/gemma-4-12B-it-qat-w4a16-ct` on zerolabs0:8001

| Metric | Result | Target | Verdict |
|---|---|---|---|
| TTFT, 1584-tok prompt, cold | **470.1 ms** (mean of 4: 468.2/470.0/471.0/471.3) | < 200 ms | **MISS** |
| TTFT, same prompt, prefix-cached | **39.6 ms** | < 200 ms | **PASS** |
| TTFT, short prompt | 29.8 ms | — | — |
| Sustained throughput | **80.3 tok/s** (mean of 3) | > 60 tok/s | **PASS** |
| Swahili/English code-switch | correct and idiomatic | sanity check | **PASS** |

Cold-prefill of ~1600 fresh tokens cannot hit 200ms on a 5070 Ti with a 12B model — that would need ~8k tok/s prefill.
ZERO sends a fixed persona prompt each turn, so the operating path is the **cached** 39.6ms figure.

Code-switch output: `Habari ya jioni, Sam! Nimefurahi kukusalimia. Karibu!  (Good evening, Sam! I am happy to greet you. Welcome!)`

### Kyutai STT — zerolabs2:8090

Streamed `audio/bria.mp3` via `scripts/stt_from_file_rust_server.py`. **Word-by-word partial transcripts with per-word
timestamps confirmed** (e.g. `45.28 - 45.52  to` / `45.52 - 45.52  swirl.`). Verified twice: manual run and under systemd.

### Kyutai TTS — zerolabs2:8091

| Metric | Result | Target | Verdict |
|---|---|---|---|
| Time-to-first-audio | **213.5 ms** mean (212.7–214.1) | < 500 ms | **PASS** |
| Generation rate | ~2.7x realtime | — | — |

Measured with `~/kyutai/tts_ttfa.py` over 3 utterances after a warmup. Pre-systemd run gave 199.5 ms mean.

## PHASE 3 — CUTOVER PREP (not executed; old stack left running)

### New endpoints

| Service | Host | Endpoint | Protocol |
|---|---|---|---|
| vLLM (Gemma 4 12B) | zerolabs0 `100.95.210.94` | `http://100.95.210.94:8001/v1/chat/completions` | OpenAI HTTP |
| Kyutai STT | zerolabs2 `100.100.95.12` | `ws://100.100.95.12:8090/api/asr-streaming` | WebSocket |
| Kyutai TTS | zerolabs2 `100.100.95.12` | `ws://100.100.95.12:8091/api/tts_streaming` | WebSocket |

Kyutai auth: `authorized_ids = ["public_token"]` — pass header `kyutai-api-key: public_token` or `?auth_id=public_token`.

### Tunnel changes

The Pi (`zero-head`) is already on the tailnet, and all three services bind `0.0.0.0`, so **no SSH tunnel is
strictly required** — the Pi can reach the tailscale IPs directly. That is the simplest option.

If you prefer to keep the loopback-tunnel pattern, the new services live on **two different hosts**, so the single
autossh line in `/home/head/Mzee/ZERO/scripts/pi_tunnel.sh` cannot cover them; you need two more autossh instances:

    # existing (unchanged — old stack on zerolabs1)
    autossh -M 0 -N -L 11435:localhost:11434 -L 9000:localhost:9000 \
            -L 9100:localhost:9100 -L 8000:localhost:8000 -L 8080:localhost:8080 \
            obilasam3@100.110.56.17

    # new: vLLM on zerolabs0
    autossh -M 0 -N -L 8001:localhost:8001 maxwell@100.95.210.94

    # new: Kyutai STT + TTS on zerolabs2
    autossh -M 0 -N -L 8090:localhost:8090 -L 8091:localhost:8091 maxwell@100.100.95.12

### BLOCKER — a config repoint alone will NOT work

All three new services speak different protocols from what ZERO's clients currently send. Verified in the code:

| ZERO client | Sends today | New service expects | Compatible? |
|---|---|---|---|
| `zero/llm/ollama_engine.py:55,92,128` | HTTP POST `{host}/api/chat` (**Ollama API**) | OpenAI `/v1/chat/completions` | **No** |
| `zero/stt/remote_engine.py:47` | HTTP POST `/transcribe` (whole clip) | WebSocket streaming | **No** |
| `zero/tts/orchestrator.py` | HTTP streaming `/tts` | WebSocket streaming | **No** |

Changing `config.yaml` alone will not cut over. Each needs a client adapter:
1. **LLM** — an OpenAI-compatible engine alongside `ollama_engine.py` (smallest job; the payload shapes are close).
2. **STT** — a real rework: HTTP request/response → persistent WebSocket with incremental partials. This is also
   the change that unlocks Kyutai's latency advantage, since the current design waits for a complete clip.
3. **TTS** — WebSocket client emitting PCM chunks into the existing `synthesize_stream` path.

## KNOWN ISSUE — Kyutai has no Swahili

Kyutai STT ships only `kyutai/stt-1b-en_fr` (English+French) and `kyutai/stt-2.6b-en` (English). TTS is likewise English/French.
The live Whisper `large-v3-turbo` handles Swahili; **a full STT cutover to Kyutai removes ZERO's spoken-Swahili comprehension.**
Raised with the user 2026-08-03; user elected to install both and plan full cutover anyway.
