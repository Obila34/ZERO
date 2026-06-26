# Plan: moving heavy compute off the Pi to a remote GPU

## Background

ZERO currently runs everything on the Raspberry Pi 5 CPU. That works, but it caps
what we can do. Whisper base.en transcribes in about 1.5 seconds, which is fine,
but anything more accurate is too slow: small.en takes around 5 seconds and
large-v3-turbo took 29 seconds for a single clip when we tested it on the Pi. The
language model has the same ceiling. Qwen 3B replies in 2 to 4 seconds when its
cache is warm, but slows down as the conversation grows.

We now have a GPU on a separate machine, reachable over SSH. The idea is to keep
the Pi as the device you talk to, and move the slow work (speech to text, and the
language model) onto the GPU. The Pi still records audio, detects speech, and
plays the spoken reply. The GPU does the parts that need real compute.

This means ZERO is no longer fully offline. It depends on the GPU being up and the
connection being open. We accept that trade for the accuracy and speed it buys.
Where possible we keep a local fallback so the Pi can still work on its own if the
GPU is unreachable.

## How the Pi reaches the GPU

We only have SSH to the GPU, so we use SSH tunnels. The Pi opens an SSH connection
to the GPU and forwards the ports the inference servers listen on back to the Pi's
own localhost. ZERO then talks to localhost as if the servers were running on the
Pi, and the traffic travels through the encrypted SSH connection. Nothing needs to
be exposed to the public internet, and no router changes are needed.

To keep the tunnel alive across drops and reboots we use autossh (or a small
systemd service that restarts the tunnel if it dies).

## What runs where

GPU machine:
- Ollama, serving a larger language model than the Pi can run.
- A Whisper server running large-v3-turbo for speech to text.

Pi:
- Audio capture and playback.
- Wake word and voice activity detection.
- Piper for text to speech (already fast on the Pi, no reason to move it).
- The SSH tunnel to the GPU.
- ZERO itself, pointed at localhost for the language model and speech to text.

## Phase 1: language model on the GPU

This is the easy win and we do it first. ZERO already talks to Ollama over HTTP
through the `llm.host` setting, so almost no code changes.

Steps:
1. Confirm the Pi can SSH into the GPU directly (not just from the laptop).
2. On the GPU, run Ollama and pull a model that fits its memory. Likely something
   in the 14B to 32B range depending on VRAM.
3. On the Pi, open an SSH tunnel forwarding the GPU's Ollama port. The Pi already
   runs its own Ollama on the default port, so we forward to a different local
   port to avoid a clash, or stop the local one.
4. Point `llm.host` in config.yaml at the forwarded port.
5. Test a few turns and check the reply latency.

Expected result: replies start faster and stay fast even deep into a conversation,
and the model is noticeably smarter than the 3B.

Fallback: if the tunnel is down, switch `llm.host` back to the Pi's local Ollama.

## Phase 2: speech to text on the GPU

Once the language model works over the tunnel, we move Whisper.

Steps:
1. On the GPU, run a Whisper server. Options are faster-whisper with a small HTTP
   wrapper, or the whisper.cpp server. Load large-v3-turbo.
2. On the Pi, forward the server's port through the same SSH tunnel.
3. Add a new STT engine to ZERO that sends the captured audio to the server and
   reads back the text. Because of the factory pattern, this is a drop-in engine
   selected in config.yaml, and main.py does not change.
4. Switch the STT engine in config to the remote one and test accuracy and speed.

Expected result: large-model transcription accuracy at roughly the speed we get
now from base.en, because the GPU does the encode that was crushing the Pi CPU.

Fallback: set the STT engine back to whispercpp with base.en for offline use.

## Things to decide before building

- Can the Pi SSH into the GPU on its own? ZERO runs on the Pi, so the Pi needs the
  connection, not just the laptop.
- Is the GPU always on, or does it stop when idle? If it stops, we need a way to
  handle that, or accept that ZERO only works when the GPU is up.
- How much VRAM does the GPU have? This decides the language model size and whether
  we can run Whisper and the LLM at the same time.
- What is already installed on the GPU: Ollama, Python, CUDA, any Whisper build.

## Order of work

1. Verify the Pi can SSH to the GPU.
2. Set up the persistent SSH tunnel.
3. Phase 1: language model over the tunnel, test, keep local fallback.
4. Phase 2: Whisper server, remote STT engine, test, keep local fallback.
5. Tidy up: make the tunnel start on boot, document the setup, add a config switch
   to flip between local and GPU modes.

## Risk and rollback

The main risk is depending on the GPU and the network. We reduce it by keeping the
local engines installed and a config switch to return to fully local mode. Every
change is in config.yaml or an SSH tunnel, so rolling back is changing a setting,
not rewriting code.
