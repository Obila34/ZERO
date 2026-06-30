# ZERO: Build Journey and Architecture

**Status:** Working end to end. You hold a natural voice conversation; a Raspberry Pi
runs the front end and a GPU runs the heavy models.
**Last updated:** 2026-06-27 · **Branch:** `offline_v5`

---

## What ZERO is

You say a wake word once, then just talk. ZERO hears you, thinks, and talks back in
a natural voice, holding the thread of the conversation and remembering things about
you between sessions. It started as a fully on-device assistant on a Raspberry Pi and
grew into a Pi front end paired with a GPU that does the slow work.

## How it's put together now

The Pi is the part you interact with. The GPU is the muscle. They talk over an SSH
tunnel, so nothing is exposed to the internet.

```
You speak
   |
   v
+--------------------- Raspberry Pi ----------------------+
|  mic -> wake word -> speech detection -> record clip     |
|  conversation loop, memory (SQLite), Bluetooth playback  |
+--------------------------|------------------------------+
                           |  SSH tunnel (audio up, text/audio back)
                           v
+----------------------- GPU node ------------------------+
|  Whisper large-v3-turbo   audio   -> text                |
|  Gemma 4 (8B) via Ollama  text    -> reply               |
|  Orpheus (quantized)      reply   -> streamed speech     |
+---------------------------------------------------------+
```

The Pi handles the microphone, the wake word, deciding when you've started and
stopped talking, the memory store, and playing the reply through the Bluetooth
speaker. The GPU runs the three models. Audio goes up the tunnel, text and spoken
audio come back. If the GPU is unreachable, the Pi can fall back to small local
models (whisper base.en, a 3B via local Ollama, Piper) and keep working.

## The journey

**Phase 1: everything on the Pi.** The first build ran the whole pipeline on the
Pi's CPU. Getting it installed was its own fight: Python 3.13 had no wheel for the
wake-word library, the speech detector needed an older support package, our first
voice-activity detector secretly wanted the internet, and a couple of model
downloads pointed at files that didn't exist. Once it ran, the experience worked but
was rough.

**Phase 2: making the on-device version usable.** Speech-to-text took about fifteen
seconds per sentence because Whisper always processes a thirty-second window; capping
that window dropped it to roughly a second and a half. The first reply used to freeze
for about thirty seconds while the model loaded, so we load and pin it at startup. We
turned it from a one-question tool into a flowing conversation (wake once, talk
freely, it bows out on "goodbye" or a long silence), and gave it a small SQLite
memory so it remembers you across restarts.

**The wall, and the pivot.** We wanted a much more accurate speech model and a
smarter brain, but the Pi's CPU couldn't keep up. Whisper large-v3-turbo took
twenty-nine seconds for one clip on the Pi. So we moved the heavy models to a GPU
reached over SSH and turned the Pi into a thin client. The language model swap was
almost free, since the code already talked to Ollama over HTTP; we just pointed it at
the GPU. Speech-to-text became a small server on the GPU that the Pi sends audio to.

**The audio saga.** This took the longest. The mic turned out to be the wrong kind of
input at first, the microphone gain was clipping and feeding Whisper distortion that
it "translated" into nonsense, a real bug in our endpointer was chopping speech into
fragments, and getting sound out to a Bluetooth speaker meant wiring up the Pi's
audio system (PipeWire) properly. We also added a loudness gate so background voices
and room noise get dropped before they reach the model.

**Phase 3: making it feel human.** Piper's voice was robotic, so we moved to Orpheus
(a quantized 3B TTS on the GPU) for a natural voice. To kill the pauses, the speech
now streams: audio plays as it's generated instead of waiting for a whole sentence,
so the first words come back in about a second. We also gave it an actual personality
in the prompt (casual, a little playful, leans on memory) and emotion cues the voice
can perform.

## Where it stands

| Thing | Now |
|-------|-----|
| Speech to text | ~1s, large-v3-turbo on the GPU |
| Brain | Gemma 4 (8B) on the GPU, first token ~0.6s |
| Voice | Orpheus, streamed, first audio ~1s |
| Memory | persists across sessions (SQLite): durable facts + per-conversation episode summaries |
| Conversation | wake once, free back-and-forth, stops on "goodbye" |
| Barge-in | say the wake word again while it's talking to cut it off and take your turn |

## What's left

Voice ID is written but not switched on yet. We enroll your voice on the Pi and
enable it, and from then on ZERO only answers you. It runs on the Pi, no GPU needed.

Barge-in (cutting it off while it's talking) is half done. The audio player already
checks for an interrupt every 50 milliseconds, so finishing it mostly means keeping
the mic live during playback and tripping that check when it hears your voice.

The last loose end is persistence: the SSH tunnels and the audio defaults don't yet
survive a reboot, so startup still takes a few manual steps.

## Running it

Servers on the GPU node (kept alive with `setsid`):
```
python server/whisper_server.py --port 9000
python server/orpheus_cpp_server.py --port 9100
# Gemma runs as the Ollama service
```

On the Pi:
```
# tunnel the three services
autossh -M 0 -f -N -L 11435:localhost:11434 -L 9000:localhost:9000 -L 9100:localhost:9100 <gpu>
cd ~/Desktop/Mzee/offline_v5 && source .venv/bin/activate
python -m zero.main          # then say "Hey Jarvis"
```

The design stays swappable: every stage is chosen in `config.yaml`, so moving a model
between the Pi and the GPU, or back to local fallback, is a setting change rather than
a rewrite.
