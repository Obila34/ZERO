# ZERO — Offline Voice Assistant on Raspberry Pi 5

**Status:** Working end-to-end — you can hold a real conversation. Polishing speed.
**Last updated:** 2026-06-25 · **Branch:** `offline_v5`

---

## Overview

ZERO is a voice assistant that runs entirely on a single Raspberry Pi 5 with no
internet, no cloud, and no external servers — every part of it lives on the device
itself, which keeps it completely private and able to work anywhere. The idea was
to take the loose collection of speech, language, and audio pieces we had been
experimenting with and turn them into one coherent product: you say a wake word
once, and from there you simply talk to it like a person. Under the hood three
things happen in sequence — it listens and turns your speech into text, it feeds
that text to a local AI model that writes a reply, and it speaks the reply back out
loud through a Bluetooth speaker. The goal throughout has been to make those three
steps feel seamless enough that the seams disappear and it just feels like a
conversation.

## The approach and the stack

Early on we made a deliberate decision not to chase the biggest, most powerful AI
model we could quantize onto the Pi. A larger model is smarter, but on a small
CPU-only computer it produces words so slowly that the experience falls apart, so
we settled on a mid-sized model that sits in the sweet spot of being clever enough
to hold a conversation while still being quick to respond. For the same reason we
kept the overall design simple and linear rather than reaching for the fashionable
"start answering before the user has finished speaking" tricks — on a device this
small, trying to run speech recognition, the language model, and the voice all at
once simply starves the processor and makes the audio stutter. The pieces we landed
on are openWakeWord for the "Hey Jarvis" trigger, a lightweight voice-activity
detector to notice when you start and stop talking, Whisper (the `base.en` model)
to turn speech into text, Qwen 2.5 (a 3-billion-parameter model served through
Ollama) as the brain, and Piper for the spoken voice. We chose Qwen over the
smaller alternatives because the tiniest models tended to make things up and give
shallow answers, whereas Qwen stays reliable and to the point.

## Phase 1 — Getting it to run at all

The first stretch of work was less about intelligence and more about simply getting
every piece to install and cooperate on the Pi, and it threw up a steady run of
obstacles. Copying the code onto the Pi failed at first because GitHub no longer
accepts plain passwords, so we switched to a secure access token; then, partway
through, the working copy on the Pi got wiped and every file showed as deleted,
which we recovered by resetting it cleanly from GitHub. The wake-word software
refused to install because it expected an older version of Python than the Pi was
running, so we installed it a different way that sidesteps the incompatibility, and
the voice-detection library similarly wouldn't start until we pinned one of its
older supporting packages. We also discovered that our original choice for voice
detection quietly depended on the internet, which broke the whole "fully offline"
promise, so we replaced it with a lighter component that runs locally. Downloading
the AI models brought its own snags: one model link was simply broken and pointed
at a file that didn't exist, and at another point the entire folder holding the
models was deleted and had to be fetched again. Finally, the assistant had no voice
at all until we installed the missing speech software, and we found that the Pi
forgets its microphone and speaker settings every time it restarts — for now we
re-apply those on each startup, with a permanent fix still on the list.

## Phase 2 — Making it work, and work well

With everything installed, the focus shifted to the experience itself, and this is
where the most meaningful progress happened. The microphone was unreliable at
first and kept failing to start, so we pointed the software directly at the USB
microphone and had it retry on the occasional hiccup; related to that, it was
initially "hearing" silence and inventing random text, which disappeared once we
made certain it was listening to the correct microphone. The single biggest win
came from the speech-to-text step, which originally took around fifteen seconds for
even a short sentence because it was over-processing every clip — once we stopped
that waste, the same step dropped to roughly a second and a half, and we settled on
a model setting that is both fast and accurate. On the language side, the very
first reply used to freeze for nearly thirty seconds while the model loaded, so we
now load it the moment the assistant starts and keep it resident in memory so it is
always ready. Perhaps the most important change conceptually was turning ZERO from
a one-question-at-a-time tool into a genuine conversation: previously you had to say
the wake word before every single question, and now you say it once and simply talk,
with the assistant bowing out only when you say "goodbye" or after a long silence.
We also loosened the replies, which had become too short and clipped, into slightly
longer and more natural-sounding answers.

Two stubborn problems remained, and both were about the realities of a small device
in a real room. Deep into a conversation the assistant would sometimes freeze for
forty seconds, which turned out to be it re-reading the entire conversation from the
start on every turn; trimming how much history it carries cut that dramatically. And
because the microphone is an open webcam mic, it was picking up background people and
even its own voice played through the speaker, creating a backlog of nonsense that
it tried to answer — we addressed this by muting the microphone while it is busy
thinking or speaking, ignoring crowd-type noise, and only treating the closest,
loudest voice (yours) as real input. Finally, to mask the unavoidable thinking time
on a small processor, ZERO now speaks a short filler like "Let me think…" the instant
you finish, while it quietly prepares the real answer in the background, so there is
no dead air.

## Where it stands now

The result is a private, offline assistant you can genuinely chat with on a tiny
computer. Writing down your speech, once a fifteen-second bottleneck, now takes about
a second and a half. The AI usually begins replying within roughly two seconds, and
the spoken filler smooths over the moments when it takes longer. Memory use sits at
around a third of what the Pi has available, leaving comfortable headroom. The one
remaining rough edge is that, deep into a long conversation, the AI can still take
ten to seventeen seconds to begin answering, which is the next thing we intend to
fix by testing a smaller and roughly twice-as-fast model. Beyond that, the open
microphone could be improved further with a headset or directional mic for noisy
rooms, the audio settings should be made to survive a restart, and the wake word —
currently "Hey Jarvis" — could be trained into a custom "Hey Zero." In short, we
took ZERO from "won't even install" to a working, natural, fully offline voice
assistant, and we know exactly what the last speed fix is.

## Running it

```bash
# once per startup: connect the speaker and set the mic/speaker
bluetoothctl connect <speaker>
pactl set-default-sink   <speaker>
pactl set-default-source <usb-mic>

# start the assistant
cd ~/Mzee/offline_v5 && source .venv/bin/activate
python -m zero.main

# then say "Hey Jarvis", talk normally, and say "goodbye" to end.
```
