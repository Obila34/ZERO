# ZERO — Offline Voice Assistant on Raspberry Pi 5

**Status:** Working end-to-end. You can hold a real conversation. Still polishing speed.
**Last updated:** 2026-06-25
**Branch:** `offline_v5`

---

## 1. What we built

A voice assistant that runs **completely on a small Raspberry Pi 5 computer** — no
internet, no cloud, no big servers. You say a wake word once, then just talk to it
like a person. It listens, understands, thinks, and talks back, all on the device.

Three things happen inside it:
1. It **hears** you and turns your speech into text.
2. It **thinks** using a local AI model to come up with a reply.
3. It **speaks** the reply out loud.

Everything is private and offline.

---

## 2. How it works (the flow)

```
You talk ─► Microphone ─► "Is someone talking?" ─► Turn speech into text
                                                          │
                                                          ▼
        Speaker plays it ◄─ Turn text into speech ◄─ AI writes a reply
```

| Step | What it does | Tool used |
|------|--------------|-----------|
| Wake word | Listens for "Hey Jarvis" to start | openWakeWord |
| Speech detection | Notices when you start/stop talking | webrtcvad |
| Speech-to-text | Writes down what you said | Whisper (`base.en`) |
| The "brain" | Comes up with a reply | Qwen 2.5 (3B) via Ollama |
| Text-to-speech | Says the reply out loud | Piper voice |

---

## 3. The choices we made (and why)

- **We did NOT use a huge AI model.** A bigger model is smarter but far too slow on
  a small Pi — replies would crawl out word by word. We picked a **mid-size model**
  that's the sweet spot: smart enough, but still quick.
- **We picked Qwen over the alternatives.** The smallest models (like TinyLlama)
  made things up and gave poor answers. Qwen is reliable and to-the-point.
- **We keep the AI model "warm" in memory** so it's ready to answer instantly,
  instead of reloading every time.
- **We kept the design simple.** Fancier "answer before the user finishes" tricks
  sound good on paper but overload the little Pi and make the audio stutter. Simple
  and steady wins here.

---

## 4. Everything we ran into along the way

Nothing here is left out — this is the full story, start to finish.

### 4a. Just getting it set up

| # | What went wrong | What we did |
|---|-----------------|-------------|
| 1 | GitHub wouldn't let us copy the code to the Pi with a password | Used a secure access token instead |
| 2 | The code folder on the Pi got wiped out | Restored it cleanly from GitHub |
| 3 | The wake-word software refused to install (incompatible with the Pi's Python) | Installed it a different way that works |
| 4 | The speech-detection software wouldn't start | Installed a specific older support library |
| 5 | Our first choice for speech detection secretly needed the internet | Swapped it for a lightweight offline one |
| 6 | A download link for the speech model was broken | Found and used the correct file |
| 7 | The folder holding the AI models got deleted | Re-downloaded the models |
| 8 | The assistant had no voice — the speaking software was missing | Installed the voice software |
| 9 | The microphone and speaker settings reset every time the Pi restarts | Re-apply them on each startup (a permanent fix is still on the list) |

### 4b. Making it actually work well

| # | What went wrong | What we did | Result |
|---|-----------------|-------------|--------|
| 10 | The microphone kept failing to start | Pointed it straight at the USB mic and made it retry | Reliable mic |
| 11 | It "heard" silence and invented random text | Made sure it listens to the correct microphone | Real, accurate text |
| 12 | The logs were flooded with noise, hard to read | Quieted the spam | Clean logs |
| 13 | **Writing down speech took ~15 seconds** | Stopped it from over-processing each clip | **15s → ~1.5s** |
| 14 | A trade-off between speed and accuracy on the speech model | Chose the option that's accurate *and* fast | Good balance |
| 15 | **The very first reply froze for ~28 seconds** | Load the AI model at startup and keep it ready | First reply is quick |
| 16 | It only answered ONE question, then needed the wake word again | Rebuilt it as a flowing conversation — talk freely, it stops on "goodbye" or after a long silence | Real back-and-forth |
| 17 | Replies were too short and clipped | Allowed slightly longer, more natural answers | Sounds human |
| 18 | **It froze for ~40 seconds deep in a chat** | It was re-reading the whole conversation each time; trimmed how much it re-reads | **40s → ~10s** |
| 19 | It picked up **background people and even its own voice** | Mutes the mic while it's busy, ignores crowd noise, and only listens to the closest/loudest voice (you) | Stops reacting to the room |
| 20 | Awkward silence while it was thinking | It now says a quick "Let me think…" *while* preparing the real answer | Feels natural, no dead air |

---

## 5. Where it stands now (real numbers)

| Thing | Now |
|-------|-----|
| Time to write down your speech | **~1.5 seconds** (was 15) |
| Time for the AI to start replying (usual) | **~2 seconds** |
| Time for the AI to start replying (deep in a long chat) | **~10–17 seconds** ← the last rough edge |
| Overall feel | Smooth on most turns; the "thinking" line hides the slow ones |
| Memory used | About 1/3 of what the Pi has — plenty of room |

---

## 6. What's left to do

| Item | Plan |
|------|------|
| **The occasional 15-second pause** deep in a chat | Try a smaller, faster AI model — about twice as fast. This is the main next step. |
| Background noise in a busy room | We've reduced it in software; a **headset or directional mic** would fully solve it. |
| Audio settings reset on restart | Make them stick automatically. |
| Wake word is "Hey Jarvis", not "Hey Zero" | A custom "Hey Zero" needs extra training. |

---

## 7. How to run it

```bash
# once per startup: connect the speaker and set the mic/speaker
bluetoothctl connect <speaker>
pactl set-default-sink   <speaker>
pactl set-default-source <usb-mic>

# start the assistant
cd ~/Mzee/offline_v5 && source .venv/bin/activate
python -m zero.main

# then: say "Hey Jarvis", talk normally, and say "goodbye" to end.
```

---

*In short: we took it from "can't even install" to a private, offline assistant you
can genuinely chat with on a tiny computer — and we know exactly what the last
speed fix is.*
