# Learning from day-to-day interactions

ZERO gets better over time in **two** ways. Keeping them straight is what keeps
this honest.

1. **Memory (live, already on).** What it knows about *you* — facts, your last
   conversation, preferences — retrieved as it talks. This changes behaviour
   *today*, with no training. See `zero/memory/`.
2. **Training (offline, this doc).** Periodically fine-tuning the *models* on how
   people actually talk to it: phrasing, code-switching (Swahili/English),
   accents. This changes the model's *instincts*, and it happens in batches on
   the GPU node — not in real time, and not on the Pi.

> **What this is not.** There is no autonomous, real-time self-improvement here,
> and there is no path where the robot "browses YouTube on its own and becomes
> AGI." No system today does that safely. What *is* real is a compounding loop:
> it captures every conversation, you retrain adapters on a cadence, and it comes
> back measurably better. Do that for months and the ZERO of tomorrow genuinely
> isn't the ZERO of today — through engineering, not magic.

## The loop

```
  live conversations ──▶ data/corpus/interactions.jsonl   (written automatically)
        │
        ├─ scripts/export_corpus.py ──▶ data/train/chat.jsonl   (chat pairs)
        │
        └─ LoRA fine-tune on the GPU node ──▶ adapter ──▶ hot-swap into Ollama
```

### 1. Collect (automatic, on the Pi)
With `learning.corpus.enabled: true` (default), every ended session is appended
to `data/corpus/interactions.jsonl` — **one record per speaker**, already split
by voice so a stranger's speech never contaminates another's example:

```json
{"ts": 1789..., "speaker": 1, "speaker_kind": "known", "turns": [{"role":"user","text":"..."}, {"role":"assistant","text":"..."}]}
{"ts": 1789..., "speaker": -2, "speaker_kind": "guest", "turns": [...]}
```

`speaker`: positive = an enrolled person, negative = a provisional guest
(`guest-N`), null = anonymous. Writing is **privacy-gated** — a session ZERO
isn't allowed to remember (strict/guarded bystander mode) is not recorded.

### 2. Export (on any box)
```bash
python scripts/export_corpus.py --in data/corpus/interactions.jsonl \
    --out data/train/chat.jsonl --min-turns 4
# only real enrolled people:  --kinds known
```
Each record becomes a `{"messages": [...]}` chat example with ZERO's current
persona prepended (imported from `zero/llm/persona.py`, so it can't drift).

### 3. Fine-tune (on the GPU node)
Use any LoRA trainer that eats chat JSONL — e.g. **Unsloth**, **Axolotl**, or
`llama.cpp`'s finetune. Sketch with Unsloth:

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("gemma-2-9b-it", load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)
# load data/train/chat.jsonl, apply the chat template, train 1-3 epochs, low LR
```

Keep it gentle: **1–3 epochs, low learning rate, a held-out slice for eval.**
Fine-tuning can *degrade* as easily as improve — always compare the adapter
against the base on a few real prompts before shipping it.

### 4. Ship
Merge/convert the adapter to GGUF, register it with Ollama
(`ollama create zero-gemma -f Modelfile`), and point `llm.model` at it. Keep the
base model around; reverting is a one-line config change.

## Accents / STT (harder, optional)
The corpus above is **text** (LLM style). To adapt *hearing* to your household's
accents you need the **audio** too — that's a bigger change (saving utterance WAVs
keyed to the same speaker ids) and a Whisper fine-tune, which is easy to get
wrong. Treat it as a separate, later project; the text loop above is the high-
value, low-risk place to start.

## Cadence
Weekly or monthly, not continuous — the GPU is shared with Gemma/Whisper/Orpheus/
YOLO and can't train while serving. A simple cron on the GPU node that runs steps
2–4 and swaps the adapter is plenty.
