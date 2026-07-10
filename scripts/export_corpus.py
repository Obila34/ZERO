#!/usr/bin/env python3
"""Turn ZERO's interaction corpus into an LLM fine-tuning set.

Reads the JSONL that ZERO writes live (data/corpus/interactions.jsonl) and emits
chat-format JSONL — one {"messages": [...]} example per speaker record — ready
for a LoRA fine-tune of the conversational model on the GPU node.

The corpus is already split per speaker (voice-owned sessions), so one person's
speech never leaks into another's example. See docs/TRAINING.md for the full loop.

Usage:
    python scripts/export_corpus.py \
        --in data/corpus/interactions.jsonl \
        --out data/train/chat.jsonl \
        [--kinds known guest] [--min-turns 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Prepend the SAME persona the live model uses, imported so it can't drift from
# what ZERO actually is. Falls back to a stub if run outside the repo.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from zero.llm.persona import SYSTEM_TEMPLATE
except Exception:  # pragma: no cover - convenience only
    SYSTEM_TEMPLATE = "You are Zero, a warm, curious home robot."


def to_chat(record: dict, system: str) -> dict | None:
    """One corpus record -> a {"messages": [...]} chat example, or None if it has
    no usable user+assistant pair."""
    msgs = [{"role": "system", "content": system}]
    for t in record.get("turns") or []:
        content = (t.get("text") or "").strip()
        if not content:
            continue
        role = "user" if t.get("role") == "user" else "assistant"
        msgs.append({"role": role, "content": content})
    has_user = any(m["role"] == "user" for m in msgs)
    has_asst = any(m["role"] == "assistant" for m in msgs)
    return {"messages": msgs} if (has_user and has_asst) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/corpus/interactions.jsonl")
    ap.add_argument("--out", dest="out", default="data/train/chat.jsonl")
    ap.add_argument("--kinds", nargs="*",
                    default=["known", "guest", "anonymous"],
                    help="speaker_kind values to include")
    ap.add_argument("--min-turns", type=int, default=2,
                    help="minimum turns in a record to keep it")
    args = ap.parse_args()

    src = Path(args.inp)
    if not src.exists():
        print(f"no corpus at {src} — talk to ZERO first")
        return 1
    kinds = set(args.kinds)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    with src.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if (rec.get("speaker_kind") not in kinds
                    or len(rec.get("turns") or []) < args.min_turns):
                skipped += 1
                continue
            chat = to_chat(rec, SYSTEM_TEMPLATE)
            if chat is None:
                skipped += 1
                continue
            w.write(json.dumps(chat, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept} training examples to {out_path} ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
