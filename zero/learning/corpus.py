"""Interaction corpus — every day-to-day conversation saved as training data.

ZERO improves in two complementary ways:

  * MEMORY (live)     — what it knows about you, retrieved as it talks.
  * TRAINING (offline) — periodically fine-tuning the models on how people
                         actually speak to it: phrasing, code-switching, accents.

This is the raw material for the second. Each ended session is appended as
newline-delimited JSON — ONE record per speaker, holding that speaker's turns,
their id (a real person = positive, a provisional guest = negative, anonymous =
null), and a timestamp. Because the session is already split per speaker (voice-
owned), one stranger's speech never contaminates another's — the same property
that fixed memory now keeps the training data clean. An offline job
(``scripts/export_corpus.py`` + ``docs/TRAINING.md``) turns accumulated JSONL
into LoRA fine-tunes.

Privacy: the caller gates writing by the same rule as memory — a session ZERO
isn't allowed to remember is a session it doesn't record.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from zero.utils.logging import get_logger

log = get_logger("learning.corpus")


class Corpus:
    def __init__(self, path: str):
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "interactions.jsonl"
        self._lock = threading.Lock()  # JSONL lines must never interleave
        log.info("interaction corpus at %s", self._file)

    @staticmethod
    def _kind(speaker) -> str:
        if isinstance(speaker, int) and speaker > 0:
            return "known"
        if isinstance(speaker, int) and speaker < 0:
            return "guest"
        return "anonymous"

    def add_session(self, turns_by_speaker: dict, *, meta: dict | None = None) -> int:
        """Append one JSONL record per speaker. ``turns_by_speaker`` maps a
        speaker id (int person / negative guest / None) to that speaker's list of
        ``{"role", "text"}`` turns. Returns the number of records written."""
        now = time.time()
        written = 0
        with self._lock, self._file.open("a", encoding="utf-8") as f:
            for speaker, turns in turns_by_speaker.items():
                clean = [{"role": t["role"], "text": t["text"].strip()}
                         for t in turns if (t.get("text") or "").strip()]
                if not clean:
                    continue
                record = {
                    "ts": now,
                    "speaker": speaker,
                    "speaker_kind": self._kind(speaker),
                    "turns": clean,
                }
                if meta:
                    record["meta"] = meta
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
        if written:
            log.info("corpus: wrote %d speaker record(s)", written)
        return written
