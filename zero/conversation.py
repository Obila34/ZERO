"""Conversation memory — system prompt + the recent turns of the chat.

Two design points that matter on the Pi:

1. Cache-friendly trimming. Ollama caches the prompt prefix, so as long as we only
   APPEND messages each turn, it reprocesses just the new ones (fast, ~2s). The
   moment we drop the oldest message the prefix shifts and the whole context is
   re-read (slow, ~10-15s). So instead of trimming every turn, we let history grow
   to `trim_at` turns and only then trim back to `target` — most turns stay
   append-only (cache hit), and the expensive re-read happens rarely.

2. Long-term memory block. Durable facts loaded from SQLite are injected into the
   system message as a short block, so ZERO "remembers" across sessions without
   replaying whole past conversations (which would bloat the prompt).

3. In-session compaction. On a long session, trimmed-away turns are NOT discarded —
   they're folded into a short rolling summary that stays in the prompt. So context
   stays bounded (persona + memory + summary + a few live turns) no matter how long
   the session runs, keeping the shared GPU's KV cache small, without losing the
   thread. The summary is produced off the hot path (see main._compact) and installed
   via apply_summary(); everything here is just bookkeeping behind a small lock.
"""
from __future__ import annotations

import threading

from zero.llm.base import Message


class Conversation:
    def __init__(self, system_prompt: str, history_turns: int = 3,
                 trim_at_turns: int = 8):
        self.system_prompt = system_prompt
        # Counts are in user+assistant PAIRS.
        self.target_messages = history_turns * 2          # size we trim back DOWN to
        self.trim_at_messages = max(trim_at_turns, history_turns) * 2  # trigger
        self._history: list[Message] = []
        self._memory_block = ""  # durable facts injected per-session (set once at start)
        self._running_summary = ""          # rolling compaction of trimmed turns
        self._pending: list[Message] = []   # dropped turns awaiting summarisation
        self._lock = threading.Lock()       # guards _pending/_running_summary

    def set_memory(self, block: str) -> None:
        """Inject long-term memory into the system prompt for this conversation.
        Set once at conversation start so the prefix stays stable (cache-friendly).
        """
        self._memory_block = block or ""

    @property
    def memory_block(self) -> str:
        """The durable memory text currently injected into the system prompt.
        Read by per-turn recall so it can skip facts already present here (no
        point spending tokens saying the same fact twice)."""
        return self._memory_block

    def add_user(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})
        self._maybe_trim()

    def add_assistant(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})
        self._maybe_trim()

    def messages(self) -> list[Message]:
        system = self.system_prompt
        if self._memory_block:
            system = f"{system}\n\n{self._memory_block}"
        with self._lock:
            if self._running_summary:
                system = (f"{system}\n\nEarlier in this conversation (summary, so "
                          f"you can keep the thread):\n{self._running_summary}")
            # Pending turns are rendered RAW until the summary folds them in, so
            # nothing is lost from context in the window before compaction lands.
            pending = list(self._pending)
        return [{"role": "system", "content": system}, *pending, *self._history]

    # ── in-session compaction ─────────────────────────────────────────────────
    def pending_snapshot(self) -> tuple[str, list[Message]] | None:
        """(running_summary, trimmed-but-unsummarised turns) when there's
        compaction work to do, else None. The caller summarises off the hot path
        and feeds the result back via apply_summary()."""
        with self._lock:
            if not self._pending:
                return None
            return self._running_summary, list(self._pending)

    def apply_summary(self, new_summary: str, covered: list[Message]) -> None:
        """Install the rolling summary and drop the turns it now covers. Safe to
        call from a background thread. Turns trimmed AFTER the snapshot stay
        pending for the next pass.

        Stale-apply guard: if the conversation was reset while the summary was
        being computed (session ended), ``covered`` no longer matches the head of
        ``_pending`` — the summary belongs to a dead conversation and is dropped,
        so an old session can never ghost into a fresh one."""
        with self._lock:
            if self._pending[:len(covered)] != covered:
                return  # conversation moved on (reset) — stale summary, drop it
            self._running_summary = (new_summary or "").strip()
            self._pending = self._pending[len(covered):]

    def transcript(self) -> str:
        """Plain-text dump of the chat, for the end-of-session memory extraction."""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self._history)

    def _maybe_trim(self) -> None:
        # Only trim once we've grown past the trigger — keeps most turns append-only
        # (cache hit) and pays the expensive re-read only occasionally.
        if len(self._history) > self.trim_at_messages:
            dropped = self._history[:-self.target_messages]
            trimmed = self._history[-self.target_messages:]
            # Keep the window aligned to a user turn — a history that opens with a
            # dangling assistant message reads as replying to nothing.
            if trimmed and trimmed[0]["role"] == "assistant":
                dropped = dropped + trimmed[:1]  # summarise it, don't drop it
                trimmed = trimmed[1:]
            self._history = trimmed
            # Hold the dropped turns until the rolling summary folds them in, so
            # nothing is lost from context mid-session.
            with self._lock:
                self._pending.extend(dropped)

    def reset(self) -> None:
        self._history.clear()
        with self._lock:
            self._running_summary = ""
            self._pending = []
        # memory_block is reloaded from SQLite at the next conversation start
