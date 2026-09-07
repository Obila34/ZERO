"""Sign-along — ZERO signs what it says (speech -> sign, live).

The classroom mission in one module: while ZERO speaks, the content
words its dictionary knows are signed in spoken order (sign-supported
speech — not full grammatical ASL/KSL translation, which needs a gloss
reordering model; this is the honest, useful first rung deaf education
actually uses).

Wiring: a wrapper around the SpeechTap listener chain. Living Hands'
scheduler stays attached and receives every event untouched; this
wrapper additionally watches sentence text (on_audio, ahead of
audibility) and, at each sentence's FIRST playout, hands the picked
words to the sign engine's sequence player on a worker thread — the
tap's calls stay microseconds, the speech path is never blocked, and
the sign track's priority already arbitrates hands between signing and
gesturing (sign wins, by design: when interpreting, signing IS the
hand behavior).

If the engine is still signing the previous sentence, the new one is
SKIPPED, not queued — signs must never lag sentences behind the voice.
"""
from __future__ import annotations

import re
import threading

from zero.utils.logging import get_logger

log = get_logger("sign.along")

_WORD = re.compile(r"[a-zA-Z']+")
_STOP = frozenset("""
a an the and or but so if then than that this these those is are was were
be been being am do does did have has had will would can could shall
should may might must of to in on at by for with from as it its it's im
i'm ive dont don't uh um oh hey well just very really quite there here
what when which how my mine your yours our ours their theirs his hers i
""".split())


class SignAlong:
    def __init__(self, cfg, inner, engine):
        g = lambda k, d: cfg.get(k, d)   # noqa: E731
        self._inner = inner              # Living Hands scheduler (or None)
        self._engine = engine
        self._max_words = int(g("sign.along.max_words_per_sentence", 3))
        self._min_len = int(g("sign.along.min_word_len", 3))
        self._pending: dict[int, list[str]] = {}
        self._fired: set[int] = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_evt = threading.Event()
        self._due: int | None = None
        self._worker = threading.Thread(target=self._run, name="sign-along",
                                        daemon=True)
        self._worker.start()

    # ── SpeechTap listener interface (must stay near-free) ──────────────────
    def on_audio(self, idx: int, sentence: str, piece, sr: int) -> None:
        if self._inner is not None:
            self._inner.on_audio(idx, sentence, piece, sr)
        with self._lock:
            if idx not in self._pending and idx not in self._fired:
                self._pending[idx] = self._pick(sentence)
                for k in [k for k in self._pending if k < idx - 3]:
                    self._pending.pop(k, None)

    def on_playout(self, idx: int, n_samples: int) -> None:
        if self._inner is not None:
            self._inner.on_playout(idx, n_samples)
        with self._lock:
            if idx in self._fired or idx not in self._pending:
                return
            self._fired.add(idx)
            for k in [k for k in self._fired if k < idx - 8]:
                self._fired.discard(k)
            self._due = idx
        self._wake.set()

    # ── worker: the only place the engine is touched ────────────────────────
    def _run(self) -> None:
        while not self._stop_evt.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                idx, self._due = self._due, None
                words = self._pending.pop(idx, []) if idx is not None else []
            if not words:
                continue
            try:
                if self._engine.busy or self._engine.estopped:
                    log.info("sign-along: still signing — sentence skipped")
                    continue
                played = self._engine.sign_sequence(words)
                if played:
                    log.info("sign-along: signing %s", played)
            except Exception as e:     # never let signing break speaking
                log.warning("sign-along failed: %s", e)

    def _pick(self, sentence: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for w in _WORD.findall(sentence or ""):
            lw = w.lower().replace("'", "")
            # dictionary membership is the REAL filter — min_len only
            # guards against noise words the dictionary happens to hold
            if lw in _STOP or len(lw) < 2 or lw in seen:
                continue
            if self._engine.knows_sign(lw):
                out.append(lw)
                seen.add(lw)
                if len(out) >= self._max_words:
                    break
        return out

    def stop(self) -> None:
        self._stop_evt.set()
        self._wake.set()


def build_sign_along(cfg, engine):
    """Wrap the current tap listener with sign-along, or None when gated
    off (`sign.along.enabled`) or there is no sign engine/dictionary."""
    if not cfg.get("sign.along.enabled", False):
        return None
    if engine is None or getattr(engine, "_dictionary", None) is None:
        log.info("sign-along enabled but no sign dictionary — skipped")
        return None
    from zero.expr.tap import TAP

    along = SignAlong(cfg, TAP.listener, engine)
    TAP.attach(along)
    log.info("sign-along up: ZERO signs what it says (max %d words/sentence)",
             along._max_words)
    return along
