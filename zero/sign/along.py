"""Sign-along — ZERO signs what it says (speech -> sign, live).

The classroom mission in one module: while ZERO speaks, the content
words its dictionary knows are signed in spoken order (sign-supported
speech — not full grammatical ASL/KSL translation, which needs a gloss
reordering model; this is the honest, useful first rung deaf education
actually uses). A word with NO sign that still matters — a name, a key
noun — is FINGERSPELLED inline, the way a human interpreter handles it,
rationed to one per sentence because spelling is slow and fluency dies
the moment the hands lag the voice.

Fluency is a TIME BUDGET, not a word count: the speech tap reports each
sentence's audio as it streams, so the picker takes as many signs as
actually fit the sentence's length (with a little spillover), in spoken
order, skipping words signed moments ago.

Wiring: a wrapper around the SpeechTap listener chain. Living Hands'
scheduler stays attached and receives every event untouched; picking
and playing happen on a worker thread — the tap's calls stay
microseconds and the speech path is never blocked. If the engine is
still signing the previous sentence, the new one is SKIPPED, not
queued — signs must never trail sentences behind the voice.
"""
from __future__ import annotations

import re
import threading
import time

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

_SIGN_COST_S = 2.2       # one dictionary sign, keyframes + settle
_RECENT_S = 60.0         # don't re-sign the same word within this window

# Common words that are unknown to the dictionary but NOT worth spelling
# (spelling is for names and rare content words, interpreter-style).
_COMMON = frozenset("""
more coming going thing things think thought right okay today tomorrow
yesterday really actually little people something anything everything
nothing someone anyone everyone always never sometimes maybe about
because before after around through over under again still also even
much many most some other another back down only good great nice sure
want wants wanted need needs needed know knows knew make makes made
take takes took come comes came look looks looked tell tells told
""".split())


class SignAlong:
    def __init__(self, cfg, inner, engine):
        g = lambda k, d: cfg.get(k, d)   # noqa: E731
        self._inner = inner              # Living Hands scheduler (or None)
        self._engine = engine
        self._budget_scale = float(g("sign.along.budget_scale", 1.25))
        self._spell_on = bool(g("sign.along.spell_fallback", True))
        self._spell_letter_s = float(g("sign.along.spell_letter_s", 0.45))
        self._spell_min_len = int(g("sign.along.spell_min_len", 4))
        self._sent: dict[int, dict] = {}    # idx -> {text, dur_s}
        self._fired: set[int] = set()
        self._recent: dict[str, float] = {}
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
        n = getattr(piece, "shape", None)
        dur = (n[0] if n else len(piece)) / max(1, int(sr))
        with self._lock:
            st = self._sent.get(idx)
            if st is None and idx not in self._fired:
                st = self._sent[idx] = {"text": sentence, "dur": 0.0}
                for k in [k for k in self._sent if k < idx - 3]:
                    self._sent.pop(k, None)
            if st is not None:
                st["dur"] += dur

    def on_playout(self, idx: int, n_samples: int) -> None:
        if self._inner is not None:
            self._inner.on_playout(idx, n_samples)
        with self._lock:
            if idx in self._fired or idx not in self._sent:
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
                st = self._sent.pop(idx, None) if idx is not None else None
            if not st:
                continue
            try:
                if self._engine.busy or self._engine.estopped:
                    log.info("sign-along: still signing — sentence skipped")
                    continue
                tokens = self._pick(st["text"], st["dur"])
                if not tokens:
                    continue
                played = self._engine.sign_sequence(
                    tokens, spell_letter_s=self._spell_letter_s)
                if played:
                    now = time.monotonic()
                    for w in played:
                        self._recent[w] = now
                    log.info("sign-along: %s", played)
            except Exception as e:     # never let signing break speaking
                log.warning("sign-along failed: %s", e)

    def _pick(self, sentence: str, dur_s: float) -> list:
        """Signs that FIT the sentence, in spoken order, plus at most one
        fingerspelled fallback for an important unknown word."""
        budget = max(2.5, float(dur_s) * self._budget_scale)
        now = time.monotonic()
        out: list = []
        seen: set[str] = set()
        spell_candidate: str | None = None
        words = _WORD.findall(sentence or "")
        for pos, w in enumerate(words):
            lw = w.lower().replace("'", "")
            if lw in _STOP or len(lw) < 2 or lw in seen:
                continue
            seen.add(lw)
            if now - self._recent.get(lw, -1e9) < _RECENT_S:
                continue
            if self._engine.knows_sign(lw):
                if budget >= _SIGN_COST_S:
                    out.append(lw)
                    budget -= _SIGN_COST_S
            elif (self._spell_on and spell_candidate is None
                  and len(lw) >= self._spell_min_len and lw.isalpha()
                  # names and rare content only — a mid-sentence capital
                  # (how TTS text writes a name), or a longer word that
                  # is not everyday vocabulary
                  and ((w[0].isupper() and pos > 0)
                       or (len(lw) >= 5 and lw not in _COMMON))):
                spell_candidate = lw
        if spell_candidate is not None:
            cost = len(spell_candidate) * self._spell_letter_s + 0.5
            # a sentence whose only signable content is a NAME gets it
            # spelled even slightly over budget — dropping the one word
            # that matters is worse than running a beat long
            if budget >= cost or not out:
                out.append(("spell", spell_candidate))
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
    log.info("sign-along up: ZERO signs what it says "
             "(time-budgeted, spell fallback %s)",
             "on" if along._spell_on else "off")
    return along
