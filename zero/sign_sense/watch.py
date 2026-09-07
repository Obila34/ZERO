"""Sign-sense watcher — ZERO's first sign INPUT (fingerspell reading).

Watches the live camera feed for a hand spelling letters, asks the
sign-sense sidecar (server/sign_sense_server.py, lumen-codex on the GPU
box) what it sees, debounces the per-frame answers into whole words, and
posts each word onto the event bus — the same safe additive channel
timers and proactive nudges use, drained only at turn boundaries, so
this can never talk over or alter existing behavior.

Debouncing (per-frame -> letters -> words):
  * a letter is ACCEPTED when the sidecar returns the same known letter
    for `stable_frames` consecutive frames (a held handshape), and the
    same letter can't repeat until the hand changed or dropped between
    (fingerspelling doubles are separated by a small hand bounce -> seen
    here as instability);
  * a word ENDS when the hand disappears (or nothing is stable) for
    `word_gap_s`; single stray letters are dropped as noise unless they
    are meaningful alone (a, i).

Contract, same as every additive layer here: ships OFF
(`sign_sense.enabled: false`), never blocks (own thread, short HTTP
timeouts, backoff on a dead sidecar), never moves anything, and OFF is
bit-identical to the system as it was.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import urllib.request

from zero.events import Event
from zero.utils.logging import get_logger

log = get_logger("sign.sense")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_SINGLE_LETTER_WORDS = {"a", "i"}


class SignWatcher:
    def __init__(self, cfg, eyes, events, *, on_word=None):
        g = lambda k, d: cfg.get(k, d)   # noqa: E731
        self._eyes = eyes
        self._events = events
        self._on_word = on_word
        self._url = str(g("sign_sense.url",
                          "http://100.110.56.17:8210")).rstrip("/")
        self._fps = float(g("sign_sense.fps", 6.0))
        self._timeout = float(g("sign_sense.timeout_s", 0.6))
        self._stable = int(g("sign_sense.stable_frames", 3))
        self._word_gap = float(g("sign_sense.word_gap_s", 1.2))
        self._min_conf = float(g("sign_sense.min_confidence", 0.55))
        self._echo_sign = bool(g("sign_sense.echo_sign", True))
        self._fail_streak = 0
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sign-sense",
                                        daemon=True)
        self._thread.start()
        log.info("sign-sense watcher up (%s, %.0f fps)", self._url, self._fps)

    # ── sidecar round-trip (overridable in tests) ───────────────────────────
    def _ask(self, frame_rgb) -> dict | None:
        try:
            import cv2
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb,
                                                        cv2.COLOR_RGB2BGR))
            if not ok:
                return None
            body = json.dumps({"image": base64.b64encode(
                jpg.tobytes()).decode()}).encode()
            req = urllib.request.Request(
                f"{self._url}/spell", data=body,
                headers={"Content-Type": "application/json"})
            with _OPENER.open(req, timeout=self._timeout) as r:
                out = json.loads(r.read().decode())
            self._fail_streak = 0
            return out
        except Exception as e:
            self._fail_streak += 1
            if self._fail_streak == 5:
                log.warning("sign sidecar unreachable (%s) — watcher idles "
                            "with backoff", e)
            return None

    # ── the loop ────────────────────────────────────────────────────────────
    def _run(self) -> None:
        interval = 1.0 / max(1.0, self._fps)
        letters: list[str] = []
        cand, streak = None, 0
        last_seen = 0.0          # last time a stable letter was accepted
        last_any = 0.0           # last time any hand was in frame
        emitted_at_streak = False
        while not self._stop_evt.wait(interval):
            if self._fail_streak >= 5:
                # dead sidecar: probe rarely instead of hammering
                if self._stop_evt.wait(10.0):
                    break
            frame = None
            try:
                frame = self._eyes.raw_frame() if self._eyes else None
            except Exception:
                frame = None
            now = time.monotonic()
            res = self._ask(frame) if frame is not None else None
            if res is None or res.get("no_hand"):
                cand, streak, emitted_at_streak = None, 0, False
            else:
                last_any = now
                letter = res.get("letter")
                ok = (bool(res.get("known"))
                      and float(res.get("confidence", 0.0)) >= self._min_conf)
                if not ok:
                    cand, streak, emitted_at_streak = None, 0, False
                elif letter == cand:
                    streak += 1
                    if streak >= self._stable and not emitted_at_streak:
                        letters.append(letter)
                        last_seen = now
                        emitted_at_streak = True   # once per hold
                else:
                    cand, streak, emitted_at_streak = letter, 1, False
            # word boundary: quiet hands after letters were collected
            if letters and (now - max(last_seen, last_any)) > self._word_gap:
                word = "".join(letters)
                letters = []
                if len(word) < 2 and word not in _SINGLE_LETTER_WORDS:
                    log.info("sign-sense: dropped stray letter %r", word)
                    continue
                self._emit(word)

    def _emit(self, word: str) -> None:
        log.info("sign-sense: read fingerspelled %r", word)
        if self._on_word is not None:
            try:
                self._on_word(word)
            except Exception as e:
                log.warning("sign-sense on_word failed: %s", e)
        if self._events is not None:
            self._events.post(Event(
                kind="sign",
                text=f"I read your hand — you spelled {word.upper()}.",
                meta={"open_conversation": True, "sign_word": word,
                      "echo_sign": self._echo_sign}))

    def stop(self) -> None:
        self._stop_evt.set()


def build_sign_sense(cfg, eyes, events, *, on_word=None):
    """The watcher, or None when gated off (`sign_sense.enabled`, default
    false — OFF is bit-identical to the system as shipped)."""
    if not cfg.get("sign_sense.enabled", False):
        return None
    if eyes is None:
        log.info("sign-sense enabled but no camera — skipped")
        return None
    try:
        return SignWatcher(cfg, eyes, events, on_word=on_word)
    except Exception as e:
        log.warning("sign-sense failed to start — skipped: %s", e)
        return None
