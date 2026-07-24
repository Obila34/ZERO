"""Tier 2 — the streaming scene narrator.

A background thread that, every ``interval_s``, sends the current keyframe to
a vision-language model and writes its one-sentence understanding ("David is
at the desk typing; a mug is next to the laptop") into the world state. The
brains then READ that narration instantly instead of triggering their own
analysis — Tier 2 keeps the answer warm.

Backends (config ``world.narrator.backend``):

* ``analyze`` — the vision server's existing ``/analyze`` endpoint (Qwen2-VL,
  already deployed on zerolabs1). Runnable today on the shared 16 GB card.
* ``openai``  — any OpenAI-compatible ``/v1/chat/completions`` server with
  image input; this is the Cosmos 3 Nano Reasoner slot (vLLM serve with
  ``--hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}'``,
  see server/cosmos/). Swapping backends — or moving Tier 2 to a second GPU /
  edge box — is a config change, not a rewrite: the narrator only needs a URL.

Degradation: if the backend is unreachable the narrator backs off
(exponential, capped) and the world's narration simply goes stale — readers
see its age via ``narration_age_s()`` and stop quoting it. Perception never
breaks conversation.

Skip conditions per tick (sparse firing): no frame yet, or the scene has been
motionless since the last narration AND the scene graph is unchanged — a
still room does not need re-describing.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from zero.utils.logging import get_logger
from zero.world.state import WorldState

log = get_logger("world.narrator")

_PROMPT = ("In one short sentence, state who/what you see and what is "
           "happening (actions, notable changes). Objects likely present: "
           "{hint}. Plain factual description, no preamble.")


class Narrator:
    def __init__(self, world: WorldState,
                 frame_supplier: Callable[[], Optional[object]],
                 backend, *, interval_s: float = 3.0,
                 max_backoff_s: float = 60.0, rate_budget=None):
        """``backend`` is any object with
        ``narrate(frame_rgb, hint: str) -> str | None`` (see below).
        ``frame_supplier`` returns the latest RGB frame or None."""
        self._world = world
        self._frames = frame_supplier
        self._backend = backend
        self.interval_s = float(interval_s)
        self._max_backoff = float(max_backoff_s)
        # Tier 2 hard ceiling (Phase 3): a surprise storm degrades to routine
        # cadence instead of melting the shared GPU.
        self._rate_budget = rate_budget
        self._stop = threading.Event()
        self._wake = threading.Event()       # poke() -> narrate NOW
        self._poked = False
        self._thread: Optional[threading.Thread] = None
        self._last_narrated_version = -1
        self.last_latency_s: float = 0.0     # observability for benchmarks
        self.ticks = 0
        self.skips = 0
        self.failures = 0

    def poke(self) -> None:
        """Surprise gating (Phase 3): something unexpected happened — narrate
        immediately instead of waiting out the interval, and bypass the
        nothing-changed skip once."""
        self._poked = True
        self._wake.set()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="Narrator",
                                        daemon=True)
        self._thread.start()
        log.info("narrator running (every %.1fs, backend=%s)",
                 self.interval_s, type(self._backend).__name__)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()                     # unblock the interval wait
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ── loop ─────────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        backoff = 0.0
        while True:
            self._wake.wait(timeout=self.interval_s + backoff)
            self._wake.clear()
            if self._stop.is_set():
                return
            poked, self._poked = self._poked, False
            snap = self._world.snapshot()
            # Sparse firing: nothing moved and no scene-graph change since the
            # last narration -> the previous sentence is still true. A poke
            # (surprise) overrides — the gate exists to save compute, not to
            # sleep through the unexpected.
            if (not poked and not snap.motion_active
                    and self._last_narrated_version >= 0
                    and snap.objects_ts <= snap.narration_ts):
                self.skips += 1
                continue
            frame = self._frames()
            if frame is None:
                self.skips += 1
                continue
            if self._rate_budget is not None and not self._rate_budget.allowed():
                self.skips += 1
                continue
            hint = ", ".join(sorted({o.label for o in snap.objects})) or "none"
            t0 = time.perf_counter()
            try:
                text = self._backend.narrate(frame, hint)
            except Exception as e:
                self.failures += 1
                backoff = min(self._max_backoff, (backoff or 2.0) * 2.0)
                log.debug("narrate failed (backoff %.0fs): %s", backoff, e)
                continue
            self.last_latency_s = time.perf_counter() - t0
            backoff = 0.0
            self.ticks += 1
            if text:
                self._world.update_narration(text)
                self._last_narrated_version = self._world.version


# ── backends ─────────────────────────────────────────────────────────────────
class AnalyzeBackend:
    """The vision server's existing /analyze endpoint (Qwen2-VL today)."""

    def __init__(self, client):        # zero.vision.gpu_client.VisionClient
        self._client = client

    def narrate(self, frame_rgb, hint: str) -> str | None:
        reply = self._client.analyze(frame_rgb, [],
                                     question=_PROMPT.format(hint=hint))
        return (reply or "").strip() or None


class OpenAIVisionBackend:
    """Any OpenAI-compatible chat-completions server with image input — the
    Cosmos 3 Nano Reasoner slot (vLLM), or any other VLM on any host."""

    def __init__(self, url: str, model: str, timeout_s: float = 20.0,
                 max_tokens: int = 60, jpeg_quality: int = 80):
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.jpeg_quality = int(jpeg_quality)

    def narrate(self, frame_rgb, hint: str) -> str | None:
        import base64

        import cv2
        import requests

        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None
        b64 = base64.b64encode(buf).decode("ascii")
        resp = requests.post(self.url, timeout=self.timeout_s, json={
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": _PROMPT.format(hint=hint)},
            ]}],
        })
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return (text or "").strip() or None
