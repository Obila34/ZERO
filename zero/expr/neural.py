"""Neural gesture client — Phase E's Pi side (docs/NEURAL_GESTURES_PLAN.md).

Streams each sentence's tapped audio to the gesture sidecar and caches the
closure-frame texture it returns; the HandScheduler asks `frames_for()`
per render tick and falls back to procedural the instant the answer is
None. The protocol is incremental-stateless — one POST per ~300 ms of new
audio, audio-so-far in, frames-so-far out — so a dead or slow sidecar
needs no cleanup: it just stops being consulted.

Everything here must obey the Living Hands contract: never block the
speech threads (network runs on this module's own worker), never move
anything itself (the scheduler owns the bus track), and OFF must be
bit-identical to the layer as it shipped.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("expr.neural")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class EnergyMockModel:
    """Plumbing-test model: closure texture from the audio energy envelope.

    This is NOT AI gestures — it is deliberately simple, clearly named,
    and exists so the whole client→server→blend→fallback pipe can be
    built and tested before a trained model exists (plan N0). It runs
    in-process (no server) when `expression.hands.neural.url` is 'mock'.
    """

    FRAME_HZ = 20.0
    FINGERS = ("thumb", "index", "middle", "ring", "pinky")

    def frames(self, audio: np.ndarray, sr: int,
               seed: int = 0) -> list[dict]:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        hop = max(1, int(sr / self.FRAME_HZ))
        n = len(x) // hop
        if n == 0:
            return []
        env = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
        if env.max() > 1e-6:
            env = env / env.max()
        out = []
        for i, e in enumerate(env):
            t = i / self.FRAME_HZ
            # right hand rides a ~250 ms delayed envelope and shifted
            # phase: the mock must exercise the genuinely-per-side path
            # the real model now uses (hands lead/lag in human data)
            er = env[max(0, i - 5)]
            cl_l, cl_r = {}, {}
            for k, f in enumerate(self.FINGERS):
                phase = 1.7 * t + 0.9 * k
                cl_l[f] = float(np.clip(
                    0.10 * e * (1.0 + 0.6 * np.sin(phase)), 0.0, 0.35))
                cl_r[f] = float(np.clip(
                    0.10 * er * (1.0 + 0.6 * np.sin(phase + 1.3)),
                    0.0, 0.35))
            out.append({"t": round(t, 3),
                        "closure": {f: 0.5 * (cl_l[f] + cl_r[f])
                                    for f in self.FINGERS},
                        "closure_l": cl_l, "closure_r": cl_r,
                        "wrist_deg": {"left": float(-3.0 * e),
                                      "right": float(3.0 * e)}})
        return out


class NeuralGestureClient:
    """Per-sentence sessions against the sidecar (or the in-process mock)."""

    def __init__(self, cfg):
        g = lambda k, d: cfg.get(k, d)   # noqa: E731
        self._url = str(g("expression.hands.neural.url", "mock")).rstrip("/")
        self._timeout = float(g("expression.hands.neural.timeout_s", 0.8))
        self._min_growth_s = 0.3
        self._mock = EnergyMockModel() if self._url == "mock" else None
        # sentence idx -> {"sr", "audio" (list of arrays), "fed_s",
        #                  "sent_s", "frames" [dict], "dead" bool}
        self._sess: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_evt = threading.Event()
        self._fail_streak = 0
        self._thread = threading.Thread(target=self._run,
                                        name="neural-gesture", daemon=True)
        self._thread.start()

    # ── fed from the SpeechTap path (via the scheduler) ─────────────────────
    def feed(self, idx: int, piece, sr: int) -> None:
        with self._lock:
            s = self._sess.get(idx)
            if s is None:
                s = self._sess[idx] = {"sr": int(sr), "audio": [],
                                       "fed_s": 0.0, "sent_s": 0.0,
                                       "frames": [], "dead": False}
                for k in [k for k in self._sess if k < idx - 2]:
                    self._sess.pop(k, None)
            p = np.asarray(piece, dtype=np.float32).reshape(-1)
            s["audio"].append(p)
            s["fed_s"] += len(p) / s["sr"]
        self._wake.set()

    # ── consumed by the scheduler's render ──────────────────────────────────
    def frames_for(self, idx: int, t_rel: float) -> dict | None:
        """The texture frame nearest t_rel for sentence idx, or None —
        and None ALWAYS means: use procedural."""
        with self._lock:
            s = self._sess.get(idx)
            if s is None or s["dead"] or not s["frames"]:
                return None
            frames = s["frames"]
        # frames are 1/FRAME_HZ apart, sorted; nearest lookup
        i = int(round(t_rel * EnergyMockModel.FRAME_HZ))
        if i < 0 or i >= len(frames):
            return None
        fr = frames[i]
        if abs(fr["t"] - t_rel) > 0.15:
            return None
        return fr

    @property
    def healthy(self) -> bool:
        return self._fail_streak < 3

    # ── the worker ──────────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop_evt.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                todo = [(idx, s) for idx, s in self._sess.items()
                        if not s["dead"]
                        and s["fed_s"] - s["sent_s"] >= self._min_growth_s]
            for idx, s in todo:
                self._infer(idx, s)

    def _infer(self, idx: int, s: dict) -> None:
        with self._lock:
            audio = np.concatenate(s["audio"]) if s["audio"] else None
            sr = s["sr"]
            fed = s["fed_s"]
        if audio is None:
            return
        try:
            if self._mock is not None:
                frames = self._mock.frames(audio, sr, seed=idx)
            else:
                body = json.dumps({
                    "sentence": idx, "sr": sr,
                    "audio": audio.astype(np.float16).tobytes().hex(),
                }).encode()
                req = urllib.request.Request(
                    f"{self._url}/gesture", data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST")
                with _OPENER.open(req, timeout=self._timeout) as r:
                    frames = json.loads(r.read().decode())["frames"]
            with self._lock:
                s["frames"] = frames
                s["sent_s"] = fed
            self._fail_streak = 0
        except Exception as e:
            self._fail_streak += 1
            if self._fail_streak == 3:
                log.warning("gesture sidecar unreachable (%s) — falling "
                            "back to procedural texture", e)
            with self._lock:
                if self._fail_streak >= 3:
                    s["dead"] = True     # this sentence stays procedural

    def stop(self) -> None:
        self._stop_evt.set()
        self._wake.set()


def build_neural(cfg):
    """The client, or None when gated off (`expression.hands.neural.enabled`,
    default false — OFF is bit-identical to the layer as shipped)."""
    if not cfg.get("expression.hands.neural.enabled", False):
        return None
    try:
        c = NeuralGestureClient(cfg)
        log.info("neural gesture client up (%s)",
                 "in-process mock" if c._mock else c._url)
        return c
    except Exception as e:
        log.warning("neural gesture client failed — procedural only: %s", e)
        return None
