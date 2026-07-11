"""HTTP control plane — the AF1 fusion surface.

Runs as a daemon thread INSIDE the Zero process, so every endpoint drives the
SAME brain the native mic loop uses: one Conversation, one SqliteMemory, one
tool registry (web_search, timers, remember/recall), one voice, one speaker.
Nothing is re-implemented — an AF1 push-to-talk turn is a ZERO turn.

Endpoints (all JSON unless noted):

  GET  /health                → {ok, service, state, ready}
  GET  /zero/status           → state + last external turn + degraded flags
  POST /zero/say              → {text, voice?}            speak on the Pi speaker
  POST /zero/turn             → raw audio body (webm/ogg/wav/mp4 — anything
                                 ffmpeg decodes) ?voice=&person_id=
                                 → STT → full brain turn → spoken on the Pi
                                 speaker → {ok, heard, reply}
  POST /zero/turn_text        → {text, voice?, speak?, person_id?} typed turn
  POST /zero/control          → {action: "sleep"}         end the open conversation

Voice: `voice` is an Orpheus speaker name (leo, tara, draco, …). It overrides
the configured default for THAT request only — AF1's picker keeps all its
voices, ZERO's own default stays untouched.

Transport notes: CORS is wide open (the Tauri app fetches Rust-side and needs
none; browser dev needs `*`). Audio decode shells out to ffmpeg — present on
the Pi (installed with the audio stack); errors surface as JSON, never a hang.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

log = logging.getLogger("zero.control")

_MAX_BODY = 32 * 1024 * 1024  # 32 MB — a minute of webm/opus is well under this


def decode_audio(data: bytes, sample_rate: int) -> np.ndarray:
    """Any container ffmpeg understands → mono float32 at `sample_rate`.
    Raises RuntimeError with ffmpeg's tail on failure (surfaced as HTTP 422)."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "f32le", "-ac", "1",
         "-ar", str(sample_rate), "pipe:1"],
        input=data, capture_output=True, timeout=30,
    )
    if proc.returncode != 0 or not proc.stdout:
        tail = (proc.stderr or b"").decode("utf-8", "replace")[-300:]
        raise RuntimeError(f"ffmpeg decode failed: {tail or 'no output'}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


class ControlServer:
    """Thin HTTP wrapper over a live ``Zero`` instance's external_* methods."""

    def __init__(self, zero, host: str = "0.0.0.0", port: int = 8090):
        self.zero = zero
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self.ready = False  # flipped by main once warmup completes

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        ctrl = self

        class Handler(BaseHTTPRequestHandler):
            # Silence the default stderr access log; route to our logger.
            def log_message(self, fmt, *args):  # noqa: A003
                log.debug("%s %s", self.address_string(), fmt % args)

            # ── plumbing ────────────────────────────────────────────────────
            def _json(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self._cors()
                self.end_headers()
                self.wfile.write(body)

            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def _read_body(self) -> bytes | None:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > _MAX_BODY:
                    return None
                return self.rfile.read(n)

            def _read_json(self) -> dict:
                raw = self._read_body()
                if not raw:
                    return {}
                try:
                    j = json.loads(raw.decode("utf-8"))
                    return j if isinstance(j, dict) else {}
                except (ValueError, UnicodeDecodeError):
                    return {}

            def do_OPTIONS(self):  # noqa: N802 — CORS preflight (browser dev)
                self.send_response(204)
                self._cors()
                self.end_headers()

            # ── GET ─────────────────────────────────────────────────────────
            def do_GET(self):  # noqa: N802
                path = urllib.parse.urlparse(self.path).path
                if path in ("/health", "/zero/health"):
                    self._json(200, {"ok": True, "service": "zero-control",
                                     "ready": ctrl.ready,
                                     "state": str(ctrl.zero.state.value)})
                elif path == "/zero/status":
                    self._json(200, ctrl.zero.external_status())
                else:
                    self._json(404, {"ok": False, "error": "not-found"})

            # ── POST ────────────────────────────────────────────────────────
            def do_POST(self):  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(parsed.query)
                qs = {k: v[0] for k, v in q.items() if v}
                try:
                    if parsed.path == "/zero/say":
                        j = self._read_json()
                        text = str(j.get("text") or "").strip()[:500]
                        if not text:
                            self._json(400, {"ok": False, "error": "empty-text"})
                            return
                        ctrl.zero.external_say(text, voice=j.get("voice"))
                        self._json(200, {"ok": True})

                    elif parsed.path == "/zero/turn":
                        raw = self._read_body()
                        if not raw:
                            self._json(400, {"ok": False, "error": "empty-audio"})
                            return
                        sr = ctrl.zero.cfg.get("audio.sample_rate", 16000)
                        try:
                            audio = decode_audio(raw, sr)
                        except (RuntimeError, subprocess.TimeoutExpired,
                                FileNotFoundError) as e:
                            self._json(422, {"ok": False,
                                             "error": f"decode: {e}"})
                            return
                        out = ctrl.zero.external_turn_audio(
                            audio, sr, voice=qs.get("voice"),
                            person_id=_int_or_none(qs.get("person_id")))
                        self._json(200 if out.get("ok") else 500, out)

                    elif parsed.path == "/zero/turn_text":
                        j = self._read_json()
                        text = str(j.get("text") or "").strip()[:1200]
                        if not text:
                            self._json(400, {"ok": False, "error": "empty-text"})
                            return
                        out = ctrl.zero.external_turn_text(
                            text, voice=j.get("voice"),
                            speak=bool(j.get("speak", True)),
                            person_id=_int_or_none(j.get("person_id")))
                        self._json(200 if out.get("ok") else 500, out)

                    elif parsed.path == "/zero/control":
                        j = self._read_json()
                        action = str(j.get("action") or "")
                        if action == "sleep":
                            ctrl.zero.external_sleep()
                            self._json(200, {"ok": True})
                        else:
                            self._json(400, {"ok": False,
                                             "error": f"unknown-action:{action}"})
                    else:
                        self._json(404, {"ok": False, "error": "not-found"})
                except Exception as e:  # a control request must never kill ZERO
                    log.exception("control request failed")
                    self._json(500, {"ok": False, "error": str(e)[:200]})

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        t = threading.Thread(target=self._httpd.serve_forever,
                             name="zero-control", daemon=True)
        t.start()
        log.info("control server on http://%s:%d (AF1 fusion surface)",
                 self.host, self.port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None


def _int_or_none(v) -> int | None:
    try:
        return int(v) if v is not None and str(v) != "" else None
    except (TypeError, ValueError):
        return None
