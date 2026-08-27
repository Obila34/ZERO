#!/usr/bin/env python3
"""Neural gesture sidecar — Phase E's GPU side (docs/NEURAL_GESTURES_PLAN.md).

Stateless-incremental: POST /gesture with a sentence's audio-so-far,
receive closure frames-so-far. GET /health reports the loaded model.
Stdlib only, same serving pattern as the AF-1 gateway.

    python server/gesture_server.py --model mock --port 8200

Models register in MODELS; the trained N3 model slots in beside the mock
without touching the protocol or the Pi.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from zero.expr.neural import EnergyMockModel  # noqa: E402


def _tcn_factory():
    # lazy: torch only loads when the tcn model is requested
    from zero.expr.model import TCNServeModel

    ckpt = os.environ.get("GESTURE_CKPT", "models/gesture/tcn_v1.pt")
    device = os.environ.get("GESTURE_DEVICE", "cpu")
    return TCNServeModel(ckpt, device=device)


MODELS = {"mock": EnergyMockModel, "tcn": _tcn_factory}
_model = None
_model_name = "?"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # quiet
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "model": _model_name})
        else:
            self._json(404, {"ok": False})

    def do_POST(self):
        if self.path != "/gesture":
            self._json(404, {"ok": False})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode())
            audio = np.frombuffer(bytes.fromhex(req["audio"]),
                                  dtype=np.float16).astype(np.float32)
            # sentence id doubles as the style-latent seed: same sentence
            # -> same frames on every incremental poll (no flicker)
            frames = _model.frames(audio, int(req["sr"]),
                                   seed=int(req.get("sentence", 0)))
            self._json(200, {"frames": frames})
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})


def main() -> int:
    global _model, _model_name
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock", choices=sorted(MODELS))
    ap.add_argument("--port", type=int, default=8200)
    a = ap.parse_args()
    _model_name = a.model
    _model = MODELS[a.model]()
    print(f"gesture sidecar: model={a.model} port={a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
