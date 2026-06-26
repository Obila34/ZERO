"""Whisper server for the GPU — runs large-v3-turbo via faster-whisper.

The Pi POSTs a WAV (raw bytes) to /transcribe; this returns {"text": ...}. Runs on
the GPU where Turbo is fast (vs ~29s on the Pi CPU).

Setup on the GPU node:
    pip install faster-whisper fastapi uvicorn
    python whisper_server.py --model large-v3-turbo --port 9000

Then from the Pi, tunnel the port and point ZERO's stt.remote_url at it:
    ssh -fN -L 9000:localhost:9000 <gpu-alias>

Notes:
- First run downloads the model (cached after).
- If "large-v3-turbo" isn't recognized by your faster-whisper version, pass
  --model deepdml/faster-whisper-large-v3-turbo-ct2 instead.
- beam_size=1 (greedy) keeps latency low; bump it for a little more accuracy.
"""
from __future__ import annotations

import argparse
import io
import time

import uvicorn
from fastapi import FastAPI, Request
from faster_whisper import WhisperModel

app = FastAPI()
_model: WhisperModel | None = None
_lang = "en"


@app.get("/health")
def health():
    return {"ok": _model is not None}


@app.post("/transcribe")
async def transcribe(request: Request):
    data = await request.body()
    t = time.time()
    segments, _info = _model.transcribe(
        io.BytesIO(data), language=_lang, beam_size=1,
    )
    text = " ".join(s.text for s in segments).strip()
    dt = time.time() - t
    print(f"[whisper] {dt:.2f}s -> {text!r}", flush=True)
    return {"text": text, "seconds": round(dt, 3)}


def main() -> None:
    global _model, _lang
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default="en")
    args = ap.parse_args()
    _lang = args.language
    print(f"[whisper] loading {args.model} on {args.device} ({args.compute_type})...",
          flush=True)
    _model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print(f"[whisper] ready on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
