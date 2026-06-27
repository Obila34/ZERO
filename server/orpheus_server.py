"""Orpheus TTS server for a GPU node — natural, expressive, streaming speech.

The Pi POSTs {"text","voice"} to /tts and gets back a WAV (24 kHz). Run this on a
DIFFERENT node from Gemma/Whisper if you can — Orpheus is a 3B served via vLLM and
wants most of a GPU's VRAM, so it doesn't share a 16 GB card well.

Setup on the node:
    pip install orpheus-speech vllm fastapi uvicorn
    python server/orpheus_server.py --port 9100

First run downloads canopylabs/orpheus-3b-0.1-ft.
Voices: tara, leah, jess, leo, dan, mia, zac, zoe.
"""
from __future__ import annotations

import argparse
import io
import time
import wave

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from orpheus_tts import OrpheusModel

app = FastAPI()
_model = None


class TTSReq(BaseModel):
    text: str
    voice: str = "tara"


@app.get("/health")
def health():
    return {"ok": _model is not None}


@app.post("/tts")
def tts(req: TTSReq):
    t = time.time()
    audio = b"".join(_model.generate_speech(prompt=req.text, voice=req.voice))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(audio)
    print(f"[orpheus] {time.time() - t:.2f}s -> {req.text[:60]!r}", flush=True)
    return Response(content=buf.getvalue(), media_type="audio/wav")


def main() -> None:
    global _model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="canopylabs/orpheus-3b-0.1-ft")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"[orpheus] loading {args.model} (vLLM, first run downloads it)...", flush=True)
    _model = OrpheusModel(model_name=args.model)
    print(f"[orpheus] ready on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
