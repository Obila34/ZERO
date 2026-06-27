"""Quantized Orpheus TTS server — GGUF via llama.cpp + SNAC, using orpheus-cpp.

Lighter than the vLLM fp16 path (~2.5 GB vs ~6.6 GB), so it coexists with Gemma +
Whisper on one 16 GB GPU. Exposes the SAME endpoint as the other servers, so the
Pi's RemoteTTS works unchanged: POST /tts {"text","voice"} -> WAV (24 kHz).

Setup on the GPU node (CUDA build of llama-cpp-python so it uses the GPU):
    CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
    pip install orpheus-cpp fastapi uvicorn
    python server/orpheus_cpp_server.py --port 9100
First run downloads the quantized Orpheus GGUF + SNAC.
Voices: tara leah jess leo dan mia zac zoe.
"""
from __future__ import annotations

import argparse
import io
import time
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from orpheus_cpp import OrpheusCpp

app = FastAPI()
_model: OrpheusCpp | None = None


class TTSReq(BaseModel):
    text: str
    voice: str = "tara"


@app.get("/health")
def health():
    return {"ok": _model is not None}


@app.post("/tts")
def tts(req: TTSReq):
    t = time.time()
    # orpheus-cpp returns (sample_rate, int16 mono samples).
    sr, samples = _model.tts(req.text, options={"voice_id": req.voice})
    samples = np.asarray(samples, dtype=np.int16).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(samples.tobytes())
    print(f"[orpheus-cpp] {time.time() - t:.2f}s -> {req.text[:60]!r}", flush=True)
    return Response(content=buf.getvalue(), media_type="audio/wav")


def main() -> None:
    global _model
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="-1 = all layers on GPU (needs CUDA llama-cpp-python)")
    args = ap.parse_args()
    print("[orpheus-cpp] loading quantized Orpheus (downloads GGUF + SNAC first run)...",
          flush=True)
    _model = OrpheusCpp(n_gpu_layers=args.n_gpu_layers, verbose=False)
    print(f"[orpheus-cpp] ready on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
