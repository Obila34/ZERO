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
import glob
import io
import os
import site
import sys
import time
import wave


def _ensure_cuda_libs() -> None:
    """llama.cpp's CUDA build needs libcudart/libcublas/etc. The pip nvidia-* wheels
    ship them but aren't on LD_LIBRARY_PATH by default. Find every nvidia lib dir and
    re-exec once with the path set — so you never set LD_LIBRARY_PATH by hand.
    """
    if os.environ.get("_ZB_CUDA_RELAUNCHED"):
        return
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    dirs: list[str] = []
    for base in roots:
        for so in glob.glob(os.path.join(base, "nvidia", "**", "lib", "*.so*"),
                            recursive=True):
            dirs.append(os.path.dirname(so))
    dirs = sorted(set(dirs))
    if dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + ([existing] if existing else []))
        os.environ["_ZB_CUDA_RELAUNCHED"] = "1"
        print(f"[orpheus-cpp] added {len(dirs)} CUDA lib dir(s) to LD_LIBRARY_PATH",
              flush=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_libs()

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from orpheus_cpp import OrpheusCpp  # noqa: E402

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
    # orpheus-cpp streams (sample_rate, int16 chunk) tuples; collect into one WAV.
    sr = 24000
    chunks: list[np.ndarray] = []
    for chunk_sr, samples in _model.stream_tts_sync(
        req.text, options={"voice_id": req.voice}
    ):
        sr = int(chunk_sr)
        chunks.append(np.asarray(samples, dtype=np.int16).reshape(-1))
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    print(f"[orpheus-cpp] {time.time() - t:.2f}s -> {req.text[:60]!r}", flush=True)
    return Response(content=buf.getvalue(), media_type="audio/wav")


def main() -> None:
    global _model
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--lang", default="en", help="en (tara/leo/...) — NOT es_it")
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="-1 = offload all layers to GPU (avoids CPU OOM / slow)")
    args = ap.parse_args()
    print(f"[orpheus-cpp] loading quantized Orpheus (lang={args.lang}, "
          f"n_gpu_layers={args.n_gpu_layers}, downloads GGUF + SNAC first run)...",
          flush=True)
    _model = OrpheusCpp(lang=args.lang, n_gpu_layers=args.n_gpu_layers, verbose=False)
    print(f"[orpheus-cpp] ready on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
