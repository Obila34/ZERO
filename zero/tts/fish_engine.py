"""Fish Audio / OpenAudio S1-mini TTS — the expressive engine.

S1-mini performs ~50 emotion/non-verbal markers natively, e.g. (laughing),
(chuckling), (whispering). We translate ZERO's shared cue vocabulary into Fish's
parenthetical syntax, then run inference.

IMPORTANT (from the plan):
  * Use the PyTorch/CPU build of `fishaudio/openaudio-s1-mini`, NOT any MLX
    build — MLX is Apple-Silicon only and will not run on the Pi.
  * The exact inference invocation is version-specific (fish-speech 1.x vs S1).
    Rather than hard-code a path that may drift, the command is supplied via
    config (`tts.fish.infer_cmd`) and finalized during the Phase-6 benchmark.
    Placeholders: {model_dir} {text} {out}. It must produce a WAV at {out}.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from zero.tts.base import TTS
from zero.utils.logging import get_logger

log = get_logger("tts.fish")

# ZERO shared cue tag -> Fish native marker.
CUE_TO_FISH = {
    "[laughs]": "(laughing)",
    "[chuckles]": "(chuckling)",
    "[sighs]": "(sighing)",
    "[hmm]": "(hmm)",
    "[pause]": "...",
}


def translate_cues(text: str) -> str:
    for tag, marker in CUE_TO_FISH.items():
        text = text.replace(tag, marker)
    # Fish reads emphasis fine as plain words; drop our *asterisk* markers.
    return text.replace("*", "")


class FishTTS(TTS):
    def __init__(self, model_dir: str, precision: str = "int8",
                 infer_cmd: str | None = None, sample_rate: int = 44100):
        self.model_dir = str(model_dir)
        self.precision = precision  # int8 | int4 | fp16 (transformer; codec stays fp16)
        self.infer_cmd = infer_cmd
        self.sample_rate = sample_rate
        if not Path(self.model_dir).exists():
            log.warning("Fish model dir not found: %s", self.model_dir)
        if not infer_cmd:
            log.warning(
                "tts.fish.infer_cmd is not set — finalize it in the Phase-6 spike "
                "(see scripts/setup_pi.sh). Fish synthesis will error until then."
            )
        log.info("Fish S1-mini configured (precision=%s)", precision)

    def synthesize(self, text: str) -> np.ndarray:
        if not self.infer_cmd:
            raise RuntimeError(
                "Fish infer_cmd not configured. Set tts.fish.infer_cmd in config.yaml."
            )
        prompt = translate_cues(text.strip())
        if not prompt:
            return np.zeros(0, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fish_out.wav"
            cmd = self.infer_cmd.format(
                model_dir=self.model_dir, text=prompt, out=str(out)
            )
            try:
                subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                import soundfile as sf  # lazy

                data, sr = sf.read(out, dtype="float32")
            except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as e:
                log.error("Fish synthesis failed: %s", e)
                return np.zeros(0, dtype=np.float32)

        if data.ndim > 1:
            data = data[:, 0]
        self.sample_rate = sr
        return data
