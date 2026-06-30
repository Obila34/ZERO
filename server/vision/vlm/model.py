"""Local vision-language model wrapper (CP5, GPU side).

``VLM`` turns ``(frame, facts, question, history)`` into the short, grounded
sentence the Pi speaks. Two backends are selectable in ``server/config.yaml``:

* **Qwen2-VL-2B-Instruct** (default) — a proper chat VLM. We feed it the frame,
  the CP4 facts (via ``vlm/prompt.py``), the question, and a little history using
  the processor's chat template.
* **Moondream2** — a smaller single-prompt VLM. History + facts + question are
  folded into one instruction; handy when VRAM is tight.

VRAM coexistence with the CP4 depth model
-----------------------------------------
The depth model (a few hundred MB for the Small checkpoint) and this VLM must fit
together in 12-16 GB. Qwen2-VL-2B is ~5 GB in fp16; set ``vlm.quantization: 4bit``
(needs ``bitsandbytes``) to roughly halve that and leave headroom. The model is
**lazy-loaded once** and cached on the instance, so ``/health`` and ``/facts``
never pay for it and the first ``/analyze`` call absorbs the load.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

try:
    from vlm.prompt import SYSTEM_PROMPT, build_user_text
except ImportError:  # pragma: no cover - import path fallback (cwd=repo root)
    from server.vlm.prompt import SYSTEM_PROMPT, build_user_text

try:
    from shared.schemas import SceneFact
except ImportError:  # pragma: no cover - import path fallback
    from server.shared.schemas import SceneFact


# Roles we accept from the rolling history; anything else is dropped so a stray
# "system" turn from the client can't override our grounding instruction.
_HISTORY_ROLES = {"user", "assistant"}


class VLM:
    """Loads the configured VLM and generates spoken replies, grounded in facts."""

    def __init__(self, config: Optional[dict] = None) -> None:
        try:
            from config import load_config
        except ImportError:  # pragma: no cover - import path fallback
            from server.config import load_config

        cfg = config or load_config()
        vlm = cfg.get("vlm", {})
        runtime = cfg.get("runtime", {})

        self._model_id = str(vlm.get("model", "Qwen/Qwen2-VL-2B-Instruct"))
        self._backend = str(vlm.get("backend", "qwen")).lower()
        self._quant = str(vlm.get("quantization", "none")).lower()
        self._max_new_tokens = int(vlm.get("max_new_tokens", 96))
        self._temperature = float(vlm.get("temperature", 0.2))
        self._max_history_turns = int(vlm.get("max_history_turns", 6))
        self._send_image = bool(vlm.get("send_image", True))

        self._cache_dir = runtime.get("cache_dir") or None
        self._device_pref = str(runtime.get("device", "auto"))

        self._model = None
        self._processor = None  # processor (Qwen) or tokenizer (Moondream)
        self._device = None  # resolved on first load

    # ── loading ────────────────────────────────────────────────────────────────
    def _resolve_device(self) -> str:
        import torch

        if self._device_pref in ("cpu", "cuda"):
            return self._device_pref
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _quant_config(self):
        """A BitsAndBytesConfig for 4/8-bit, or None for full precision.

        Quantization needs CUDA + bitsandbytes; on CPU we silently fall back to
        full precision so the eval script still runs on a laptop.
        """
        if self._quant not in ("4bit", "8bit"):
            return None
        if self._device != "cuda":
            print(
                f"[vlm] quantization={self._quant} requested but device is "
                f"{self._device}; loading full precision instead.",
                file=sys.stderr,
            )
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError:  # pragma: no cover - bitsandbytes/transformers missing
            print(
                "[vlm] bitsandbytes not available; loading full precision.",
                file=sys.stderr,
            )
            return None
        if self._quant == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._device = self._resolve_device()
        if self._backend == "moondream":
            self._load_moondream()
        else:
            self._load_qwen()

    def _load_qwen(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "torch + transformers are required for the VLM. Install them from "
                "server/requirements.txt (a CUDA torch build for the GPU)."
            ) from exc

        # Prefer the dedicated Qwen2-VL class; fall back to the generic
        # vision2seq auto class for transformers versions that expose it that way.
        try:
            from transformers import Qwen2VLForConditionalGeneration as _Model
        except ImportError:  # pragma: no cover - older/newer transformers
            from transformers import AutoModelForVision2Seq as _Model

        print(
            f"[vlm] loading {self._model_id} (qwen) on {self._device} "
            f"quant={self._quant}",
            file=sys.stderr,
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        quant = self._quant_config()
        kwargs: dict = {"cache_dir": self._cache_dir}
        if quant is not None:
            kwargs["quantization_config"] = quant
            kwargs["device_map"] = "auto"  # bitsandbytes places the shards
            kwargs["torch_dtype"] = torch.float16
        elif self._device == "cuda":
            kwargs["torch_dtype"] = torch.float16

        model = _Model.from_pretrained(self._model_id, **kwargs)
        if quant is None:
            model = model.to(self._device)
        self._model = model.eval()

    def _load_moondream(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "torch + transformers are required for the VLM."
            ) from exc

        print(
            f"[vlm] loading {self._model_id} (moondream) on {self._device}",
            file=sys.stderr,
        )
        self._processor = AutoTokenizer.from_pretrained(
            self._model_id, cache_dir=self._cache_dir, trust_remote_code=True
        )
        kwargs: dict = {"cache_dir": self._cache_dir, "trust_remote_code": True}
        if self._device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(self._model_id, **kwargs)
        self._model = model.to(self._device).eval()

    # ── helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(frame_bgr: np.ndarray):
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _clean_history(self, history: list[dict]) -> list[dict]:
        """Keep only well-formed user/assistant turns, most recent N."""
        turns = []
        for item in history or []:
            role = str(item.get("role", "")).lower()
            content = item.get("content")
            if role in _HISTORY_ROLES and isinstance(content, str) and content.strip():
                turns.append({"role": role, "content": content.strip()})
        if self._max_history_turns > 0:
            turns = turns[-self._max_history_turns :]
        return turns

    # ── inference ──────────────────────────────────────────────────────────────
    def generate(
        self,
        frame_bgr: np.ndarray,
        facts: list[SceneFact],
        question: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        """Return a short, grounded spoken reply for ``question``.

        ``facts`` are the CP4 scene facts; they are injected as text so the model
        cannot invent distances/colors. ``history`` is a rolling list of
        ``{role, content}`` turns used for follow-up questions.
        """
        self._ensure_model()
        user_text = build_user_text(facts, question)
        turns = self._clean_history(history or [])
        if self._backend == "moondream":
            return self._generate_moondream(frame_bgr, user_text, turns)
        return self._generate_qwen(frame_bgr, user_text, turns)

    def _generate_qwen(
        self, frame_bgr: np.ndarray, user_text: str, history: list[dict]
    ) -> str:
        import torch

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        messages.extend(history)  # plain {role, content} string turns

        if self._send_image:
            user_content = [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]
            images = [self._to_pil(frame_bgr)]
        else:
            user_content = [{"type": "text", "text": user_text}]
            images = None
        messages.append({"role": "user", "content": user_content})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._temperature > 0,
                temperature=max(self._temperature, 1e-4),
            )
        # Drop the prompt tokens, decode only the new completion.
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)
        ]
        reply = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return reply.strip()

    def _generate_moondream(
        self, frame_bgr: np.ndarray, user_text: str, history: list[dict]
    ) -> str:
        pil = self._to_pil(frame_bgr)
        # Fold system rules + history + the grounded turn into one instruction.
        preamble = SYSTEM_PROMPT
        if history:
            convo = "\n".join(f"{t['role']}: {t['content']}" for t in history)
            preamble = f"{preamble}\n\nConversation so far:\n{convo}"
        prompt = f"{preamble}\n\n{user_text}"

        # Newer moondream exposes .query(); older builds use .answer_question().
        query = getattr(self._model, "query", None)
        if callable(query):
            out = query(pil, prompt)
            answer = out.get("answer") if isinstance(out, dict) else out
            return str(answer).strip()

        enc = self._model.encode_image(pil)
        answer = self._model.answer_question(enc, prompt, self._processor)
        return str(answer).strip()


def main(argv: Optional[list[str]] = None) -> int:
    """Smoke test: ``python -m server.vlm.model <image.jpg> [question]``."""
    import cv2

    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m server.vlm.model <image_path> [question]", file=sys.stderr)
        return 2
    frame = cv2.imread(argv[0])
    if frame is None:
        print(f"[vlm] could not read image: {argv[0]}", file=sys.stderr)
        return 1
    question = argv[1] if len(argv) > 1 else "What are you looking at?"

    vlm = VLM()
    reply = vlm.generate(frame, facts=[], question=question)
    print(f"Q: {question}\nA: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
