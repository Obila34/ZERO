"""Gloss translation — from sign-supported speech to actual sign ORDER.

Today sign-along signs content words in ENGLISH word order. Real sign
languages have their own grammar (topic-comment: "STORE I GO TOMORROW",
not "I will go to the store tomorrow"). The brain is a language model —
reordering into gloss is its native skill, so each spoken sentence gets
an ASYNC side-call to the same vLLM that wrote it: "render this as ASL
gloss." The call races the sentence's own playout; a miss or timeout
falls back to the existing word-picker, so this layer can only improve
the signing, never stall it.

The model outputs free gloss text; the CALLER (sign-along's picker)
filters every token through the dictionary/spell gates exactly as it
does its own picks — so a hallucinated gloss word simply doesn't play.
Ships dark: sign.along.gloss.enabled, default false.
"""
from __future__ import annotations

import json
import re
import urllib.request

from zero.utils.logging import get_logger

log = get_logger("sign.gloss")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_PROMPT = (
    "You translate English into ASL gloss for a signing robot. Rewrite "
    "the sentence in ASL gloss word order: topic-comment structure, "
    "time-markers first, drop articles/copulas/prepositions where ASL "
    "does, ALL-CAPS single-word glosses separated by spaces (use common "
    "single signs; fingerspellable names stay as words). Output ONLY "
    "the gloss line, nothing else.")

_WORD = re.compile(r"[A-Za-z']+")


class GlossTranslator:
    def __init__(self, cfg):
        g = lambda k, d: cfg.get(k, d)   # noqa: E731
        self._url = str(g("llm.host", "")).rstrip("/")
        self._model = str(g("llm.model", "gemma4-12b"))
        self._timeout = float(g("sign.along.gloss.timeout_s", 1.5))
        self._max_words = 8

    def gloss(self, sentence: str) -> list[str] | None:
        """Gloss tokens (lowercased) for one sentence, or None on any
        failure — None ALWAYS means: use the plain word-picker."""
        if not self._url or not sentence or not sentence.strip():
            return None
        try:
            body = json.dumps({
                "model": self._model,
                "messages": [{"role": "system", "content": _PROMPT},
                             {"role": "user", "content": sentence.strip()}],
                "max_tokens": 48,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(
                f"{self._url}/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"})
            with _OPENER.open(req, timeout=self._timeout) as r:
                out = json.loads(r.read().decode())
            text = out["choices"][0]["message"]["content"]
            words = [w.lower().replace("'", "")
                     for w in _WORD.findall(text)][:self._max_words]
            return words or None
        except Exception as e:
            log.debug("gloss call failed (%s) — plain picker", e)
            return None


def build_gloss(cfg):
    """The translator, or None when gated off (`sign.along.gloss.enabled`,
    default false — OFF is bit-identical to sign-along as shipped)."""
    if not cfg.get("sign.along.gloss.enabled", False):
        return None
    try:
        return GlossTranslator(cfg)
    except Exception as e:
        log.warning("gloss translator unavailable: %s", e)
        return None
