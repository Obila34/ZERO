"""Text embedders for semantic memory recall.

Primary: Ollama's /api/embeddings on the GPU (same host/tunnel the LLM already
uses). Fallback: a deterministic, dependency-free hashing embedder — character
trigrams hashed into a fixed-dim TF vector. Far weaker than a real model, but
offline, instant, and good enough to rank "wifi password" above "sister's
birthday" for a query about the network. The store records each vector's dim,
so switching backends never corrupts retrieval — mismatched rows just fall
back to keyword overlap.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("memory.embed")


class HashEmbedder:
    """Char-trigram hashing TF vector. Deterministic, no deps, no network."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def embed(self, text: str) -> np.ndarray | None:
        t = re.sub(r"\s+", " ", (text or "").lower()).strip()
        if not t:
            return None
        v = np.zeros(self.dim, dtype=np.float32)
        padded = f"  {t}  "
        for i in range(len(padded) - 2):
            gram = padded[i:i + 3]
            h = int.from_bytes(
                hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest(),
                "little")
            v[h % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None


class OllamaEmbedder:
    """Embeddings from the GPU Ollama over the existing tunnel."""

    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 10.0):
        self.url = host.rstrip("/") + "/api/embeddings"
        self.model = model
        self.timeout = float(timeout)
        self.dim: int | None = None  # learned from the first reply

    def embed(self, text: str) -> np.ndarray | None:
        text = (text or "").strip()
        if not text:
            return None
        import requests  # lazy; already a project dependency

        try:
            r = requests.post(self.url,
                              json={"model": self.model, "prompt": text},
                              timeout=self.timeout)
            r.raise_for_status()
            emb = np.asarray(r.json().get("embedding", []), dtype=np.float32)
        except Exception as e:
            log.debug("ollama embed failed: %s", e)
            return None
        if emb.size == 0:
            return None
        self.dim = emb.size
        n = float(np.linalg.norm(emb))
        return emb / n if n > 0 else None


class ResilientEmbedder:
    """Primary with a permanent-fallback switch: after ``max_failures``
    consecutive primary errors it stops paying the timeout every call and
    stays on the fallback (the store handles mixed dims gracefully)."""

    def __init__(self, primary, fallback, max_failures: int = 3):
        self._primary = primary
        self._fallback = fallback
        self._failures = 0
        self._max = int(max_failures)

    @property
    def name(self) -> str:
        if self._primary is not None and self._failures < self._max:
            return self._primary.name
        return self._fallback.name

    def embed(self, text: str) -> np.ndarray | None:
        if self._primary is not None and self._failures < self._max:
            v = self._primary.embed(text)
            if v is not None:
                self._failures = 0
                return v
            self._failures += 1
            if self._failures == self._max:
                model = getattr(self._primary, "model", "?")
                log.warning(
                    "primary embedder (%s, model=%s) failing — semantic recall "
                    "degraded to %s for this session. A chat model can't serve "
                    "/api/embeddings: pull a real embedding model (e.g. `ollama "
                    "pull nomic-embed-text`) and set "
                    "memory.embeddings.ollama_model to it.",
                    self._primary.name, model, self._fallback.name)
        return self._fallback.embed(text)


def build_embedder(backend: str, *, host: str = "", model: str = "",
                   hash_dim: int = 256):
    """backend: auto | ollama | hash | off -> embedder or None."""
    backend = (backend or "auto").lower()
    if backend == "off":
        return None
    if backend == "hash":
        return HashEmbedder(hash_dim)
    if backend == "ollama":
        return ResilientEmbedder(OllamaEmbedder(host, model), HashEmbedder(hash_dim))
    # auto: ollama when a host is configured, hash otherwise
    if host:
        return ResilientEmbedder(OllamaEmbedder(host, model), HashEmbedder(hash_dim))
    return HashEmbedder(hash_dim)


def keyword_overlap(query: str, text: str) -> float:
    """Cheap relevance for rows whose stored vector can't be compared (different
    embedder). Jaccard over lowercase word sets, ignoring 2-char noise."""
    q = {w for w in re.findall(r"[a-z0-9']+", query.lower()) if len(w) > 2}
    t = {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2}
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)
