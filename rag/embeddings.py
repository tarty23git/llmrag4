"""Local embedding backends. Embeddings run on-device; no API key needed."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class FastEmbedEmbedder:
    """ONNX sentence embeddings via fastembed (downloads the model once, then runs offline)."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize(np.stack(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.stack(list(self._model.query_embed([text])))[0])


def load_embedder(kind: str, model_name: str, cache_dir: str | None = None) -> Embedder | None:
    """Build the configured embedder, or return None (BM25-only) if disabled or unavailable."""
    if kind.lower() in {"none", "off", "bm25", ""}:
        log.info("Embeddings disabled; using BM25 keyword retrieval only.")
        return None
    if kind.lower() != "fastembed":
        raise ValueError(f"Unknown RAG_EMBEDDINGS backend: {kind!r} (use 'fastembed' or 'none')")
    try:
        return FastEmbedEmbedder(model_name, cache_dir=cache_dir)
    except Exception as exc:  # model download blocked, missing runtime, etc.
        log.warning(
            "Could not load embedding model %s (%s). Falling back to BM25 keyword retrieval only.",
            model_name,
            exc,
        )
        return None
