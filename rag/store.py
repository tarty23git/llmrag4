"""Persistent document index with hybrid retrieval (BM25 keywords + dense vectors, fused by RRF)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Plus

from rag.embeddings import Embedder

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60  # standard reciprocal-rank-fusion constant


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    index: int
    text: str


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None


def doc_id_for(source: str, text: str) -> str:
    return hashlib.sha256(f"{source}\0{text}".encode()).hexdigest()[:16]


class DocumentStore:
    """Chunks + embeddings persisted under `data_dir`. Thread-safe for the web server."""

    def __init__(self, data_dir: Path, embedder: Embedder | None):
        self.data_dir = Path(data_dir)
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self._bm25: BM25Plus | None = None
        self._token_sets: list[set[str]] = []
        self._lock = threading.RLock()
        self._load()

    # ---------- persistence ----------
    @property
    def _chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def _vectors_path(self) -> Path:
        return self.data_dir / "vectors.npy"

    @property
    def _meta_path(self) -> Path:
        return self.data_dir / "meta.json"

    def _load(self) -> None:
        if not self._chunks_path.exists():
            return
        with self._chunks_path.open(encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        meta = json.loads(self._meta_path.read_text()) if self._meta_path.exists() else {}
        if self.embedder is not None:
            if meta.get("embedding_model") == self.embedder.name and self._vectors_path.exists():
                self.vectors = np.load(self._vectors_path)
                if len(self.vectors) != len(self.chunks):
                    self.vectors = None
            if self.vectors is None and self.chunks:
                log.info("Embedding model changed or vectors missing; re-embedding %d chunks.", len(self.chunks))
                self.vectors = self.embedder.embed_documents([c.text for c in self.chunks])
                self._save()
        self._rebuild_bm25()

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._chunks_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        tmp.replace(self._chunks_path)
        if self.vectors is not None and self.embedder is not None:
            with self._vectors_path.open("wb") as f:
                np.save(f, self.vectors)
            self._meta_path.write_text(json.dumps({"embedding_model": self.embedder.name}))
        elif self._vectors_path.exists() and not self.chunks:
            self._vectors_path.unlink()

    def _rebuild_bm25(self) -> None:
        corpus = [tokenize(f"{c.source} {c.text}") for c in self.chunks]
        self._token_sets = [set(toks) for toks in corpus]
        # BM25+ keeps IDF positive even for terms in most chunks (Okapi zeroes them in small corpora).
        self._bm25 = BM25Plus(corpus) if corpus else None

    # ---------- mutation ----------
    def add_document(self, source: str, text: str, chunks: list[str]) -> tuple[str, int]:
        """Index a document's chunks. Re-adding the same source replaces the old version."""
        doc_id = doc_id_for(source, text)
        with self._lock:
            if any(c.doc_id == doc_id for c in self.chunks):
                return doc_id, 0  # identical content already indexed
            self._remove_where(lambda c: c.source == source)
            new = [Chunk(id=f"{doc_id}:{i}", doc_id=doc_id, source=source, index=i, text=t) for i, t in enumerate(chunks)]
            if not new:
                return doc_id, 0
            if self.embedder is not None:
                vecs = self.embedder.embed_documents([c.text for c in new])
                self.vectors = vecs if self.vectors is None or len(self.vectors) == 0 else np.vstack([self.vectors, vecs])
            self.chunks.extend(new)
            self._rebuild_bm25()
            self._save()
            return doc_id, len(new)

    def _remove_where(self, pred) -> int:
        keep = [i for i, c in enumerate(self.chunks) if not pred(c)]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            if self.vectors is not None:
                self.vectors = self.vectors[keep]
        return removed

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            removed = self._remove_where(lambda c: c.doc_id == doc_id)
            if removed:
                self._rebuild_bm25()
                self._save()
            return removed

    def clear(self) -> None:
        with self._lock:
            self.chunks, self.vectors = [], None
            self._rebuild_bm25()
            for p in (self._chunks_path, self._vectors_path, self._meta_path):
                p.unlink(missing_ok=True)

    # ---------- queries ----------
    def documents(self) -> list[dict]:
        with self._lock:
            docs: dict[str, dict] = {}
            for c in self.chunks:
                d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "source": c.source, "chunks": 0, "chars": 0})
                d["chunks"] += 1
                d["chars"] += len(c.text)
            return sorted(docs.values(), key=lambda d: d["source"].lower())

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + embeddings)" if self.embedder is not None else "BM25 keyword"

    def search(self, query: str, top_k: int = 6) -> list[SearchResult]:
        with self._lock:
            if not self.chunks or not query.strip():
                return []
            candidates = max(top_k * 4, 20)
            fused: dict[int, SearchResult] = {}

            if self._bm25 is not None:
                q_tokens = tokenize(query)
                if q_tokens:
                    q_set = set(q_tokens)
                    scores = self._bm25.get_scores(q_tokens)
                    # BM25+ gives every chunk a baseline score, so only rank chunks sharing a query term.
                    ranked = [i for i in np.argsort(-scores) if q_set & self._token_sets[i]][:candidates]
                    for rank, i in enumerate(ranked):
                        r = fused.setdefault(int(i), SearchResult(self.chunks[i], 0.0))
                        r.bm25_rank = rank + 1
                        r.score += 1.0 / (RRF_K + rank + 1)

            if self.embedder is not None and self.vectors is not None and len(self.vectors):
                q = self.embedder.embed_query(query)
                sims = self.vectors @ q
                for rank, i in enumerate(np.argsort(-sims)[:candidates]):
                    r = fused.setdefault(int(i), SearchResult(self.chunks[i], 0.0))
                    r.vector_rank = rank + 1
                    r.score += 1.0 / (RRF_K + rank + 1)

            return sorted(fused.values(), key=lambda r: r.score, reverse=True)[:top_k]
