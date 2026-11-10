"""High-level RAG pipeline: ingest documents, retrieve, and answer with Claude."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anthropic

from rag.chunking import chunk_text
from rag.config import Settings
from rag.embeddings import Embedder, load_embedder
from rag.llm import ClaudeAnswerer, build_messages
from rag.loaders import extract_text, iter_files
from rag.store import DocumentStore, SearchResult

log = logging.getLogger(__name__)


def _source_payload(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "n": n,
            "source": r.chunk.source,
            "doc_id": r.chunk.doc_id,
            "chunk": r.chunk.index + 1,
            "score": round(r.score, 5),
            "text": r.chunk.text,
        }
        for n, r in enumerate(results, start=1)
    ]


def _api_error_message(exc: Exception) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return "Anthropic authentication failed. Set ANTHROPIC_API_KEY (see .env.example)."
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by the Anthropic API. Please retry shortly."
    if isinstance(exc, anthropic.BadRequestError):
        return f"The API rejected the request: {exc.message}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API error ({exc.status_code}): {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API. Check your network connection."
    if isinstance(exc, TypeError) and "api_key" in str(exc).lower():
        return "No Anthropic credentials found. Set ANTHROPIC_API_KEY (see .env.example)."
    return f"Unexpected error: {exc}"


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None | str = "auto",
        answerer: ClaudeAnswerer | None = None,
    ):
        self.settings = settings or Settings.from_env()
        s = self.settings
        if embedder == "auto":
            embedder = load_embedder(s.embeddings, s.embedding_model, cache_dir=str(s.data_dir / "models"))
        self.store = DocumentStore(s.data_dir / "index", embedder)
        self.answerer = answerer or ClaudeAnswerer(s.model, effort=s.effort, max_tokens=s.max_tokens)

    # ---------- ingestion ----------
    def ingest_text(self, source: str, text: str) -> dict:
        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        doc_id, added = self.store.add_document(source, text, chunks)
        return {"doc_id": doc_id, "source": source, "chunks": added, "skipped": added == 0}

    def ingest_bytes(self, filename: str, data: bytes) -> dict:
        return self.ingest_text(filename, extract_text(filename, data))

    def ingest_paths(self, paths: list[str | Path]) -> list[dict]:
        results = []
        for f in iter_files(paths):
            try:
                results.append(self.ingest_bytes(str(f), f.read_bytes()))
            except Exception as exc:
                log.warning("Skipping %s: %s", f, exc)
                results.append({"source": str(f), "error": str(exc)})
        return results

    # ---------- retrieval + generation ----------
    def retrieve(self, question: str, history: list[dict] | None = None) -> list[SearchResult]:
        # For follow-ups ("what about the second one?") include the previous user turn in the search.
        prev_user = next((t["content"] for t in reversed(history or []) if t.get("role") == "user"), "")
        query = f"{prev_user}\n{question}" if prev_user else question
        return self.store.search(query, top_k=self.settings.top_k)

    def ask_stream(self, question: str, history: list[dict] | None = None) -> Iterator[dict[str, Any]]:
        """Yield a 'sources' event, 'text' deltas, then 'done' (or 'error')."""
        history = history or []
        results = self.retrieve(question, history)
        yield {"type": "sources", "sources": _source_payload(results)}
        try:
            yield from self.answerer.stream(build_messages(question, results, history))
        except Exception as exc:
            log.warning("Answer generation failed: %s", exc)
            log.debug("Answer generation traceback", exc_info=True)
            yield {"type": "error", "message": _api_error_message(exc)}

    def ask(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        answer, sources, final = [], [], {}
        for event in self.ask_stream(question, history):
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "text":
                answer.append(event["text"])
            else:
                final = event
        return {"answer": "".join(answer), "sources": sources, **{k: v for k, v in final.items() if k != "type"}}
