"""FastAPI server: document upload/management and a streaming chat endpoint (Server-Sent Events)."""

from __future__ import annotations

import json
import logging

import anthropic
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.loaders import UnsupportedFileType
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    history: list[Turn] = Field(default_factory=list, max_length=40)


class ApiKeyRequest(BaseModel):
    api_key: str = Field(min_length=8, max_length=500)


class TextDoc(BaseModel):
    source: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)


def create_app() -> FastAPI:
    app = FastAPI(title="llmrag4", version="0.1.0")

    @app.get("/api/health")
    def health(rag: RAGPipeline = Depends(get_pipeline)):
        return {
            "status": "ok",
            "retrieval": rag.store.mode,
            "documents": len(rag.store.documents()),
            "chunks": len(rag.store.chunks),
            **rag.engine_info(),
        }

    @app.post("/api/settings/api-key")
    def set_api_key(req: ApiKeyRequest, rag: RAGPipeline = Depends(get_pipeline)):
        """Validate the key with Anthropic, then use it immediately and save it to .env.
        The key is never echoed back by any endpoint."""
        try:
            rag.set_api_key(req.api_key)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except anthropic.AuthenticationError:
            raise HTTPException(400, "Anthropic rejected this key. Check that you copied the whole key.")
        except anthropic.PermissionDeniedError:
            raise HTTPException(400, "This key is valid but lacks permission to use the API.")
        except anthropic.APIConnectionError:
            raise HTTPException(502, "Could not reach api.anthropic.com to check the key. Check your network.")
        except anthropic.APIStatusError as exc:
            raise HTTPException(502, f"Anthropic API error while checking the key ({exc.status_code}).")
        return {"api_key_configured": True}

    @app.get("/api/documents")
    def list_documents(rag: RAGPipeline = Depends(get_pipeline)):
        return rag.store.documents()

    @app.post("/api/documents")
    async def upload(files: list[UploadFile] = File(...), rag: RAGPipeline = Depends(get_pipeline)):
        results = []
        for f in files:
            data = await f.read()
            name = Path(f.filename or "upload.txt").name
            if len(data) > MAX_UPLOAD_BYTES:
                results.append({"source": name, "error": "File too large (max 25 MB)"})
                continue
            try:
                results.append(rag.ingest_bytes(name, data))
            except UnsupportedFileType as exc:
                results.append({"source": name, "error": str(exc)})
            except Exception as exc:  # corrupt PDF, etc.
                results.append({"source": name, "error": f"Could not read file: {exc}"})
        return results

    @app.post("/api/documents/text")
    def add_text(doc: TextDoc, rag: RAGPipeline = Depends(get_pipeline)):
        return rag.ingest_text(doc.source, doc.text)

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str, rag: RAGPipeline = Depends(get_pipeline)):
        if not rag.store.delete_document(doc_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": doc_id}

    @app.get("/api/search")
    def search(q: str, k: int = 6, rag: RAGPipeline = Depends(get_pipeline)):
        return [
            {"source": r.chunk.source, "chunk": r.chunk.index + 1, "score": round(r.score, 5), "text": r.chunk.text}
            for r in rag.store.search(q, top_k=max(1, min(k, 50)))
        ]

    @app.post("/api/chat")
    def chat(req: ChatRequest, rag: RAGPipeline = Depends(get_pipeline)):
        history = [t.model_dump() for t in req.history]

        def events():
            for event in rag.ask_stream(req.question, history):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
