# llmrag4

A retrieval-augmented generation (RAG) app: index your own documents, ask questions, and get answers from Claude that are grounded in those documents, with numbered citations you can click to see the source passage.

- **Hybrid retrieval, fully local.** BM25 keyword search plus on-device sentence embeddings ([fastembed](https://github.com/qdrant/fastembed), `BAAI/bge-small-en-v1.5`, ONNX, no PyTorch), fused with reciprocal rank fusion. The only API key you need is Anthropic's.
- **Grounded answers with citations.** Retrieved chunks are sent to Claude as numbered `<source>` blocks, and the system prompt asks for `[n]` citations and an explicit "not in the documents" when the answer isn't there.
- **Streaming** answers over Server-Sent Events, multi-turn follow-ups, prompt caching on the system prompt, and the server-side refusal fallback.
- **Web UI** (drag-and-drop upload, document list, chat) plus a **CLI**.
- Supports PDF, TXT, Markdown, HTML, CSV and JSON.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then set ANTHROPIC_API_KEY in .env
```

Index the sample documents and ask a question:

```bash
python -m rag.cli ingest sample_docs
python -m rag.cli ask "How many vacation days can I carry over?"
```

Run the web app at http://127.0.0.1:8000:

```bash
python -m rag.cli serve
```

(Install with `pip install -e .` to get a `rag` command instead of `python -m rag.cli`.)

## CLI

| Command | What it does |
|---|---|
| `ingest PATH...` | Index files or directories (recursive). Re-ingesting a changed file replaces it; an unchanged one is skipped. |
| `ask "QUESTION"` | Retrieve and answer once, streaming the answer (`--no-sources` hides the source list). |
| `chat` | Interactive multi-turn chat. |
| `search "QUERY" [-k N]` | Show the retrieved chunks without calling Claude, for checking retrieval quality. |
| `list` / `delete DOC_ID` / `clear` | Manage the index. |
| `serve [--host --port --reload]` | Start the web server. |

## HTTP API

| Method & path | Description |
|---|---|
| `GET /api/health` | Model, retrieval mode, document and chunk counts. |
| `GET /api/documents` | List indexed documents. |
| `POST /api/documents` | Upload files (multipart field `files`, repeatable). |
| `POST /api/documents/text` | Index raw text: `{"source": "...", "text": "..."}`. |
| `DELETE /api/documents/{doc_id}` | Remove a document. |
| `GET /api/search?q=...&k=6` | Retrieval only. |
| `POST /api/chat` | `{"question": "...", "history": [{"role": "user", "content": "..."}, ...]}`. Streams SSE events: `sources`, then `text` deltas, then `done` (or `error`). |

## Configuration

Set these in the environment or in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required to generate answers). |
| `RAG_MODEL` | `claude-opus-5-5` | Claude model. |
| `RAG_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max`. Lower is faster and cheaper. |
| `RAG_MAX_TOKENS` | `16000` | Answer token cap. |
| `RAG_EMBEDDINGS` | `fastembed` | `fastembed` for hybrid search, `none` for BM25 keyword search only. |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed text model. Changing it re-embeds the index on the next start. |
| `RAG_DATA_DIR` | `data` | Where the index (`data/index`) and the downloaded embedding model (`data/models`) are stored. |
| `RAG_TOP_K` | `6` | Chunks retrieved per question. |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `150` | Chunk size and overlap, in characters. |

**Embedding model download:** the first run downloads the ~130 MB embedding model from Hugging Face into `data/models`. After that it runs offline. If the download fails (for example on a network that blocks Hugging Face), the app logs a warning after fastembed's retries and falls back to BM25-only search. Set `RAG_EMBEDDINGS=none` to skip the attempt entirely.

## How it works

```
files ──► loaders (pdf/html/text) ──► chunker (paragraph/sentence-aware, overlapping)
                                         │
                                         ▼
                     DocumentStore: chunks.jsonl + vectors.npy + BM25 index
                                         │
question (+ previous user turn) ──► BM25 top-N ─┐
                                 └► vector top-N ┴► reciprocal rank fusion ──► top-k chunks
                                                                                  │
                         Claude (system prompt cached; sources + question) ◄──────┘
                                         │
                                         ▼
                           streamed answer with [n] citations
```

- `rag/loaders.py`: text extraction · `rag/chunking.py`: chunking · `rag/embeddings.py`: local embedder
- `rag/store.py`: persistent index and hybrid search · `rag/llm.py`: prompt and Claude streaming call
- `rag/pipeline.py`: ties it together · `app/server.py` + `app/static/`: web app · `rag/cli.py`: CLI

## Tests

```bash
pip install pytest httpx
python -m pytest
```

The tests use a deterministic hashing embedder and a fake Claude client, so they need neither the embedding model nor an API key.
