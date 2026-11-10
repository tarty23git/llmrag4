# llmrag4

A retrieval-augmented generation (RAG) app: index your own documents, ask questions, and get answers grounded in those documents, with numbered citations you can click to see the source passage.

**No API key required.** The app picks its answer engine automatically (`RAG_LLM=auto`):

| Engine | Needs | Answers |
|---|---|---|
| **Ollama** (local LLM) | [Ollama](https://ollama.com) installed and a model pulled (`ollama pull llama3.2`). Free, runs on your machine. | Written answers with citations |
| **Extractive** (fallback) | Nothing | Quotes the most relevant sentences from your documents, with citations |
| **Claude** (optional) | An Anthropic API key | Highest-quality written answers |

In `auto` mode the app uses Claude if a key is set, otherwise Ollama if it's running with the model installed, otherwise the extractive fallback. It re-checks for Ollama while running, so starting Ollama later switches over without a restart. Run `python -m rag.cli engine` to see which engine is active and why.

### Free written answers with Ollama

1. Install Ollama from https://ollama.com/download and start it.
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Start (or keep running) this app. The status line in the web app changes to `ollama:llama3.2`.

Any Ollama chat model works: set `RAG_OLLAMA_MODEL` (for example `qwen2.5:7b` or `mistral`). Larger models give better answers but need more RAM.

- **Hybrid retrieval, fully local.** BM25 keyword search plus on-device sentence embeddings ([fastembed](https://github.com/qdrant/fastembed), `BAAI/bge-small-en-v1.5`, ONNX, no PyTorch), fused with reciprocal rank fusion. No API key needed.
- **Grounded answers with citations.** Retrieved chunks are sent to the model as numbered `<source>` blocks, and the system prompt asks for `[n]` citations and an explicit "not in the documents" when the answer isn't there.
- **Streaming** answers over Server-Sent Events and multi-turn follow-ups. With Claude, it also uses prompt caching on the system prompt and the server-side refusal fallback.
- **Web UI** (drag-and-drop upload, document list, chat) plus a **CLI**.
- Supports PDF, TXT, Markdown, HTML, CSV and JSON.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Optional Claude API key:** you only need this to use Claude as the answer engine. You can add it in any of three ways. Each one is saved to `.env`, which git ignores.
- In the web app: click **API key** (or the banner shown when no key is set) and paste it. The app checks it with Anthropic before saving and starts using it right away, with no restart.
- From the CLI: run `python -m rag.cli set-key` (input is hidden).
- By hand: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

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
| `engine` | Show which answer engine is active and why. |
| `set-key` | Optional: enter, verify and save an Anthropic API key. |
| `serve [--host --port --reload]` | Start the web server. |

## HTTP API

| Method & path | Description |
|---|---|
| `GET /api/health` | Active answer engine, Ollama status, retrieval mode, document and chunk counts, and whether a Claude key is configured. |
| `POST /api/settings/api-key` | `{"api_key": "..."}`: verifies the key with Anthropic, then uses and saves it. The key is never returned by any endpoint. The server listens on 127.0.0.1 by default; if you expose it on a network, anyone who can reach it can change the key. |
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
| `RAG_LLM` | `auto` | Answer engine: `auto`, `ollama`, `extractive` or `claude`. Only `claude` needs a key. |
| `RAG_OLLAMA_MODEL` | `llama3.2` | Ollama model to use. |
| `RAG_OLLAMA_URL` | `http://localhost:11434` | Where Ollama is listening. |
| `ANTHROPIC_API_KEY` | — | Optional. Enables Claude. |
| `RAG_MODEL` | `claude-opus-5-5` | Claude model (when Claude is the engine). |
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
              answer engine: Claude | Ollama | extractive (sources + question) ◄──┘
                                         │
                                         ▼
                           streamed answer with [n] citations
```

- `rag/loaders.py`: text extraction · `rag/chunking.py`: chunking · `rag/embeddings.py`: local embedder
- `rag/store.py`: persistent index and hybrid search · `rag/llm.py`: prompt and Claude streaming call
- `rag/answerers.py`: the no-key engines (Ollama and extractive)
- `rag/pipeline.py`: ties it together · `app/server.py` + `app/static/`: web app · `rag/cli.py`: CLI

## Tests

```bash
pip install pytest httpx
python -m pytest
```

The tests use a deterministic hashing embedder, a fake Claude client and a stand-in Ollama server, so they need no model downloads, no Ollama install and no API key.
