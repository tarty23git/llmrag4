"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines; existing env vars win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.split(" #", 1)[0].strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


@dataclass
class Settings:
    # Answer engine: auto (Claude if a key is set, else local Ollama if running, else extractive),
    # or force one of: claude | ollama | extractive. Only "claude" needs an API key.
    llm: str = field(default_factory=lambda: _env("RAG_LLM", "auto"))
    ollama_model: str = field(default_factory=lambda: _env("RAG_OLLAMA_MODEL", "llama3.2"))
    ollama_url: str = field(default_factory=lambda: _env("RAG_OLLAMA_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: _env("RAG_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("RAG_EFFORT", "medium"))
    max_tokens: int = field(default_factory=lambda: int(_env("RAG_MAX_TOKENS", "16000")))
    embeddings: str = field(default_factory=lambda: _env("RAG_EMBEDDINGS", "fastembed"))
    embedding_model: str = field(
        default_factory=lambda: _env("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("RAG_DATA_DIR", "data")))
    top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "6")))
    chunk_size: int = field(default_factory=lambda: int(_env("RAG_CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "150")))

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls()


def save_env_var(name: str, value: str, path: Path = Path(".env")) -> None:
    """Set NAME=value in a .env file (created with owner-only permissions), replacing any existing line."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    lines = [ln for ln in lines if not ln.strip().startswith(f"{name}=")]
    lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    os.environ[name] = value
