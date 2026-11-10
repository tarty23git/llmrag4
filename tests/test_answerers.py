import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rag.answerers import ExtractiveAnswerer, OllamaAnswerer, OllamaError
from rag.config import Settings
from rag.pipeline import RAGPipeline

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.fixture
def fake_ollama():
    """A tiny HTTP server that speaks the parts of Ollama's API the app uses."""
    state = {"models": ["llama3.2:latest"], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                self._json(200, {"models": [{"name": m, "model": m} for m in state["models"]]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(req)
            if req["model"] not in {m.split(":")[0] for m in state["models"]} | set(state["models"]):
                return self._json(404, {"error": f"model '{req['model']}' not found"})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for piece in ["Olympus Mons ", "is about 22 km high [1]."]:
                self.wfile.write(json.dumps({"message": {"role": "assistant", "content": piece}, "done": False}).encode() + b"\n")
            self.wfile.write(json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop",
                                         "prompt_eval_count": 50, "eval_count": 9}).encode() + b"\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def make(tmp_path, **kw):
    s = Settings(data_dir=tmp_path, embeddings="none", top_k=4, ollama_url=kw.pop("ollama_url", "http://127.0.0.1:9"), **kw)
    rag = RAGPipeline(s, embedder=None)
    rag.ingest_paths([SAMPLES])
    return rag


# ---------- extractive ----------

def test_extractive_quotes_relevant_sentences_with_citations(tmp_path, no_key):
    rag = make(tmp_path)
    assert rag.engine_info()["kind"] == "extractive"
    out = rag.ask("How tall is Olympus Mons?")
    assert "22 km" in out["answer"]
    n = next(s["n"] for s in out["sources"] if s["source"].endswith("solar_system.md"))
    assert f"[{n}]" in out["answer"]
    assert "vacation" not in out["answer"].lower()


def test_extractive_handbook_question(tmp_path, no_key):
    out = make(tmp_path).ask("How many unused vacation days can be carried over?")
    assert "5 unused days" in out["answer"]


def test_extractive_no_match(tmp_path, no_key):
    out = make(tmp_path).ask("zebra quantum xylophone")
    assert "couldn't find" in out["answer"]


def test_extractive_handles_no_sources():
    assert "couldn't find" in ExtractiveAnswerer().answer([{"role": "user", "content": "<sources>\n(no matching excerpts found)\n</sources>\n\nQuestion: hi"}])


# ---------- engine selection ----------

def test_auto_prefers_claude_when_key_present(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert make(tmp_path).engine_info()["kind"] == "claude"


def test_auto_uses_ollama_when_running(tmp_path, no_key, fake_ollama):
    rag = make(tmp_path, ollama_url=fake_ollama["url"])
    assert rag.engine_info()["kind"] == "ollama"
    out = rag.ask("How tall is Olympus Mons?")
    assert out["answer"] == "Olympus Mons is about 22 km high [1]."
    assert out["usage"] == {"input_tokens": 50, "output_tokens": 9}
    sent = fake_ollama["requests"][-1]
    assert sent["model"] == "llama3.2" and sent["stream"] is True
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][-1]["content"].endswith("Question: How tall is Olympus Mons?")


def test_auto_skips_ollama_without_the_model(tmp_path, no_key, fake_ollama):
    fake_ollama["models"] = ["mistral:latest"]
    rag = make(tmp_path, ollama_url=fake_ollama["url"])
    info = rag.engine_info()
    assert info["kind"] == "extractive"
    assert "ollama pull llama3.2" in info["ollama_status"]


def test_forced_ollama_reports_problem_when_down(tmp_path, no_key):
    rag = make(tmp_path, llm="ollama")
    events = list(rag.ask_stream("hello"))
    assert events[-1]["code"] == "ollama_unavailable"
    assert "not running" in events[-1]["message"]


def test_forced_extractive_ignores_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert make(tmp_path, llm="extractive").engine_info()["kind"] == "extractive"


def test_invalid_mode(tmp_path):
    with pytest.raises(ValueError):
        RAGPipeline(Settings(data_dir=tmp_path, embeddings="none", llm="gpt"), embedder=None)


def test_ollama_http_error_is_readable(fake_ollama):
    a = OllamaAnswerer("missing-model", fake_ollama["url"])
    with pytest.raises(OllamaError, match="not found"):
        list(a.stream([{"role": "user", "content": "hi"}]))
