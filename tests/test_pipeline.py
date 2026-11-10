from pathlib import Path

import anthropic
import httpx2

from rag.llm import SYSTEM_PROMPT, ClaudeAnswerer, build_messages
from rag.pipeline import RAGPipeline

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"


def test_ingest_and_ask_streams_sources_then_text(pipeline, answerer):
    results = pipeline.ingest_paths([SAMPLES])
    assert {Path(r["source"]).name for r in results} == {"solar_system.md", "company_handbook.txt"}
    assert all(r["chunks"] > 0 for r in results)

    events = list(pipeline.ask_stream("How many moons does Mars have?"))
    assert events[0]["type"] == "sources"
    assert any("Phobos" in s["text"] for s in events[0]["sources"])
    assert events[-1]["type"] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "text").strip() == answerer.reply

    # The question and numbered sources were sent to the model.
    sent = answerer.calls[0][-1]["content"]
    assert '<source id="1"' in sent and sent.endswith("Question: How many moons does Mars have?")


def test_ask_collects_answer(pipeline):
    pipeline.ingest_text("notes.txt", "The vacation carry-over limit is 5 days.")
    out = pipeline.ask("carry-over limit?")
    assert out["answer"].strip() == "Mars has two moons [1]."
    assert out["sources"][0]["source"] == "notes.txt"
    assert out["stop_reason"] == "end_turn"


def test_followup_retrieval_uses_previous_question(pipeline):
    pipeline.ingest_text("a.txt", "Jupiter's Great Red Spot is a giant storm.")
    pipeline.ingest_text("b.txt", "Bread is baked from flour.")
    history = [{"role": "user", "content": "Tell me about Jupiter"}, {"role": "assistant", "content": "It is big."}]
    results = pipeline.retrieve("what about its storm?", history)
    assert results[0].chunk.source == "a.txt"


def test_build_messages_drops_leading_assistant_and_empty_turns():
    msgs = build_messages(
        "q?", [], [{"role": "assistant", "content": "hi"}, {"role": "user", "content": ""}, {"role": "user", "content": "u1"}]
    )
    assert [m["role"] for m in msgs] == ["user", "user"]
    assert "no matching excerpts" in msgs[-1]["content"]


def test_claude_answerer_request_shape(fake_stream_client):
    a = ClaudeAnswerer("claude-opus-5-5", effort="low", max_tokens=1234, client=fake_stream_client)
    events = list(a.stream([{"role": "user", "content": "hi"}]))
    assert [e["text"] for e in events if e["type"] == "text"] == ["Hello ", "world [1]."]
    assert events[-1]["type"] == "done" and events[-1]["usage"]["input_tokens"] == 10

    kw = fake_stream_client.beta.messages.last_kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["max_tokens"] == 1234
    assert kw["output_config"] == {"effort": "low"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["system"][0]["text"] == SYSTEM_PROMPT
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in kw and "temperature" not in kw


def test_api_errors_become_error_events(settings):
    class Boom:
        has_credentials = True

        def stream(self, messages):
            req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=req), body=None)
            yield  # pragma: no cover

    rag = RAGPipeline(settings, embedder=None, answerer=Boom())
    events = list(rag.ask_stream("hello"))
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "invalid_api_key"
    assert "API key" in events[-1]["message"]


def test_missing_key_short_circuits_before_calling_claude(settings, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    rag = RAGPipeline(settings, embedder=None, answerer=ClaudeAnswerer("claude-opus-5-5"))
    assert not rag.has_api_key
    events = list(rag.ask_stream("hello"))
    assert [e["type"] for e in events] == ["sources", "error"]
    assert events[-1]["code"] == "missing_api_key"


def test_set_api_key_persists_to_dotenv(settings, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("RAG_TOP_K=4\nANTHROPIC_API_KEY=old\n")
    rag = RAGPipeline(settings, embedder=None, answerer=ClaudeAnswerer("claude-opus-5-5"))
    rag.set_api_key("  sk-ant-new-key-456 ", validate=False)
    assert (tmp_path / ".env").read_text() == "RAG_TOP_K=4\nANTHROPIC_API_KEY=sk-ant-new-key-456\n"
    assert oct((tmp_path / ".env").stat().st_mode & 0o777) == "0o600"
    assert rag.has_api_key
    try:
        rag.set_api_key("has space", validate=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
