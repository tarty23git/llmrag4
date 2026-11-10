import json

import pytest
from fastapi.testclient import TestClient

from app import server


@pytest.fixture
def client(pipeline, monkeypatch):
    app = server.create_app()
    app.dependency_overrides[server.get_pipeline] = lambda: pipeline
    return TestClient(app)


def test_health_and_index_page(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["documents"] == 0
    assert "llmrag4" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_upload_list_search_delete(client):
    files = [
        ("files", ("planets.md", b"Mars has two moons, Phobos and Deimos.", "text/markdown")),
        ("files", ("bad.exe", b"MZ", "application/octet-stream")),
    ]
    res = client.post("/api/documents", files=files).json()
    assert res[0]["chunks"] == 1 and "error" in res[1]

    docs = client.get("/api/documents").json()
    assert [d["source"] for d in docs] == ["planets.md"]

    hits = client.get("/api/search", params={"q": "Phobos"}).json()
    assert hits[0]["source"] == "planets.md"

    assert client.delete(f"/api/documents/{docs[0]['doc_id']}").status_code == 200
    assert client.delete(f"/api/documents/{docs[0]['doc_id']}").status_code == 404
    assert client.get("/api/documents").json() == []


def test_chat_streams_sse_events(client):
    client.post("/api/documents/text", json={"source": "mars.txt", "text": "Mars has two moons."})
    with client.stream("POST", "/api/chat", json={"question": "Moons of Mars?", "history": []}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [json.loads(line[6:]) for line in body.split("\n\n") if line.startswith("data: ")]
    assert events[0]["type"] == "sources" and events[0]["sources"][0]["source"] == "mars.txt"
    assert events[-1]["type"] == "done"
    assert "two moons" in "".join(e["text"] for e in events if e["type"] == "text")


def test_chat_validation(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
