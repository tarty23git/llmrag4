import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from rag.config import Settings
from rag.pipeline import RAGPipeline
from rag.store import tokenize


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests (no model download)."""

    name = "test-hash-64"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(64, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeAnswerer:
    """Stands in for ClaudeAnswerer; records the messages it was given."""

    def __init__(self, reply="Mars has two moons [1].", stop_reason="end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def stream(self, messages):
        self.calls.append(messages)
        for word in self.reply.split(" "):
            yield {"type": "text", "text": word + " "}
        yield {"type": "done", "stop_reason": self.stop_reason, "model": "fake", "usage": {}}


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, embeddings="none", chunk_size=300, chunk_overlap=50, top_k=3)


@pytest.fixture
def answerer():
    return FakeAnswerer()


@pytest.fixture
def pipeline(settings, answerer):
    return RAGPipeline(settings, embedder=HashEmbedder(), answerer=answerer)


@pytest.fixture
def fake_stream_client():
    """A fake anthropic client whose beta.messages.stream() mimics the SDK's MessageStream."""

    class _Stream:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.text_stream = iter(["Hello ", "world [1]."])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return SimpleNamespace(
                stop_reason="end_turn",
                model=self.kwargs["model"],
                usage=SimpleNamespace(input_tokens=10, output_tokens=4, cache_read_input_tokens=0),
            )

    class _Messages:
        def __init__(self):
            self.last_kwargs = None

        def stream(self, **kwargs):
            self.last_kwargs = kwargs
            return _Stream(kwargs)

    msgs = _Messages()
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs), messages=msgs)
