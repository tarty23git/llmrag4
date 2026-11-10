from rag.store import DocumentStore
from tests.conftest import HashEmbedder


def make_store(tmp_path, embedder=None):
    return DocumentStore(tmp_path / "index", embedder)


def test_bm25_search_finds_relevant_chunk(tmp_path):
    s = make_store(tmp_path)
    s.add_document("a.txt", "x", ["Cats purr and chase mice.", "Rockets need fuel to reach orbit."])
    s.add_document("b.txt", "y", ["Bread is baked from flour and water."])
    top = s.search("how do rockets reach orbit", top_k=2)
    assert top[0].chunk.text.startswith("Rockets")
    assert top[0].bm25_rank == 1 and top[0].vector_rank is None


def test_hybrid_search_uses_both_rankers(tmp_path):
    s = make_store(tmp_path, HashEmbedder())
    s.add_document("a.txt", "x", ["Cats purr and chase mice.", "Rockets need fuel to reach orbit."])
    top = s.search("rockets orbit", top_k=1)[0]
    assert top.chunk.text.startswith("Rockets")
    assert top.bm25_rank == 1 and top.vector_rank == 1


def test_persistence_roundtrip(tmp_path):
    s = make_store(tmp_path, HashEmbedder())
    doc_id, n = s.add_document("a.txt", "x", ["alpha beta", "gamma delta"])
    assert n == 2
    s2 = make_store(tmp_path, HashEmbedder())
    assert len(s2.chunks) == 2 and s2.vectors.shape[0] == 2
    assert s2.search("gamma")[0].chunk.text == "gamma delta"


def test_reingest_same_source_replaces(tmp_path):
    s = make_store(tmp_path, HashEmbedder())
    s.add_document("a.txt", "v1", ["old content"])
    _, n_same = s.add_document("a.txt", "v1", ["old content"])
    assert n_same == 0  # unchanged doc skipped
    s.add_document("a.txt", "v2", ["new content", "more new"])
    assert [c.text for c in s.chunks] == ["new content", "more new"]
    assert s.vectors.shape[0] == 2


def test_delete_and_clear(tmp_path):
    s = make_store(tmp_path, HashEmbedder())
    a, _ = s.add_document("a.txt", "x", ["one", "two"])
    s.add_document("b.txt", "y", ["three"])
    assert s.delete_document(a) == 2
    assert [d["source"] for d in s.documents()] == ["b.txt"]
    assert s.vectors.shape[0] == 1
    s.clear()
    assert s.documents() == [] and s.search("three") == []
    assert make_store(tmp_path, HashEmbedder()).chunks == []


def test_embedder_change_triggers_reembed(tmp_path):
    make_store(tmp_path, HashEmbedder()).add_document("a.txt", "x", ["alpha"])

    class Other(HashEmbedder):
        name = "other-model"

    s = make_store(tmp_path, Other())
    assert s.vectors.shape == (1, 64)
