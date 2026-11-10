from rag.chunking import chunk_text


def test_empty_text():
    assert chunk_text("   \n\n ") == []


def test_short_text_single_chunk():
    assert chunk_text("Hello world.", 100, 10) == ["Hello world."]


def test_chunks_respect_size_and_cover_text():
    paras = [f"Paragraph {i}. " + "word " * 40 for i in range(20)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, chunk_size=400, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    for i in range(20):
        assert any(f"Paragraph {i}." in c for c in chunks)


def test_overlap_carries_context():
    text = "\n\n".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_text(text, chunk_size=200, overlap=60)
    # The tail of each chunk should reappear at the start of the next one.
    for a, b in zip(chunks, chunks[1:]):
        assert a[-20:].split()[-1] in b


def test_giant_unbroken_paragraph_is_split():
    chunks = chunk_text("x" * 2500, chunk_size=1000, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)
