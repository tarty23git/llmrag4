"""Split text into overlapping chunks that try to respect paragraph/sentence boundaries."""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long(piece: str, size: int) -> list[str]:
    """Break a paragraph longer than `size` into sentence groups, then hard-wrap if needed."""
    out: list[str] = []
    buf = ""
    for sentence in _SENTENCE_END.split(piece):
        while len(sentence) > size:  # a single enormous "sentence"
            if buf:
                out.append(buf)
                buf = ""
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if buf and len(buf) + 1 + len(sentence) > size:
            out.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Return chunks of roughly `chunk_size` characters with ~`overlap` characters of carry-over."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))
    text = _normalize(text)
    if not text:
        return []

    units: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            units.extend(_split_long(para, chunk_size) if len(para) > chunk_size else [para])

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > chunk_size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if tail:
                space = tail.find(" ")
                tail = tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail
            current = f"{tail}\n\n{unit}".strip() if tail else unit
            if len(current) > chunk_size:  # overlap pushed us over; drop it
                current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks
