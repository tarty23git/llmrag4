"""Turn files on disk (or uploaded bytes) into plain text."""

from __future__ import annotations

import html
import io
import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".pdf", ".html", ".htm", ".csv", ".json"}


class UnsupportedFileType(ValueError):
    pass


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h[1-6]>|</li>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return html.unescape(raw)


def _pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {number}]\n{text}")
    return "\n\n".join(pages)


def extract_text(filename: str, data: bytes) -> str:
    """Extract text from a file's bytes, choosing the parser by extension."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{ext or filename}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if ext == ".pdf":
        return _pdf_to_text(data)
    text = data.decode("utf-8", errors="replace")
    if ext in {".html", ".htm"}:
        text = _html_to_text(text)
    return text


def iter_files(paths: list[str | Path]) -> list[Path]:
    """Expand files and directories (recursively) into supported files."""
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(
                f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(p)
    return found
