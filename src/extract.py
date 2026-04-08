"""Text extraction from PDF, URL, and text/markdown sources with chunking."""

from __future__ import annotations

import logging
from pathlib import Path

import tiktoken

from src.config import get_settings
from src.models import ExtractedSource

logger = logging.getLogger(__name__)

# ── Token counting ───────────────────────────────────────────────────────────

_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING  # noqa: PLW0603
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_get_encoding().encode(text))


# ── Extraction ───────────────────────────────────────────────────────────────


def extract_pdf(path: Path) -> ExtractedSource:
    """Extract text from a PDF file using pymupdf."""
    import pymupdf

    doc = pymupdf.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()

    content = "\n\n".join(pages)
    title = path.stem.replace("-", " ").replace("_", " ").title()

    logger.info("extracted pdf path=%s pages=%d tokens=%d", path, len(pages), count_tokens(content))

    return ExtractedSource(
        content=content,
        source_type="pdf",
        title=title,
        metadata={"pages": str(len(pages)), "path": str(path)},
    )


def extract_url(url: str) -> ExtractedSource:
    """Extract text from a URL using trafilatura."""
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        msg = f"failed to download URL: {url}"
        raise ValueError(msg)

    content = trafilatura.extract(downloaded)
    if not content:
        msg = f"no content extracted from URL: {url}"
        raise ValueError(msg)

    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else url

    logger.info("extracted url=%s tokens=%d", url, count_tokens(content))

    return ExtractedSource(
        content=content,
        source_type="url",
        title=title,
        metadata={"url": url},
    )


def extract_text(path: Path) -> ExtractedSource:
    """Extract text from a plain text or markdown file."""
    content = path.read_text(encoding="utf-8")
    source_type = "markdown" if path.suffix.lower() == ".md" else "text"
    title = path.stem.replace("-", " ").replace("_", " ").title()

    logger.info("extracted text path=%s tokens=%d", path, count_tokens(content))

    return ExtractedSource(
        content=content,
        source_type=source_type,
        title=title,
        metadata={"path": str(path)},
    )


def extract_source(path_or_url: str) -> ExtractedSource:
    """Auto-detect source type and extract text."""
    if path_or_url.startswith(("http://", "https://")):
        return extract_url(path_or_url)

    path = Path(path_or_url)
    if not path.exists():
        msg = f"source file not found: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".txt", ".md", ".markdown", ".rst"}:
        return extract_text(path)

    msg = f"unsupported file type: {suffix}"
    raise ValueError(msg)


# ── Chunking ─────────────────────────────────────────────────────────────────


def chunk_text(text: str, max_tokens: int | None = None) -> list[str]:
    """Split text into chunks that fit within max_tokens.

    Splits on paragraph boundaries (double newlines) first, then on
    single newlines if paragraphs are too large.
    """
    if max_tokens is None:
        max_tokens = get_settings().max_chunk_tokens

    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > max_tokens:
            # Flush current chunk
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0

            # Split large paragraph by lines
            lines = para.split("\n")
            line_buf: list[str] = []
            line_tokens = 0
            for line in lines:
                lt = count_tokens(line)
                if line_tokens + lt > max_tokens and line_buf:
                    chunks.append("\n".join(line_buf))
                    line_buf = []
                    line_tokens = 0
                line_buf.append(line)
                line_tokens += lt
            if line_buf:
                chunks.append("\n".join(line_buf))
            continue

        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    logger.info("chunked text total_tokens=%d chunks=%d", count_tokens(text), len(chunks))
    return chunks
