"""Tests for text extraction and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extract import chunk_text, count_tokens, extract_source, extract_text


class TestCountTokens:
    def test_empty(self) -> None:
        assert count_tokens("") == 0

    def test_basic(self) -> None:
        tokens = count_tokens("Hello, world!")
        assert tokens > 0

    def test_longer_text_more_tokens(self) -> None:
        short = count_tokens("hello")
        long = count_tokens("hello world this is a longer sentence with more tokens")
        assert long > short


class TestExtractText:
    def test_plain_text(self, tmp_path: Path) -> None:
        f = tmp_path / "sample.txt"
        f.write_text("Hello, this is sample text.", encoding="utf-8")

        result = extract_text(f)
        assert result.source_type == "text"
        assert "sample text" in result.content
        assert result.title == "Sample"

    def test_markdown(self, tmp_path: Path) -> None:
        f = tmp_path / "notes.md"
        f.write_text("# Notes\n\nSome markdown content.", encoding="utf-8")

        result = extract_text(f)
        assert result.source_type == "markdown"
        assert "markdown content" in result.content


class TestExtractSource:
    def test_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("Content here.", encoding="utf-8")

        result = extract_source(str(f))
        assert result.source_type == "text"

    def test_markdown_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nBody.", encoding="utf-8")

        result = extract_source(str(f))
        assert result.source_type == "markdown"

    def test_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_source("/tmp/nonexistent_file_12345.txt")

    def test_unsupported_type(self, tmp_path: Path) -> None:
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")

        with pytest.raises(ValueError, match="unsupported"):
            extract_source(str(f))


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        text = "Short text."
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_on_paragraphs(self) -> None:
        paragraphs = ["Paragraph " + str(i) + ". " + "word " * 50 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) > 1

        # Verify all content is preserved
        rejoined = "\n\n".join(chunks)
        for p in paragraphs:
            assert p in rejoined

    def test_empty_text(self) -> None:
        chunks = chunk_text("", max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_respects_max_tokens(self) -> None:
        # Use paragraphs so chunk_text can split on \n\n boundaries
        paragraphs = ["word " * 40 for _ in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, max_tokens=50)
        assert len(chunks) > 1
        # Each paragraph is ~40 tokens, should be one per chunk
        for chunk in chunks:
            assert count_tokens(chunk) <= 60  # generous upper bound
