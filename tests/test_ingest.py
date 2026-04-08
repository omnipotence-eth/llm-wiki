"""Tests for LangGraph ingest pipeline with mocked LLM."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.ingest import (
    build_ingest_graph,
    chunk_source_node,
    extract_text_node,
    generate_pages_node,
    run_ingest,
    write_pages_node,
)
from src.models import Confidence, GeneratedPage, IngestResult, PageType

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_source(tmp_path: Path) -> Path:
    """Create a sample text source file."""
    f = tmp_path / "sample.txt"
    f.write_text(
        "Transformers use self-attention to process sequences in parallel.\n\n"
        "BERT is a bidirectional encoder model pre-trained on large corpora.\n\n"
        "The attention mechanism computes weighted sums over value vectors.",
        encoding="utf-8",
    )
    return f


@pytest.fixture()
def mock_ingest_result() -> IngestResult:
    """Create a mock LLM ingest result."""
    return IngestResult(
        source_summary=GeneratedPage(
            title="Sample Source Summary",
            page_type=PageType.SOURCE_SUMMARY,
            tags=["deep-learning"],
            confidence=Confidence.HIGH,
            body="# Sample Source Summary\n\nOverview of the source.",
            related_titles=["Transformer Architecture"],
        ),
        concept_pages=[
            GeneratedPage(
                title="Transformer Architecture",
                page_type=PageType.CONCEPT,
                tags=["deep-learning", "attention"],
                confidence=Confidence.HIGH,
                body="# Transformer Architecture\n\nSelf-attention for sequences.",
                related_titles=["BERT"],
            ),
        ],
        entity_pages=[
            GeneratedPage(
                title="BERT",
                page_type=PageType.ENTITY,
                tags=["nlp", "pre-training"],
                confidence=Confidence.HIGH,
                body="# BERT\n\nBidirectional encoder.",
                related_titles=[],
            ),
        ],
    )


# ── Node tests ───────────────────────────────────────────────────────────────


class TestExtractTextNode:
    @pytest.mark.asyncio
    async def test_extracts_text(self, sample_source: Path) -> None:
        state = {"source_path": str(sample_source)}
        result = await extract_text_node(state)
        assert "extracted_text" in result
        assert "self-attention" in result["extracted_text"]

    @pytest.mark.asyncio
    async def test_returns_error_on_missing_file(self) -> None:
        state = {"source_path": "/nonexistent/file.txt"}
        result = await extract_text_node(state)
        assert "errors" in result
        assert len(result["errors"]) > 0


class TestChunkSourceNode:
    @pytest.mark.asyncio
    async def test_chunks_text(self) -> None:
        state = {"extracted_text": "Short text.", "source_path": "test.txt"}
        result = await chunk_source_node(state)
        assert "chunks" in result
        assert len(result["chunks"]) >= 1

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self) -> None:
        state = {"extracted_text": "", "source_path": "test.txt"}
        result = await chunk_source_node(state)
        assert "errors" in result


class TestGeneratePagesNode:
    @pytest.mark.asyncio
    async def test_generates_pages(self, mock_ingest_result: IngestResult) -> None:
        state = {
            "chunks": ["Some source text about transformers and BERT."],
            "source_path": "test.txt",
            "source_title": "Test Source",
        }

        with patch("src.ingest.complete_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_ingest_result
            result = await generate_pages_node(state)

        assert "generated_pages" in result
        assert len(result["generated_pages"]) == 3  # summary + 1 concept + 1 entity

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_error(self) -> None:
        state = {"chunks": [], "source_path": "test.txt"}
        result = await generate_pages_node(state)
        assert "errors" in result


class TestWritePagesNode:
    @pytest.mark.asyncio
    async def test_writes_pages(
        self, tmp_path: Path, mock_ingest_result: IngestResult, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        # Point settings to tmp wiki dir
        monkeypatch.setenv("WIKI_WIKI_DIR", str(wiki_dir))
        import src.config

        src.config._settings = None

        state = {
            "generated_pages": [
                mock_ingest_result.source_summary,
                *mock_ingest_result.concept_pages,
            ],
            "source_path": "test.txt",
        }

        result = await write_pages_node(state)
        assert "written_paths" in result
        assert len(result["written_paths"]) == 2

        # Verify files exist
        for path in result["written_paths"]:
            assert (wiki_dir / path).exists()

        src.config._settings = None


# ── Integration test ─────────────────────────────────────────────────────────


class TestRunIngest:
    @pytest.mark.asyncio
    async def test_full_pipeline(
        self,
        sample_source: Path,
        mock_ingest_result: IngestResult,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        monkeypatch.setenv("WIKI_WIKI_DIR", str(wiki_dir))
        import src.config

        src.config._settings = None

        with patch("src.ingest.complete_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_ingest_result
            result = await run_ingest(str(sample_source))

        assert result.get("errors", []) == []
        assert len(result.get("written_paths", [])) == 3

        # Verify pages on disk
        md_files = list(wiki_dir.glob("*.md"))
        assert len(md_files) == 3

        src.config._settings = None

    @pytest.mark.asyncio
    async def test_pipeline_with_missing_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.config

        src.config._settings = None

        result = await run_ingest("/nonexistent/file.txt")
        assert len(result.get("errors", [])) > 0

        src.config._settings = None


class TestBuildIngestGraph:
    def test_graph_compiles(self) -> None:
        graph = build_ingest_graph()
        app = graph.compile()
        assert app is not None
