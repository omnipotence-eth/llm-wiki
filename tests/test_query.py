"""Tests for LangGraph query pipeline with mocked LLM."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.models import Confidence, GeneratedPage, PageType, QueryAnswer, WikiFrontmatter
from src.query import (
    build_query_graph,
    persist_synthesis_node,
    retrieve_pages_node,
    run_query,
    search_wiki_node,
    synthesize_node,
)
from src.wiki import write_page

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def wiki_with_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a wiki directory with 3+ pages (needed for BM25 IDF)."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()

    monkeypatch.setenv("WIKI_WIKI_DIR", str(wiki_dir))
    import src.config

    src.config._settings = None

    today = date(2026, 4, 8)

    pages = [
        (
            "Transformer Architecture",
            PageType.CONCEPT,
            "Self-attention mechanism processes sequences in parallel.",
            ["deep-learning", "attention"],
        ),
        (
            "BERT",
            PageType.ENTITY,
            "Bidirectional encoder representations from transformers for NLP.",
            ["nlp", "pre-training"],
        ),
        (
            "CNN",
            PageType.CONCEPT,
            "Convolutional neural networks for image classification tasks.",
            ["computer-vision"],
        ),
    ]

    for title, ptype, body, tags in pages:
        fm = WikiFrontmatter(
            title=title,
            page_type=ptype,
            tags=tags,
            created=today,
            updated=today,
            confidence=Confidence.HIGH,
        )
        write_page(fm, f"# {title}\n\n{body}", wiki_dir)

    yield wiki_dir

    src.config._settings = None


@pytest.fixture()
def mock_query_answer() -> QueryAnswer:
    return QueryAnswer(
        answer="Transformers use self-attention to weigh input tokens.",
        citations=["Transformer Architecture"],
        confidence=Confidence.HIGH,
        follow_up_queries=["How does multi-head attention work?"],
        should_persist=False,
    )


@pytest.fixture()
def mock_query_answer_persist() -> QueryAnswer:
    return QueryAnswer(
        answer="BERT combines transformer architecture with bidirectional pre-training.",
        citations=["Transformer Architecture", "BERT"],
        confidence=Confidence.HIGH,
        follow_up_queries=[],
        should_persist=True,
        synthesis_page=GeneratedPage(
            title="BERT and Transformers",
            page_type=PageType.SYNTHESIS,
            tags=["synthesis"],
            confidence=Confidence.HIGH,
            body="# BERT and Transformers\n\nSynthesis content.",
        ),
    )


# ── Node tests ───────────────────────────────────────────────────────────────


class TestSearchWikiNode:
    @pytest.mark.asyncio
    async def test_finds_results(self, wiki_with_pages: Path) -> None:
        state = {"question": "attention mechanism transformer"}
        result = await search_wiki_node(state)
        assert "search_results" in result
        assert len(result["search_results"]) > 0

    @pytest.mark.asyncio
    async def test_empty_wiki(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        empty_wiki = tmp_path / "empty_wiki"
        empty_wiki.mkdir()
        monkeypatch.setenv("WIKI_WIKI_DIR", str(empty_wiki))
        import src.config

        src.config._settings = None

        state = {"question": "anything"}
        result = await search_wiki_node(state)
        assert "errors" in result

        src.config._settings = None


class TestRetrievePagesNode:
    @pytest.mark.asyncio
    async def test_builds_context(self, wiki_with_pages: Path) -> None:
        # First search
        search_state = {"question": "transformer attention"}
        search_result = await search_wiki_node(search_state)

        state = {"search_results": search_result["search_results"]}
        result = await retrieve_pages_node(state)
        assert "retrieved_context" in result
        assert len(result["retrieved_context"]) > 0

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        state = {"search_results": []}
        result = await retrieve_pages_node(state)
        assert "errors" in result


class TestSynthesizeNode:
    @pytest.mark.asyncio
    async def test_synthesizes_answer(self, mock_query_answer: QueryAnswer) -> None:
        state = {
            "question": "How does attention work?",
            "retrieved_context": "Some context about attention.",
        }

        with patch("src.query.complete_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_query_answer
            result = await synthesize_node(state)

        assert result["answer"] == mock_query_answer.answer
        assert result["should_persist"] is False

    @pytest.mark.asyncio
    async def test_no_context_returns_error(self) -> None:
        state = {"question": "test", "retrieved_context": ""}
        result = await synthesize_node(state)
        assert "errors" in result


class TestPersistSynthesisNode:
    @pytest.mark.asyncio
    async def test_creates_page(
        self, wiki_with_pages: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = {
            "question": "How do BERT and transformers relate?",
            "answer": "BERT uses transformer encoder architecture.",
            "citations": ["Transformer Architecture", "BERT"],
            "should_persist": True,
        }

        result = await persist_synthesis_node(state)
        assert "persisted_page" in result
        assert (wiki_with_pages / result["persisted_page"]).exists()


# ── Integration ──────────────────────────────────────────────────────────────


class TestRunQuery:
    @pytest.mark.asyncio
    async def test_full_pipeline_no_persist(
        self, wiki_with_pages: Path, mock_query_answer: QueryAnswer
    ) -> None:
        with patch("src.query.complete_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_query_answer
            result = await run_query("How does attention work?")

        assert result.get("errors", []) == []
        assert result["answer"] == mock_query_answer.answer
        assert result.get("persisted_page") is None

    @pytest.mark.asyncio
    async def test_full_pipeline_with_persist(
        self, wiki_with_pages: Path, mock_query_answer_persist: QueryAnswer
    ) -> None:
        with patch("src.query.complete_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_query_answer_persist
            result = await run_query("How do BERT and transformers relate?")

        assert result.get("errors", []) == []
        assert result.get("persisted_page") is not None


class TestBuildQueryGraph:
    def test_graph_compiles(self) -> None:
        graph = build_query_graph()
        app = graph.compile()
        assert app is not None
