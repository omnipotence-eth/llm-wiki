"""Tests for Pydantic models."""

from __future__ import annotations

from datetime import date

import pytest

from src.models import (
    Confidence,
    ExtractedSource,
    GeneratedPage,
    IngestResult,
    LintIssue,
    PageType,
    QueryAnswer,
    WikiFrontmatter,
)


class TestPageType:
    def test_values(self):
        assert PageType.CONCEPT == "concept"
        assert PageType.ENTITY == "entity"
        assert PageType.SOURCE_SUMMARY == "source_summary"
        assert PageType.SYNTHESIS == "synthesis"

    def test_all_types_defined(self):
        assert len(PageType) == 4


class TestWikiFrontmatter:
    def test_create_with_alias(self):
        """PageType field uses 'type' alias for YAML compatibility."""
        fm = WikiFrontmatter(title="Test", type="concept")
        assert fm.page_type == PageType.CONCEPT

    def test_create_with_field_name(self):
        fm = WikiFrontmatter(title="Test", page_type=PageType.ENTITY)
        assert fm.page_type == PageType.ENTITY

    def test_defaults(self):
        fm = WikiFrontmatter(title="Test", type="concept")
        assert fm.sources == []
        assert fm.tags == []
        assert fm.confidence == Confidence.MEDIUM
        assert fm.created == date.today()

    def test_invalid_page_type(self):
        with pytest.raises(ValueError):
            WikiFrontmatter(title="Test", type="invalid")


class TestWikiPage:
    def test_create(self, sample_page):
        assert sample_page.path == "test-page.md"
        assert sample_page.frontmatter.title == "Test Page"
        assert "content" in sample_page.body


class TestGeneratedPage:
    def test_create(self, sample_generated_page):
        assert sample_generated_page.title == "Attention Mechanism"
        assert sample_generated_page.page_type == PageType.CONCEPT
        assert len(sample_generated_page.tags) == 2

    def test_defaults(self):
        gp = GeneratedPage(
            title="Min",
            page_type=PageType.CONCEPT,
            tags=["t"],
            body="content",
        )
        assert gp.confidence == Confidence.MEDIUM
        assert gp.related_titles == []


class TestIngestResult:
    def test_create(self, sample_generated_page):
        result = IngestResult(
            source_summary=sample_generated_page,
            concept_pages=[sample_generated_page],
        )
        assert result.source_summary.title == "Attention Mechanism"
        assert len(result.concept_pages) == 1
        assert result.entity_pages == []


class TestQueryAnswer:
    def test_no_persist(self):
        qa = QueryAnswer(
            answer="Attention is...",
            citations=["Transformer Architecture"],
        )
        assert not qa.should_persist
        assert qa.synthesis_page is None
        assert qa.follow_up_queries == []

    def test_with_persist(self, sample_generated_page):
        qa = QueryAnswer(
            answer="Cross-attention combines...",
            citations=["Transformer Architecture", "BERT"],
            should_persist=True,
            synthesis_page=sample_generated_page,
        )
        assert qa.should_persist
        assert qa.synthesis_page is not None


class TestLintIssue:
    def test_create(self):
        issue = LintIssue(
            page="orphan.md",
            issue_type="orphan",
            severity="warning",
            message="No inbound links",
        )
        assert issue.suggestion == ""

    def test_with_suggestion(self):
        issue = LintIssue(
            page="stale.md",
            issue_type="stale",
            severity="info",
            message="Source modified after page update",
            suggestion="Re-ingest the source",
        )
        assert issue.suggestion == "Re-ingest the source"


class TestExtractedSource:
    def test_create(self):
        es = ExtractedSource(
            content="Hello world",
            source_type="text",
            title="test.txt",
        )
        assert es.metadata == {}
