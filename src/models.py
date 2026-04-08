"""Pydantic schemas for wiki pages, LLM responses, and lint issues."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field

# ── Enums ────────────────────────────────────────────────────────────────────


class PageType(StrEnum):
    CONCEPT = "concept"
    ENTITY = "entity"
    SOURCE_SUMMARY = "source_summary"
    SYNTHESIS = "synthesis"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Wiki page models ────────────────────────────────────────────────────────


class WikiFrontmatter(BaseModel):
    """YAML frontmatter parsed from a wiki page."""

    title: str
    page_type: PageType = Field(alias="type")
    sources: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created: date = Field(default_factory=date.today)
    updated: date = Field(default_factory=date.today)
    confidence: Confidence = Confidence.MEDIUM
    related: list[str] = Field(default_factory=list, description="[[WikiLink]] targets")

    model_config = {"populate_by_name": True}


class WikiPage(BaseModel):
    """A complete wiki page: frontmatter + body content."""

    path: str = Field(description="Relative path within wiki/ dir")
    frontmatter: WikiFrontmatter
    body: str = Field(description="Markdown body after frontmatter")
    raw: str = Field(default="", description="Original raw markdown")


# ── LLM structured output models ────────────────────────────────────────────


class GeneratedPage(BaseModel):
    """LLM output: one wiki page to create."""

    title: str = Field(description="Page title")
    page_type: PageType = Field(description="concept, entity, source_summary, or synthesis")
    tags: list[str] = Field(description="3-7 relevant tags")
    confidence: Confidence = Field(
        default=Confidence.MEDIUM, description="high/medium/low based on source quality"
    )
    body: str = Field(description="Full markdown body content")
    related_titles: list[str] = Field(
        default_factory=list,
        description="Titles of other wiki pages this should link to",
    )


class IngestResult(BaseModel):
    """LLM output: all pages to create from a single source."""

    source_summary: GeneratedPage = Field(description="Summary page for the source itself")
    concept_pages: list[GeneratedPage] = Field(
        description="Key concepts extracted from the source (3-10 pages)"
    )
    entity_pages: list[GeneratedPage] = Field(
        default_factory=list,
        description="Named entities worth their own page (people, orgs, tools)",
    )


class QueryAnswer(BaseModel):
    """LLM output: answer to a wiki query."""

    answer: str = Field(description="Direct answer to the question")
    citations: list[str] = Field(description="Wiki page titles used as sources")
    confidence: Confidence = Field(
        default=Confidence.MEDIUM, description="Confidence in the answer"
    )
    follow_up_queries: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )
    should_persist: bool = Field(
        default=False,
        description="True if this answer reveals a new synthesis worth saving",
    )
    synthesis_page: GeneratedPage | None = Field(
        default=None, description="If should_persist, the page to create"
    )


# ── Lint models ──────────────────────────────────────────────────────────────


class LintIssue(BaseModel):
    """One issue found during wiki lint."""

    page: str = Field(description="Wiki page path")
    issue_type: str = Field(
        description="orphan | contradiction | stale | broken_ref | missing_field"
    )
    severity: str = Field(description="error | warning | info")
    message: str
    suggestion: str = Field(default="")


# ── Extraction models ───────────────────────────────────────────────────────


class ExtractedSource(BaseModel):
    """Result of extracting text from a source file or URL."""

    content: str
    source_type: str = Field(description="pdf | url | text | markdown")
    title: str
    metadata: dict[str, str] = Field(default_factory=dict)
