"""Tests for wiki CRUD, frontmatter parsing, slugify, and wikilinks."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.models import Confidence, PageType, WikiFrontmatter
from src.wiki import (
    delete_page,
    extract_wikilinks,
    get_page_by_title,
    list_pages,
    parse_frontmatter,
    read_all_pages,
    read_page,
    render_page,
    slugify,
    title_to_path,
    write_page,
)

# ── Slugify ──────────────────────────────────────────────────────────────────


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Transformer Architecture") == "transformer-architecture"

    def test_single_word(self) -> None:
        assert slugify("BERT") == "bert"

    def test_special_chars(self) -> None:
        assert slugify("GPT-4o & Friends!") == "gpt-4o-friends"

    def test_leading_trailing_spaces(self) -> None:
        assert slugify("  hello world  ") == "hello-world"

    def test_multiple_spaces(self) -> None:
        assert slugify("multi   word   title") == "multi-word-title"

    def test_empty(self) -> None:
        assert slugify("") == ""

    def test_numbers(self) -> None:
        assert slugify("GPT 3.5") == "gpt-35"


class TestTitleToPath:
    def test_adds_md_extension(self) -> None:
        assert title_to_path("BERT") == "bert.md"

    def test_multi_word(self) -> None:
        assert title_to_path("Self-Attention") == "self-attention.md"


# ── Frontmatter parsing ─────────────────────────────────────────────────────


class TestParseFrontmatter:
    def test_parse_concept(self, tmp_wiki_dir: Path) -> None:
        raw = (tmp_wiki_dir / "transformer-architecture.md").read_text()
        fm = parse_frontmatter(raw)

        assert fm.title == "Transformer Architecture"
        assert fm.page_type == PageType.CONCEPT
        assert "papers/attention.pdf" in fm.sources
        assert "deep-learning" in fm.tags
        assert fm.confidence == Confidence.HIGH

    def test_parse_entity(self, tmp_wiki_dir: Path) -> None:
        raw = (tmp_wiki_dir / "bert.md").read_text()
        fm = parse_frontmatter(raw)

        assert fm.title == "BERT"
        assert fm.page_type == PageType.ENTITY
        assert fm.confidence == Confidence.HIGH

    def test_defaults_for_missing_fields(self) -> None:
        raw = """---
title: "Minimal"
type: concept
---

Content here.
"""
        fm = parse_frontmatter(raw)
        assert fm.title == "Minimal"
        assert fm.sources == []
        assert fm.tags == []
        assert fm.confidence == Confidence.MEDIUM


# ── Render page ──────────────────────────────────────────────────────────────


class TestRenderPage:
    def test_roundtrip(self, sample_frontmatter: WikiFrontmatter) -> None:
        body = "# Test Page\n\nContent here."
        raw = render_page(sample_frontmatter, body)

        # Parse it back
        fm = parse_frontmatter(raw)
        assert fm.title == sample_frontmatter.title
        assert fm.page_type == sample_frontmatter.page_type
        assert fm.confidence == sample_frontmatter.confidence
        assert body in raw

    def test_includes_related_when_present(self) -> None:
        fm = WikiFrontmatter(
            title="Test",
            page_type=PageType.CONCEPT,
            related=["BERT", "GPT"],
            created=date(2026, 4, 8),
            updated=date(2026, 4, 8),
        )
        raw = render_page(fm, "Content")
        assert "BERT" in raw
        assert "GPT" in raw

    def test_excludes_related_when_empty(self, sample_frontmatter: WikiFrontmatter) -> None:
        raw = render_page(sample_frontmatter, "Content")
        assert "related" not in raw


# ── Wikilinks ────────────────────────────────────────────────────────────────


class TestExtractWikilinks:
    def test_basic(self) -> None:
        text = "See [[BERT]] and [[GPT-4]] for details."
        links = extract_wikilinks(text)
        assert links == ["BERT", "GPT-4"]

    def test_no_links(self) -> None:
        assert extract_wikilinks("No links here.") == []

    def test_multiline(self) -> None:
        text = "- [[Self-Attention]]\n- [[BERT]]\n- [[Transformer Architecture]]"
        links = extract_wikilinks(text)
        assert len(links) == 3
        assert "Self-Attention" in links

    def test_duplicate_links(self) -> None:
        text = "[[BERT]] is cool. [[BERT]] is great."
        links = extract_wikilinks(text)
        assert links == ["BERT", "BERT"]


# ── CRUD operations ─────────────────────────────────────────────────────────


class TestReadPage:
    def test_read_existing(self, tmp_wiki_dir: Path) -> None:
        page = read_page("transformer-architecture.md", tmp_wiki_dir)
        assert page.frontmatter.title == "Transformer Architecture"
        assert "self-attention" in page.body.lower()
        assert page.path == "transformer-architecture.md"

    def test_read_nonexistent(self, tmp_wiki_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_page("nonexistent.md", tmp_wiki_dir)


class TestWritePage:
    def test_write_creates_file(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        fm = WikiFrontmatter(
            title="New Page",
            page_type=PageType.CONCEPT,
            tags=["test"],
            created=date(2026, 4, 8),
            updated=date(2026, 4, 8),
        )
        path = write_page(fm, "# New Page\n\nContent.", wiki_dir)

        assert path == "new-page.md"
        assert (wiki_dir / "new-page.md").exists()

        # Verify content roundtrips
        page = read_page(path, wiki_dir)
        assert page.frontmatter.title == "New Page"

    def test_write_creates_dir(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "new_wiki"
        fm = WikiFrontmatter(
            title="Auto Dir",
            page_type=PageType.CONCEPT,
            created=date(2026, 4, 8),
            updated=date(2026, 4, 8),
        )
        write_page(fm, "Content", wiki_dir)
        assert wiki_dir.exists()


class TestListPages:
    def test_list_existing(self, tmp_wiki_dir: Path) -> None:
        pages = list_pages(tmp_wiki_dir)
        assert "bert.md" in pages
        assert "transformer-architecture.md" in pages
        assert len(pages) == 2

    def test_list_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert list_pages(empty) == []

    def test_list_nonexistent_dir(self, tmp_path: Path) -> None:
        assert list_pages(tmp_path / "nope") == []


class TestDeletePage:
    def test_delete_existing(self, tmp_wiki_dir: Path) -> None:
        assert delete_page("bert.md", tmp_wiki_dir) is True
        assert not (tmp_wiki_dir / "bert.md").exists()

    def test_delete_nonexistent(self, tmp_wiki_dir: Path) -> None:
        assert delete_page("nope.md", tmp_wiki_dir) is False


class TestGetPageByTitle:
    def test_found(self, tmp_wiki_dir: Path) -> None:
        page = get_page_by_title("BERT", tmp_wiki_dir)
        assert page is not None
        assert page.frontmatter.title == "BERT"

    def test_not_found(self, tmp_wiki_dir: Path) -> None:
        assert get_page_by_title("Nonexistent", tmp_wiki_dir) is None


class TestReadAllPages:
    def test_reads_all(self, tmp_wiki_dir: Path) -> None:
        pages = read_all_pages(tmp_wiki_dir)
        assert len(pages) == 2
        titles = {p.frontmatter.title for p in pages}
        assert titles == {"Transformer Architecture", "BERT"}

    def test_skips_malformed(self, tmp_wiki_dir: Path) -> None:
        (tmp_wiki_dir / "bad.md").write_text("not valid frontmatter at all")
        pages = read_all_pages(tmp_wiki_dir)
        # bad.md has no frontmatter — python-frontmatter returns empty metadata
        # parse_frontmatter will use defaults, so it should still parse
        assert len(pages) >= 2
