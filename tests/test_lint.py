"""Tests for wiki lint checks."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.lint import check_broken_refs, check_missing_fields, check_orphans, lint_wiki
from src.models import Confidence, PageType, WikiFrontmatter
from src.wiki import write_page


def _write_test_page(
    wiki_dir: Path,
    title: str,
    body: str,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
) -> str:
    fm = WikiFrontmatter(
        title=title,
        page_type=PageType.CONCEPT,
        tags=tags or [],
        sources=sources or [],
        created=date(2026, 4, 8),
        updated=date(2026, 4, 8),
        confidence=Confidence.HIGH,
    )
    return write_page(fm, body, wiki_dir)


class TestCheckMissingFields:
    def test_no_tags(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(wiki_dir, "No Tags", "Content.", tags=[])

        issues = check_missing_fields(wiki_dir)
        tag_issues = [i for i in issues if "tags" in i.message]
        assert len(tag_issues) == 1

    def test_no_sources(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(wiki_dir, "No Sources", "Content.", tags=["test"])

        issues = check_missing_fields(wiki_dir)
        source_issues = [i for i in issues if "sources" in i.message]
        assert len(source_issues) == 1
        assert source_issues[0].severity == "info"

    def test_complete_page_no_issues(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(wiki_dir, "Complete", "Content.", tags=["test"], sources=["doc.pdf"])

        issues = check_missing_fields(wiki_dir)
        assert len(issues) == 0


class TestCheckBrokenRefs:
    def test_finds_broken_link(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(
            wiki_dir,
            "Page A",
            "See [[Nonexistent Page]] for more.",
            tags=["test"],
        )

        issues = check_broken_refs(wiki_dir)
        assert len(issues) == 1
        assert issues[0].issue_type == "broken_ref"
        assert "Nonexistent Page" in issues[0].message

    def test_valid_link_no_issue(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(wiki_dir, "Page A", "See [[Page B]].", tags=["test"])
        _write_test_page(wiki_dir, "Page B", "Content.", tags=["test"])

        issues = check_broken_refs(wiki_dir)
        assert len(issues) == 0

    def test_no_links_no_issues(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(wiki_dir, "Solo", "No links here.", tags=["test"])

        issues = check_broken_refs(wiki_dir)
        assert len(issues) == 0


class TestCheckOrphans:
    def test_finds_orphan(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(wiki_dir, "Linked", "See [[Other]].", tags=["test"])
        _write_test_page(wiki_dir, "Other", "Content.", tags=["test"])
        _write_test_page(wiki_dir, "Orphan", "Nobody links here.", tags=["test"])

        issues = check_orphans(wiki_dir)
        orphan_pages = [i.page for i in issues]
        assert "orphan.md" in orphan_pages

    def test_no_orphans(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(wiki_dir, "A", "See [[B]].", tags=["test"])
        _write_test_page(wiki_dir, "B", "See [[A]].", tags=["test"])

        issues = check_orphans(wiki_dir)
        assert len(issues) == 0

    def test_single_page_no_orphan(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(wiki_dir, "Solo", "Content.", tags=["test"])

        issues = check_orphans(wiki_dir)
        assert len(issues) == 0


class TestLintWiki:
    def test_integration(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        _write_test_page(wiki_dir, "A", "See [[Missing]].", tags=[])
        _write_test_page(wiki_dir, "B", "Content.", tags=["test"], sources=["doc.pdf"])
        _write_test_page(wiki_dir, "C", "Orphan page.", tags=["test"])

        issues = lint_wiki(wiki_dir)
        issue_types = {i.issue_type for i in issues}
        # Should find: missing tags on A, missing sources on A/C, broken ref on A, orphan C
        assert "missing_field" in issue_types
        assert "broken_ref" in issue_types

    def test_empty_wiki(self, tmp_path: Path) -> None:
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        issues = lint_wiki(wiki_dir)
        assert issues == []
