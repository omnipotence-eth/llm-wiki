"""Wiki health checks: orphans, broken refs, stale pages, missing fields."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import get_settings
from src.models import LintIssue
from src.wiki import extract_wikilinks, list_pages, read_all_pages, title_to_path

logger = logging.getLogger(__name__)

# Required frontmatter fields for a healthy page
REQUIRED_FIELDS = ("title", "page_type", "sources", "tags", "confidence")


def lint_wiki(wiki_dir: Path | None = None) -> list[LintIssue]:
    """Run all lint checks on the wiki and return issues found."""
    base = wiki_dir or get_settings().wiki_dir
    issues: list[LintIssue] = []

    issues.extend(check_missing_fields(base))
    issues.extend(check_broken_refs(base))
    issues.extend(check_orphans(base))

    logger.info("lint complete issues=%d", len(issues))
    return issues


def check_missing_fields(wiki_dir: Path) -> list[LintIssue]:
    """Check for pages missing required frontmatter fields."""
    issues: list[LintIssue] = []
    pages = read_all_pages(wiki_dir)

    for page in pages:
        fm = page.frontmatter
        if not fm.title:
            issues.append(
                LintIssue(
                    page=page.path,
                    issue_type="missing_field",
                    severity="error",
                    message="missing title",
                    suggestion="Add a title to the frontmatter",
                )
            )
        if not fm.tags:
            issues.append(
                LintIssue(
                    page=page.path,
                    issue_type="missing_field",
                    severity="warning",
                    message="no tags defined",
                    suggestion="Add relevant tags to improve discoverability",
                )
            )
        if not fm.sources:
            issues.append(
                LintIssue(
                    page=page.path,
                    issue_type="missing_field",
                    severity="info",
                    message="no sources linked",
                    suggestion="Add source references for traceability",
                )
            )

    return issues


def check_broken_refs(wiki_dir: Path) -> list[LintIssue]:
    """Check for [[wikilinks]] that point to nonexistent pages."""
    issues: list[LintIssue] = []
    existing = set(list_pages(wiki_dir))
    pages = read_all_pages(wiki_dir)

    for page in pages:
        links = extract_wikilinks(page.body)
        for link in links:
            target_path = title_to_path(link)
            if target_path not in existing:
                issues.append(
                    LintIssue(
                        page=page.path,
                        issue_type="broken_ref",
                        severity="warning",
                        message=f"broken link to [[{link}]] (no file {target_path})",
                        suggestion=f"Create {target_path} or remove the link",
                    )
                )

    return issues


def check_orphans(wiki_dir: Path) -> list[LintIssue]:
    """Find pages that are not linked to by any other page."""
    issues: list[LintIssue] = []
    pages = read_all_pages(wiki_dir)

    if len(pages) < 2:
        return issues

    # Collect all outgoing links
    linked_paths: set[str] = set()
    for page in pages:
        links = extract_wikilinks(page.body)
        for link in links:
            linked_paths.add(title_to_path(link))

        # Also count related field
        for rel in page.frontmatter.related:
            linked_paths.add(rel)

    # Pages not referenced by anyone
    for page in pages:
        if page.path not in linked_paths:
            issues.append(
                LintIssue(
                    page=page.path,
                    issue_type="orphan",
                    severity="info",
                    message="page is not linked from any other page",
                    suggestion="Add a [[wikilink]] from a related page",
                )
            )

    return issues
