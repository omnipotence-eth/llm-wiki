"""Wiki CRUD: read/write/list/delete markdown pages with frontmatter and wikilinks."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import frontmatter

from src.config import get_settings
from src.models import Confidence, PageType, WikiFrontmatter, WikiPage

logger = logging.getLogger(__name__)

# ── Slugify ──────────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9-]")
_MULTI_DASH_RE = re.compile(r"-{2,}")

# ── Wikilink regex ───────────────────────────────────────────────────────────

WIKILINK_RE = re.compile(r"\[\[(.+?)\]\]")


def slugify(title: str) -> str:
    """Convert title to filename slug: lowercase, hyphens, strip non-alphanum.

    >>> slugify("Transformer Architecture")
    'transformer-architecture'
    >>> slugify("BERT")
    'bert'
    """
    slug = title.lower().strip().replace(" ", "-")
    slug = _SLUG_RE.sub("", slug)
    slug = _MULTI_DASH_RE.sub("-", slug)
    return slug.strip("-")


def title_to_path(title: str) -> str:
    """Convert a page title to a relative wiki path."""
    return f"{slugify(title)}.md"


# ── Frontmatter parsing ─────────────────────────────────────────────────────


def parse_frontmatter(raw: str) -> WikiFrontmatter:
    """Parse YAML frontmatter from raw markdown into WikiFrontmatter model."""
    post = frontmatter.loads(raw)
    meta = dict(post.metadata)

    from datetime import date as _date

    today = _date.today()

    return WikiFrontmatter(
        title=meta.get("title", ""),
        page_type=PageType(meta.get("type", "concept")),
        sources=meta.get("sources", []),
        tags=meta.get("tags", []),
        created=meta.get("created") or today,
        updated=meta.get("updated") or today,
        confidence=Confidence(meta.get("confidence", "medium")),
        related=meta.get("related", []),
    )


def render_page(page: WikiFrontmatter, body: str) -> str:
    """Render frontmatter + body into raw markdown string."""
    meta = {
        "title": page.title,
        "type": page.page_type.value,
        "sources": page.sources,
        "tags": page.tags,
        "created": page.created.isoformat(),
        "updated": page.updated.isoformat(),
        "confidence": page.confidence.value,
    }
    if page.related:
        meta["related"] = page.related

    post = frontmatter.Post(body, **meta)
    return frontmatter.dumps(post) + "\n"


# ── Wikilinks ────────────────────────────────────────────────────────────────


def extract_wikilinks(text: str) -> list[str]:
    """Extract all [[WikiLink]] targets from text."""
    return WIKILINK_RE.findall(text)


# ── CRUD operations ─────────────────────────────────────────────────────────


def _wiki_dir() -> Path:
    return get_settings().wiki_dir


def read_page(path: str, wiki_dir: Path | None = None) -> WikiPage:
    """Read a wiki page from disk.

    Args:
        path: Relative filename within wiki dir (e.g. 'bert.md')
        wiki_dir: Override wiki directory (for testing)
    """
    base = wiki_dir or _wiki_dir()
    filepath = base / path
    raw = filepath.read_text(encoding="utf-8")
    fm = parse_frontmatter(raw)
    post = frontmatter.loads(raw)

    return WikiPage(
        path=path,
        frontmatter=fm,
        body=post.content,
        raw=raw,
    )


def write_page(
    page: WikiFrontmatter,
    body: str,
    wiki_dir: Path | None = None,
) -> str:
    """Write a wiki page to disk. Returns the relative path."""
    base = wiki_dir or _wiki_dir()
    base.mkdir(parents=True, exist_ok=True)

    path = title_to_path(page.title)
    filepath = base / path
    raw = render_page(page, body)
    filepath.write_text(raw, encoding="utf-8")

    logger.info("wrote page path=%s title=%s", path, page.title)
    return path


def list_pages(wiki_dir: Path | None = None) -> list[str]:
    """List all .md file paths in the wiki directory."""
    base = wiki_dir or _wiki_dir()
    if not base.exists():
        return []
    return sorted(p.name for p in base.glob("*.md"))


def delete_page(path: str, wiki_dir: Path | None = None) -> bool:
    """Delete a wiki page. Returns True if deleted, False if not found."""
    base = wiki_dir or _wiki_dir()
    filepath = base / path
    if filepath.exists():
        filepath.unlink()
        logger.info("deleted page path=%s", path)
        return True
    return False


def get_page_by_title(title: str, wiki_dir: Path | None = None) -> WikiPage | None:
    """Look up a wiki page by title (via slug)."""
    path = title_to_path(title)
    base = wiki_dir or _wiki_dir()
    filepath = base / path
    if not filepath.exists():
        return None
    return read_page(path, wiki_dir)


def read_all_pages(wiki_dir: Path | None = None) -> list[WikiPage]:
    """Read all wiki pages from disk."""
    paths = list_pages(wiki_dir)
    pages = []
    for path in paths:
        try:
            pages.append(read_page(path, wiki_dir))
        except Exception:
            logger.warning("failed to parse page path=%s", path, exc_info=True)
    return pages
