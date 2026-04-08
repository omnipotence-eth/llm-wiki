"""Shared test fixtures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.models import Confidence, GeneratedPage, PageType, WikiFrontmatter, WikiPage


@pytest.fixture()
def tmp_wiki_dir(tmp_path: Path) -> Path:
    """Create a temporary wiki directory with sample pages."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()

    # Sample concept page
    (wiki_dir / "transformer-architecture.md").write_text(
        """---
title: "Transformer Architecture"
type: concept
sources: ["papers/attention.pdf"]
tags: ["deep-learning", "attention"]
created: 2026-04-08
updated: 2026-04-08
confidence: high
---

# Transformer Architecture

The Transformer uses self-attention to process sequences in parallel.

## Related
- [[Self-Attention]]
- [[BERT]]
"""
    )

    # Sample entity page
    (wiki_dir / "bert.md").write_text(
        """---
title: "BERT"
type: entity
sources: ["papers/bert.pdf"]
tags: ["nlp", "pre-training"]
created: 2026-04-08
updated: 2026-04-08
confidence: high
---

# BERT

Bidirectional Encoder Representations from Transformers.

## Related
- [[Transformer Architecture]]
"""
    )

    return wiki_dir


@pytest.fixture()
def sample_frontmatter() -> WikiFrontmatter:
    return WikiFrontmatter(
        title="Test Page",
        page_type=PageType.CONCEPT,
        sources=["test.pdf"],
        tags=["test"],
        created=date(2026, 4, 8),
        updated=date(2026, 4, 8),
        confidence=Confidence.HIGH,
    )


@pytest.fixture()
def sample_page(sample_frontmatter: WikiFrontmatter) -> WikiPage:
    return WikiPage(
        path="test-page.md",
        frontmatter=sample_frontmatter,
        body="# Test Page\n\nSome content.\n",
    )


@pytest.fixture()
def sample_generated_page() -> GeneratedPage:
    return GeneratedPage(
        title="Attention Mechanism",
        page_type=PageType.CONCEPT,
        tags=["deep-learning", "attention"],
        confidence=Confidence.HIGH,
        body="# Attention Mechanism\n\nWeighted sum over values.\n",
        related_titles=["Transformer Architecture"],
    )
