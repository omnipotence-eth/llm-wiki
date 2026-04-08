"""BM25 search index over wiki pages."""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.models import WikiPage

logger = logging.getLogger(__name__)

# ── Tokenization ─────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return _WORD_RE.findall(text.lower())


# ── Wiki Index ───────────────────────────────────────────────────────────────


class WikiIndex:
    """BM25 search index over wiki pages."""

    def __init__(self, pages: list[WikiPage] | None = None) -> None:
        self._pages: list[WikiPage] = []
        self._bm25: BM25Okapi | None = None
        if pages:
            self.build(pages)

    def build(self, pages: list[WikiPage]) -> None:
        """Build the BM25 index from wiki pages."""
        self._pages = list(pages)
        if not self._pages:
            self._bm25 = None
            return

        settings = get_settings()
        corpus = [self._page_text(p) for p in self._pages]
        tokenized = [tokenize(doc) for doc in corpus]

        self._bm25 = BM25Okapi(tokenized, k1=settings.bm25_k1, b=settings.bm25_b)
        logger.info("built index pages=%d", len(self._pages))

    def search(self, query: str, top_k: int = 5) -> list[WikiPage]:
        """Search the index and return top-k matching pages."""
        if not self._bm25 or not self._pages:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Pair scores with pages, filter zero scores, sort descending
        scored = [
            (score, page) for score, page in zip(scores, self._pages, strict=True) if score > 0
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = [page for _, page in scored[:top_k]]
        logger.info(
            "search query=%r results=%d top_score=%.2f",
            query,
            len(results),
            scored[0][0] if scored else 0.0,
        )
        return results

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @staticmethod
    def _page_text(page: WikiPage) -> str:
        """Combine title, tags, and body for indexing."""
        parts = [
            page.frontmatter.title,
            " ".join(page.frontmatter.tags),
            page.body,
        ]
        return " ".join(parts)
