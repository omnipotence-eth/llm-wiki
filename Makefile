# ==============================================================================
# LLM Wiki — Makefile
# ==============================================================================

.DEFAULT_GOAL := help

.PHONY: help test lint ingest query lint-wiki stats clean

help:
	@echo ""
	@echo "LLM Wiki — available targets"
	@echo "─────────────────────────────────"
	@echo "  test        Run unit tests (no API keys needed)"
	@echo "  lint        Ruff check + format"
	@echo "  ingest      Ingest a source:  make ingest SRC=path/to/file.pdf"
	@echo "  query       Query the wiki:   make query Q='your question'"
	@echo "  lint-wiki   Run wiki health checks"
	@echo "  stats       Show wiki statistics"
	@echo "  clean       Remove caches"
	@echo ""

# ── Quality ─────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check src/ tests/ --fix && uv run ruff format src/ tests/

# ── Wiki Operations ────────────────────────────────────────────────────────
ingest:
	uv run wiki ingest $(SRC)

query:
	uv run wiki query "$(Q)"

lint-wiki:
	uv run wiki lint

stats:
	uv run wiki stats

# ── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	rm -rf __pycache__ tests/__pycache__ src/__pycache__ .pytest_cache
