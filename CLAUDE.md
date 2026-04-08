# LLM Wiki — Claude Code Instructions

## Project Overview

Git-backed knowledge base implementing Karpathy's LLM Wiki pattern. Three layers: sources (immutable), wiki (LLM-maintained markdown), schema (behavior config). CLI tool with LangGraph pipelines for ingest and query.

## Architecture

```
src/config.py     — pydantic-settings, WIKI_ env prefix
src/models.py     — Pydantic schemas (WikiPage, GeneratedPage, IngestResult, QueryAnswer, LintIssue)
src/llm.py        — LiteLLM wrapper, instructor structured output, Groq→Gemini→Ollama fallback
src/wiki.py       — Wiki CRUD: markdown read/write, frontmatter parse, wikilinks
src/search.py     — BM25 index (rank-bm25)
src/extract.py    — Text from PDF (pymupdf), URL (trafilatura), text/md + chunking
src/ingest.py     — LangGraph: extract→chunk→generate_pages→write→update_index→update_links
src/query.py      — LangGraph: search→retrieve→synthesize→optionally persist
src/lint.py       — Sync checks: orphans, broken refs, stale, missing fields
src/cli.py        — Click CLI entry point
```

## Commands

```bash
make test                    # Unit tests (no API keys needed)
make lint                    # Ruff check + format
wiki ingest path/to/file.pdf # Ingest source
wiki query "question"        # Query wiki
wiki lint                    # Health check
wiki stats                   # Statistics
```

## Code Standards

- `from __future__ import annotations` in every module
- `logging.getLogger(__name__)` — never `print()`
- Async for LLM calls, sync for filesystem/BM25
- All LLM responses use instructor + Pydantic models
- Mock all LLM calls in tests
- Ruff for lint/format, line length 100

## Key Patterns

- **Frontmatter**: YAML in markdown via `python-frontmatter`. Type alias via `type` field.
- **WikiLinks**: Obsidian `[[double bracket]]` syntax. Extract with regex `\[\[(.+?)\]\]`.
- **Slugify**: Title → lowercase, spaces to hyphens, strip non-alphanum. `"BERT" → "bert.md"`
- **Fallback chain**: Try providers in order, catch exceptions, try next. Log which provider succeeded.
