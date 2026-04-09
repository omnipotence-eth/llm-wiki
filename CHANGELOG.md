# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-04-08

### Added
- `wiki ingest-all` batch command with glob pattern support
- Schema-driven prompts: ingest and query pipelines now load system prompts from `schema.yaml`
- Tag vocabulary from `schema.yaml` passed to LLM during ingest for controlled tagging

### Fixed
- README Ollama model reference now matches config default (`qwen2.5:3b`, not `qwen3:4b`)

## [0.2.0] — 2026-04-08

### Added
- Wiki CRUD: read/write/list/delete markdown pages with YAML frontmatter
- Slugify titles, parse frontmatter via python-frontmatter, extract [[wikilinks]]
- Text extraction: PDF (pymupdf), URL (trafilatura), text/markdown
- Token counting (tiktoken) and paragraph-aware chunking
- BM25 search index over wiki pages (rank-bm25)
- LiteLLM wrapper with instructor structured output and Groq→Gemini→Ollama fallback
- LangGraph ingest pipeline: extract→chunk→generate_pages→write→update_links
- LangGraph query pipeline: search→retrieve→synthesize→conditionally persist
- Wiki lint: orphan detection, broken [[wikilink]] refs, missing fields
- Click CLI: `wiki ingest`, `wiki query`, `wiki lint`, `wiki stats`
- 113 tests passing (no API keys required)

### Fixed
- Pin requests<2.33 via uv override (2.33.x has broken compat module on Python 3.13)

## [0.1.0] — 2026-04-08

### Added
- Project scaffold with pydantic-settings config (WIKI_ env prefix)
- Pydantic schemas: WikiPage, GeneratedPage, IngestResult, QueryAnswer, LintIssue
- Wiki behavior schema (schema.yaml) — page types, tags, prompts
- CI workflow (ruff + pytest), pre-commit hooks, dependabot

[Unreleased]: https://github.com/omnipotence-eth/llm-wiki/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/omnipotence-eth/llm-wiki/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/omnipotence-eth/llm-wiki/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/omnipotence-eth/llm-wiki/releases/tag/v0.1.0
