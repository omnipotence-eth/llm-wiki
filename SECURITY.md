# Security Policy

## Reporting Vulnerabilities

Report security vulnerabilities via [GitHub private advisory](https://github.com/omnipotence-eth/llm-wiki/security/advisories/new).

Do not open public issues for security vulnerabilities.

## Security Model

- API keys loaded from environment variables only (never committed)
- Source documents and wiki content are gitignored by default
- LLM outputs are not executed — markdown only
