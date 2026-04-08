# Contributing

## Branch Strategy
- `feature/` — new features
- `fix/` — bug fixes
- `chore/` — maintenance

## Ship Workflow

1. `ruff check . --fix && ruff format .`
2. `pytest tests/ -v --tb=short`
3. Audit: type hints, no bare except, no hardcoded secrets
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Conventional commit: `feat:`, `fix:`, `docs:`, `chore:`

## PR Checklist

- [ ] Tests pass (`make test`)
- [ ] Linter clean (`make lint`)
- [ ] CHANGELOG updated
- [ ] No hardcoded API keys
