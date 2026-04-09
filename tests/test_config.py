"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import src.config
from src.config import Settings


def test_default_settings():
    """Settings load with defaults when no env vars set."""
    s = Settings()
    assert s.wiki_dir == Path("wiki")
    assert s.sources_dir == Path("sources")
    assert s.ingest_temperature == 0.3
    assert s.bm25_k1 == 1.5


def test_settings_from_env(monkeypatch):
    """Settings pick up WIKI_ prefixed env vars."""
    monkeypatch.setenv("WIKI_WIKI_DIR", "/tmp/my-wiki")
    monkeypatch.setenv("WIKI_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("WIKI_MAX_CHUNK_TOKENS", "8000")
    s = Settings()
    assert s.wiki_dir == Path("/tmp/my-wiki")
    assert s.log_level == "DEBUG"
    assert s.max_chunk_tokens == 8000


def test_secret_str_hides_key():
    """API keys use SecretStr to prevent accidental logging."""
    s = Settings(groq_api_key="sk-secret-key")
    assert "sk-secret-key" not in str(s.groq_api_key)
    assert s.groq_api_key.get_secret_value() == "sk-secret-key"


def test_extra_env_ignored(monkeypatch):
    """Unknown WIKI_ vars don't crash settings."""
    monkeypatch.setenv("WIKI_UNKNOWN_SETTING", "ignored")
    s = Settings()
    assert not hasattr(s, "unknown_setting")


def test_settings_singleton():
    """get_settings returns cached instance."""
    from src.config import get_settings

    src.config._settings = None  # reset singleton
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    src.config._settings = None  # cleanup


def test_load_schema_from_file(tmp_path, monkeypatch):
    """load_schema reads prompts and tags from schema.yaml."""
    from src.config import load_schema

    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(
        "prompts:\n"
        "  ingest_system: 'Custom ingest prompt'\n"
        "  query_system: 'Custom query prompt'\n"
        "tags:\n"
        "  allowed:\n"
        "    - ml\n"
        "    - nlp\n"
    )
    src.config._schema_cache = None
    src.config._settings = None
    monkeypatch.setenv("WIKI_SCHEMA_PATH", str(schema_file))
    result = load_schema()
    assert result["prompts"]["ingest_system"] == "Custom ingest prompt"
    assert result["tags"]["allowed"] == ["ml", "nlp"]
    src.config._schema_cache = None
    src.config._settings = None


def test_load_schema_missing_file(tmp_path, monkeypatch):
    """load_schema returns empty dict when file missing."""
    from src.config import load_schema

    src.config._schema_cache = None
    src.config._settings = None
    monkeypatch.setenv("WIKI_SCHEMA_PATH", str(tmp_path / "nonexistent.yaml"))
    result = load_schema()
    assert result == {}
    src.config._schema_cache = None
    src.config._settings = None


def test_get_ingest_prompt_default(tmp_path, monkeypatch):
    """get_ingest_prompt returns fallback when schema has no prompts."""
    from src.config import get_ingest_prompt

    src.config._schema_cache = None
    src.config._settings = None
    monkeypatch.setenv("WIKI_SCHEMA_PATH", str(tmp_path / "nonexistent.yaml"))
    prompt = get_ingest_prompt()
    assert "knowledge wiki curator" in prompt
    src.config._schema_cache = None
    src.config._settings = None


def test_get_allowed_tags_from_schema(tmp_path, monkeypatch):
    """get_allowed_tags returns tags from schema.yaml."""
    from src.config import get_allowed_tags

    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text("tags:\n  allowed:\n    - deep-learning\n    - rag\n")
    src.config._schema_cache = None
    src.config._settings = None
    monkeypatch.setenv("WIKI_SCHEMA_PATH", str(schema_file))
    tags = get_allowed_tags()
    assert tags == ["deep-learning", "rag"]
    src.config._schema_cache = None
    src.config._settings = None
