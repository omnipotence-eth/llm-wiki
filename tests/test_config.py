"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

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
    import src.config
    from src.config import get_settings

    src.config._settings = None  # reset singleton
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    src.config._settings = None  # cleanup
