"""Configuration via pydantic-settings with WIKI_ env prefix."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Wiki configuration loaded from environment variables."""

    # Paths
    wiki_dir: Path = Path("wiki")
    sources_dir: Path = Path("sources")
    schema_path: Path = Path("schema.yaml")

    # LLM providers (LiteLLM model strings)
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "ollama/qwen3:4b"
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "groq/meta-llama/llama-4-scout-17b-16e-instruct"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini/gemini-2.5-flash"

    # Generation
    ingest_temperature: float = 0.3
    query_temperature: float = 0.5
    max_chunk_tokens: int = 4000
    max_pages_per_ingest: int = 15

    # Search
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    model_config = {"env_prefix": "WIKI_", "env_file": ".env", "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings singleton."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings
