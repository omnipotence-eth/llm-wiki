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
    ollama_model: str = "ollama/qwen2.5:3b"
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


_schema_cache: dict[str, object] | None = None


def load_schema() -> dict[str, object]:
    """Load and cache schema.yaml (prompts, tags, page types)."""
    global _schema_cache  # noqa: PLW0603
    if _schema_cache is not None:
        return _schema_cache

    import yaml  # noqa: PLC0415

    settings = get_settings()
    path = settings.schema_path
    if not path.exists():
        logger.warning("schema.yaml not found at %s, using defaults", path)
        _schema_cache = {}
        return _schema_cache

    with open(path) as f:
        _schema_cache = yaml.safe_load(f) or {}

    logger.info("loaded schema path=%s", path)
    return _schema_cache


def get_ingest_prompt() -> str:
    """Return the ingest system prompt from schema.yaml."""
    schema = load_schema()
    prompts = schema.get("prompts", {})
    return prompts.get(
        "ingest_system",
        "You are a knowledge wiki curator. Given source text, extract key concepts "
        "and entities into structured wiki pages.",
    )


def get_query_prompt() -> str:
    """Return the query system prompt from schema.yaml."""
    schema = load_schema()
    prompts = schema.get("prompts", {})
    return prompts.get(
        "query_system",
        "You are a knowledge wiki assistant. Answer questions using the provided "
        "wiki pages as context.",
    )


def get_allowed_tags() -> list[str]:
    """Return the controlled tag vocabulary from schema.yaml."""
    schema = load_schema()
    tags = schema.get("tags", {})
    return tags.get("allowed", [])
