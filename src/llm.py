"""LiteLLM wrapper with instructor structured output and Groq→Gemini→Ollama fallback."""

from __future__ import annotations

import logging
from typing import TypeVar

import instructor
import litellm
from pydantic import BaseModel

from src.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


def _get_providers() -> list[tuple[str, str, dict[str, str]]]:
    """Build ordered list of (model, name, extra_kwargs) from settings."""
    settings = get_settings()
    providers: list[tuple[str, str, dict[str, str]]] = []

    # Groq (highest priority — fastest inference)
    if settings.groq_api_key.get_secret_value():
        providers.append(
            (
                settings.groq_model,
                "groq",
                {"api_key": settings.groq_api_key.get_secret_value()},
            )
        )

    # Gemini
    if settings.gemini_api_key.get_secret_value():
        providers.append(
            (
                settings.gemini_model,
                "gemini",
                {"api_key": settings.gemini_api_key.get_secret_value()},
            )
        )

    # Ollama (local fallback — always available)
    providers.append(
        (
            settings.ollama_model,
            "ollama",
            {"api_base": settings.ollama_host},
        )
    )

    return providers


async def complete_structured(  # noqa: UP047
    messages: list[dict[str, str]],
    response_model: type[T],
    temperature: float | None = None,
) -> T:
    """Call LLM with fallback chain and return structured Pydantic output.

    Tries providers in order: Groq → Gemini → Ollama.
    Uses instructor for structured output extraction.
    """
    providers = _get_providers()
    if not providers:
        msg = "no LLM providers configured"
        raise RuntimeError(msg)

    last_error: Exception | None = None

    for model, name, kwargs in providers:
        try:
            client = instructor.from_litellm(litellm.acompletion)
            result = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                **kwargs,
            )
            logger.info("llm complete provider=%s model=%s", name, model)
            return result
        except Exception as exc:
            logger.warning(
                "llm provider failed provider=%s model=%s error=%s",
                name,
                model,
                str(exc),
            )
            last_error = exc
            continue

    msg = f"all LLM providers failed, last error: {last_error}"
    raise RuntimeError(msg)
