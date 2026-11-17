"""
src/core/llm_client.py

Unified LLM factory.
Returns a LangChain BaseChatModel based on the configured provider.
Keeps all provider-specific wiring in one place.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from src.core.logging_config import get_logger
from src.core.settings import AppSettings, get_settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)


def _build_google(settings: AppSettings) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.llm.model_name,
        temperature=settings.llm.temperature,
        max_output_tokens=settings.llm.max_tokens,
        google_api_key=settings.google_api_key or None,
    )


def _build_openai(settings: AppSettings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm.model_name,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        api_key=settings.openai_api_key or None,
    )


def _build_anthropic(settings: AppSettings) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.llm.model_name,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        api_key=settings.anthropic_api_key or None,
    )


_BUILDERS = {
    "google": _build_google,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
}


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return a singleton LangChain chat model based on config.

    Raises:
        ValueError: If the configured provider is unknown.
        ImportError: If the required provider package is not installed.
    """
    settings = get_settings()
    provider = settings.llm.provider

    builder = _BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. " f"Valid options: {list(_BUILDERS.keys())}"
        )

    logger.info(
        "Initialising LLM",
        provider=provider,
        model=settings.llm.model_name,
        temperature=settings.llm.temperature,
    )

    return builder(settings)


def get_llm_for_agent(agent_name: str, temperature: float | None = None) -> BaseChatModel:
    """
    Return an LLM instance optionally overriding temperature for a specific agent.
    Useful for agents that need deterministic output (temperature=0).
    """
    if temperature is None:
        return get_llm()

    settings = get_settings()
    provider = settings.llm.provider

    # Temporarily override temperature by creating a new instance
    # (lru_cache on get_llm() means we can't reuse it with different temp)
    class _TempOverride:
        llm = settings.llm
        google_api_key = settings.google_api_key
        openai_api_key = settings.openai_api_key
        anthropic_api_key = settings.anthropic_api_key

    override = _TempOverride()
    override.llm = settings.llm.model_copy(update={"temperature": temperature})

    builder = _BUILDERS[provider]
    logger.debug(
        "Creating agent-specific LLM",
        agent=agent_name,
        temperature=temperature,
    )
    return builder(override)  # type: ignore[arg-type]
