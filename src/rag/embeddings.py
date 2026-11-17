"""
src/rag/embeddings.py

Embedding model factory.
Supports OpenAI (primary) and HuggingFace (local fallback).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.core.logging_config import get_logger
from src.core.settings import get_settings

logger = get_logger(__name__)


def _build_openai_embeddings(model_name: str, api_key: str) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=api_key or None,
        # Batch size for efficiency (OpenAI allows up to 2048 inputs per call)
        chunk_size=500,
    )


def _build_google_embeddings(model_name: str, api_key: str) -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key or None,
    )


def _build_huggingface_embeddings(model_name: str) -> Embeddings:
    """Local CPU-based fallback — no API key required."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info("Using local HuggingFace embeddings", model=model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Return a singleton Embeddings instance based on config.

    Falls back to HuggingFace if the configured provider key is missing.
    """
    settings = get_settings()
    provider = settings.embeddings.provider
    model = settings.embeddings.model_name

    logger.info("Initialising embeddings", provider=provider, model=model)

    if provider == "openai":
        if not settings.openai_api_key:
            logger.warning("OpenAI key missing — falling back to HuggingFace embeddings")
            return _build_huggingface_embeddings("all-MiniLM-L6-v2")
        return _build_openai_embeddings(model, settings.openai_api_key)

    elif provider == "google":
        if not settings.google_api_key:
            logger.warning("Google key missing — falling back to HuggingFace embeddings")
            return _build_huggingface_embeddings("all-MiniLM-L6-v2")
        return _build_google_embeddings(model, settings.google_api_key)

    elif provider == "huggingface":
        return _build_huggingface_embeddings(model)

    else:
        raise ValueError(f"Unknown embeddings provider: '{provider}'")
