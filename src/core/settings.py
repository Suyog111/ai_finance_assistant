"""
src/core/settings.py

Centralised configuration loaded from config.yaml + environment variables.
Pydantic-settings validates and type-checks every field at startup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/core → src → project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_yaml() -> dict:
    """Load config.yaml; return empty dict if file missing."""
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open() as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml()


class LLMSettings(BaseSettings):
    provider: Literal["google", "openai", "anthropic"] = Field(
        default=_yaml.get("llm", {}).get("provider", "google")
    )
    model_name: str = Field(default=_yaml.get("llm", {}).get("model_name", "gemini-1.5-flash"))
    temperature: float = Field(default=_yaml.get("llm", {}).get("temperature", 0.1))
    max_tokens: int = Field(default=_yaml.get("llm", {}).get("max_tokens", 2048))

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


class EmbeddingSettings(BaseSettings):
    provider: Literal["google", "openai", "huggingface"] = Field(
        default=_yaml.get("embeddings", {}).get("provider", "google")
    )
    model_name: str = Field(
        default=_yaml.get("embeddings", {}).get("model_name", "models/text-embedding-004")
    )

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")


class VectorStoreSettings(BaseSettings):
    type: Literal["faiss", "chroma"] = Field(
        default=_yaml.get("vector_store", {}).get("type", "faiss")
    )
    index_path: Path = Field(
        default=PROJECT_ROOT / _yaml.get("vector_store", {}).get("index_path", "data/faiss_index")
    )
    chroma_persist_dir: Path = Field(
        default=PROJECT_ROOT
        / _yaml.get("vector_store", {}).get("chroma_persist_dir", "data/chroma_db")
    )
    collection_name: str = Field(
        default=_yaml.get("vector_store", {}).get("collection_name", "finance_knowledge")
    )

    model_config = SettingsConfigDict(env_prefix="VS_", extra="ignore")


class RAGSettings(BaseSettings):
    chunk_size: int = Field(default=_yaml.get("rag", {}).get("chunk_size", 512))
    chunk_overlap: int = Field(default=_yaml.get("rag", {}).get("chunk_overlap", 64))
    top_k: int = Field(default=_yaml.get("rag", {}).get("top_k", 5))
    score_threshold: float = Field(default=_yaml.get("rag", {}).get("score_threshold", 0.3))

    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")


class MarketDataSettings(BaseSettings):
    primary_provider: Literal["yfinance", "alpha_vantage"] = Field(
        default=_yaml.get("market_data", {}).get("primary_provider", "yfinance")
    )
    cache_ttl_seconds: int = Field(
        default=_yaml.get("market_data", {}).get("cache_ttl_seconds", 60)
    )
    max_retries: int = Field(default=_yaml.get("market_data", {}).get("max_retries", 3))
    retry_backoff_multiplier: int = Field(
        default=_yaml.get("market_data", {}).get("retry_backoff_multiplier", 2)
    )

    model_config = SettingsConfigDict(env_prefix="MARKET_", extra="ignore")


class ConversationSettings(BaseSettings):
    max_history_turns: int = Field(
        default=_yaml.get("conversation", {}).get("max_history_turns", 20)
    )
    session_timeout_minutes: int = Field(
        default=_yaml.get("conversation", {}).get("session_timeout_minutes", 60)
    )

    model_config = SettingsConfigDict(env_prefix="CONV_", extra="ignore")


class LoggingSettings(BaseSettings):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default=_yaml.get("logging", {}).get("level", "INFO")
    )
    format: Literal["json", "text"] = Field(default=_yaml.get("logging", {}).get("format", "json"))
    log_file: Path = Field(
        default=PROJECT_ROOT / _yaml.get("logging", {}).get("log_file", "logs/app.log")
    )

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")


class AppSettings(BaseSettings):
    """Top-level settings; also exposes API keys from environment."""

    # API Keys
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    alpha_vantage_api_key: str = Field(default="", alias="ALPHA_VANTAGE_API_KEY")
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")

    environment: Literal["development", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )

    # Sub-settings (not loaded from env directly)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    @field_validator("google_api_key", "openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def _coerce_none(cls, v: object) -> str:
        return str(v) if v is not None else ""

    def active_api_key(self) -> str:
        """Return the API key for the configured LLM provider."""
        mapping = {
            "google": self.google_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        return mapping.get(self.llm.provider, "")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return singleton settings instance (cached after first load)."""
    return AppSettings()
