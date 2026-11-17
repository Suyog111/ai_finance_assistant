"""
tests/unit/test_settings.py

Unit tests for src/core/settings.py
"""

from __future__ import annotations


class TestAppSettings:
    def test_loads_without_error(self):
        """Settings should load with just dummy env vars set."""
        from src.core.settings import get_settings

        # Clear lru_cache to pick up monkeypatched env
        get_settings.cache_clear()
        s = get_settings()
        assert s is not None

    def test_default_provider_is_openai(self):
        """Provider defaults to openai as configured in config.yaml."""
        from src.core.settings import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.llm.provider == "openai"

    def test_active_api_key_returns_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        from src.core.settings import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.active_api_key() == "sk-test-123"

    def test_rag_defaults(self):
        from src.core.settings import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.rag.chunk_size == 512
        assert s.rag.chunk_overlap == 64
        assert s.rag.top_k == 5

    def test_conversation_max_history(self):
        from src.core.settings import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.conversation.max_history_turns == 20

    def test_market_data_cache_ttl(self):
        from src.core.settings import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.market_data.cache_ttl_seconds == 60

    def teardown_method(self):
        from src.core.settings import get_settings

        get_settings.cache_clear()
