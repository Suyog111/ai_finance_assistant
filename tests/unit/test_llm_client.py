"""
tests/unit/test_llm_client.py

Unit tests for src/core/llm_client.py
Tests use mocking — no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGetLLM:
    def setup_method(self):
        from src.core.llm_client import get_llm
        from src.core.settings import get_settings

        get_llm.cache_clear()
        get_settings.cache_clear()

    def teardown_method(self):
        from src.core.llm_client import get_llm
        from src.core.settings import get_settings

        get_llm.cache_clear()
        get_settings.cache_clear()

    def test_get_llm_openai_provider(self, monkeypatch):
        """With openai as the configured provider, get_llm() should use the openai builder."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        mock_llm = MagicMock()

        with patch.dict("src.core.llm_client._BUILDERS", {"openai": lambda s: mock_llm}):
            from src.core.llm_client import get_llm

            get_llm.cache_clear()
            llm = get_llm()
            assert llm is mock_llm

    def test_get_llm_raises_on_unknown_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")
        from src.core.settings import get_settings

        get_settings.cache_clear()

        # Patch settings to return an unknown provider
        with patch("src.core.llm_client.get_settings") as mock_settings_fn:
            mock_s = MagicMock()
            mock_s.llm.provider = "unknown_provider"
            mock_settings_fn.return_value = mock_s

            from src.core.llm_client import get_llm

            get_llm.cache_clear()

            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_llm()

    def test_get_llm_is_cached(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        mock_llm = MagicMock()

        with patch("src.core.llm_client._build_google", return_value=mock_llm):
            from src.core.llm_client import get_llm

            get_llm.cache_clear()
            llm1 = get_llm()
            llm2 = get_llm()
            assert llm1 is llm2  # Same cached instance


class TestGetLLMForAgent:
    def test_returns_llm_with_default_temperature(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        mock_llm = MagicMock()

        with patch("src.core.llm_client.get_llm", return_value=mock_llm):
            from src.core.llm_client import get_llm_for_agent

            result = get_llm_for_agent("qa_agent")
            assert result is mock_llm

    def test_returns_new_instance_with_custom_temperature(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        mock_llm = MagicMock()

        with patch.dict("src.core.llm_client._BUILDERS", {"openai": lambda s: mock_llm}):
            from src.core.llm_client import get_llm_for_agent
            from src.core.settings import get_settings

            get_settings.cache_clear()
            result = get_llm_for_agent("portfolio_agent", temperature=0.0)
            assert result is mock_llm
