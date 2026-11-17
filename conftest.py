"""
conftest.py — pytest configuration & shared fixtures.
Placed at project root so both unit/ and integration/ can import it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable without installing the package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Set dummy env vars so settings don't fail when API keys are absent
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _dummy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject placeholder API keys for all tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-av-key")


@pytest.fixture
def sample_state():
    """Return a minimal AppState for testing."""
    from src.core.state import initial_state
    return initial_state()


@pytest.fixture
def sample_portfolio():
    """Return a small sample portfolio."""
    from src.core.state import Holding
    return [
        Holding(ticker="AAPL", quantity=10, avg_cost=150.0),
        Holding(ticker="GOOGL", quantity=5, avg_cost=2800.0),
        Holding(ticker="BND", quantity=50, avg_cost=75.0),
    ]
