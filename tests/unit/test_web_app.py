"""
tests/unit/test_web_app.py

Unit tests for Streamlit UI helpers, session state, and Plotly chart generators.
"""

from __future__ import annotations

from unittest.mock import patch

from src.web_app.components.charts import (
    plot_asset_allocation_bar,
    plot_compound_growth,
    plot_holding_gains,
    plot_sector_allocation,
)
from src.web_app.session import clear_chat_history, init_session_state


class FakeSessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, val):
        self[name] = val


class TestSessionState:
    def test_init_session_state(self):
        mock_state = FakeSessionState()
        with patch("streamlit.session_state", mock_state):
            init_session_state()
            assert "thread_id" in mock_state
            assert "messages" in mock_state
            assert "portfolio" in mock_state
            assert "user_profile" in mock_state
            assert len(mock_state.portfolio) >= 3

    def test_clear_chat_history(self):
        mock_state = FakeSessionState(
            {
                "thread_id": "old_id",
                "messages": [{"role": "user", "content": "hello"}],
                "portfolio": [],
                "user_profile": {},
            }
        )
        with patch("streamlit.session_state", mock_state):
            clear_chat_history()
            assert mock_state.thread_id != "old_id"
            # Should re-initialize with greeting
            assert len(mock_state.messages) == 1


class TestPlotlyCharts:
    def test_plot_sector_allocation(self):
        sectors = {"Technology": 60.0, "Energy": 25.0, "Financials": 15.0}
        fig = plot_sector_allocation(sectors)
        assert fig is not None
        assert len(fig.data) == 1

    def test_plot_sector_allocation_empty(self):
        fig = plot_sector_allocation({})
        assert fig is not None

    def test_plot_holding_gains(self):
        holdings = [
            {"ticker": "AAPL", "gain_loss_pct": 15.2},
            {"ticker": "MSFT", "gain_loss_pct": -4.8},
        ]
        fig = plot_holding_gains(holdings)
        assert fig is not None
        assert len(fig.data) == 1

    def test_plot_holding_gains_empty(self):
        fig = plot_holding_gains([])
        assert fig is not None

    def test_plot_compound_growth(self):
        fig = plot_compound_growth(
            years=10,
            monthly_contrib=500.0,
            annual_return_pct=10.0,
            initial_investment=1000.0,
            currency_symbol="$",
        )
        assert fig is not None
        # Should have 2 traces: Principal Invested and Compound Interest Earned
        assert len(fig.data) == 2

    def test_plot_asset_allocation_bar(self):
        alloc = {"Equities": 70, "Fixed Income": 25, "Cash": 5}
        fig = plot_asset_allocation_bar(alloc)
        assert fig is not None
        assert len(fig.data) == 3
