"""
tests/unit/test_state.py

Unit tests for src/core/state.py
"""

from __future__ import annotations


class TestInitialState:
    def test_initial_state_has_empty_messages(self):
        from src.core.state import initial_state

        s = initial_state()
        assert s["messages"] == []

    def test_initial_state_has_empty_portfolio(self):
        from src.core.state import initial_state

        s = initial_state()
        assert s["portfolio"] == []

    def test_initial_state_has_empty_intent(self):
        from src.core.state import initial_state

        s = initial_state()
        assert s["intent"] == ""

    def test_initial_state_error_is_none(self):
        from src.core.state import initial_state

        s = initial_state()
        assert s["error"] is None

    def test_initial_state_has_empty_agent_outputs(self):
        from src.core.state import initial_state

        s = initial_state()
        assert s["agent_outputs"] == []


class TestHolding:
    def test_holding_fields(self):
        from src.core.state import Holding

        h = Holding(ticker="AAPL", quantity=10, avg_cost=150.0)
        assert h["ticker"] == "AAPL"
        assert h["quantity"] == 10
        assert h["avg_cost"] == 150.0


class TestAgentOutput:
    def test_agent_output_minimal(self):
        from src.core.state import AgentOutput

        out = AgentOutput(agent_name="qa", content="Stocks are ownership shares.")
        assert out["agent_name"] == "qa"
        assert "Stocks" in out["content"]

    def test_agent_output_with_error(self):
        from src.core.state import AgentOutput

        out = AgentOutput(agent_name="market", content="", error="API timeout")
        assert out["error"] == "API timeout"
