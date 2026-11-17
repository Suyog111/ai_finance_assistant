"""
tests/integration/test_workflow_e2e.py

Integration test suite for the end-to-end LangGraph workflow.
Tests multi-turn conversation memory, checkpointer state persistence,
and cross-agent routing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.core.state import AgentOutput, Holding
from src.workflow.graph import create_finance_graph, run_finance_turn


class TestWorkflowEndToEnd:
    @pytest.fixture
    def mock_agents(self):
        qa_out = AgentOutput(
            agent_name="Finance Q&A Agent",
            content="Compound interest grows money exponentially over time.",
            sources=[{"title": "The Power of Compound Interest", "category": "basics"}],
            data={},
        )
        tax_out = AgentOutput(
            agent_name="Tax Education Agent",
            content="Section 80C provides up to ₹1.5 Lakh deductions in India.",
            sources=[{"title": "Tax-Saving Investments", "category": "tax"}],
            data={},
        )
        return {"qa": qa_out, "tax": tax_out}

    def test_multi_turn_conversation_context(self, mock_agents):
        """Verify that conversation state and messages accumulate across multiple turns."""
        graph = create_finance_graph()
        thread_id = "integration_test_session_42"

        with (
            patch("src.workflow.graph.registry.qa._safe_invoke", return_value=mock_agents["qa"]),
            patch("src.workflow.graph.registry.tax._safe_invoke", return_value=mock_agents["tax"]),
            patch("src.workflow.router.route_intent", side_effect=["qa", "tax"]),
        ):

            # Turn 1: Ask about compounding (routed to QA)
            state_turn1 = run_finance_turn(
                user_input="How does compounding work?",
                thread_id=thread_id,
                graph=graph,
            )
            assert state_turn1["intent"] == "qa"
            assert "Compound interest grows" in state_turn1["final_response"]
            assert len(state_turn1["messages"]) == 2  # [Human, AI]

            # Turn 2: Ask follow-up about Section 80C (routed to Tax)
            state_turn2 = run_finance_turn(
                user_input="Can I use Section 80C for that?",
                thread_id=thread_id,
                graph=graph,
            )
            assert state_turn2["intent"] == "tax"
            assert "Section 80C" in state_turn2["final_response"]
            # Conversation history should now contain all 4 messages preserved by checkpointer!
            assert len(state_turn2["messages"]) == 4  # [Human, AI, Human, AI]

    def test_portfolio_persistence_in_state(self):
        """Verify that user holdings persist in state across graph invocations."""
        graph = create_finance_graph()
        thread_id = "portfolio_session_99"

        portfolio_out = AgentOutput(
            agent_name="Portfolio Analysis Agent",
            content="Your portfolio has 10 shares of AAPL with total value $2,000.",
            sources=[],
            data={"metrics": {"total_value": 2000.0}},
        )

        with (
            patch("src.workflow.graph.registry.portfolio._safe_invoke", return_value=portfolio_out),
            patch("src.workflow.router.route_intent", return_value="portfolio"),
        ):

            initial_holdings = [Holding(ticker="AAPL", quantity=10, avg_cost=150.0)]

            state = run_finance_turn(
                user_input="Review my portfolio",
                thread_id=thread_id,
                graph=graph,
                portfolio=initial_holdings,
            )

            assert state["intent"] == "portfolio"
            assert len(state["portfolio"]) == 1
            assert state["portfolio"][0]["ticker"] == "AAPL"
            assert "Your portfolio has 10 shares" in state["final_response"]
