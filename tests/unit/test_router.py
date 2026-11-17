"""
tests/unit/test_router.py

Unit tests for src/workflow/router.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.workflow.router import _keyword_heuristic_classifier, route_intent


class TestRouter:
    def test_heuristic_portfolio(self):
        assert (
            _keyword_heuristic_classifier("Here are my holdings: 10 shares of AAPL") == "portfolio"
        )
        assert _keyword_heuristic_classifier("Analyze my portfolio") == "portfolio"

    def test_heuristic_tax(self):
        assert _keyword_heuristic_classifier("Tell me about Section 80C deductions") == "tax"
        assert _keyword_heuristic_classifier("How does LTCG work in Budget 2024?") == "tax"
        assert _keyword_heuristic_classifier("What is a Roth IRA?") == "tax"

    def test_heuristic_goals(self):
        assert _keyword_heuristic_classifier("How does the 4% rule work?") == "goals"
        assert _keyword_heuristic_classifier("Calculate my monthly savings for a house") == "goals"
        assert _keyword_heuristic_classifier("How to build an emergency fund?") == "goals"

    def test_heuristic_news(self):
        assert _keyword_heuristic_classifier("What is the latest news on AAPL?") == "news"
        assert _keyword_heuristic_classifier("Summarize the earnings report") == "news"

    def test_heuristic_market(self):
        assert _keyword_heuristic_classifier("What is the stock price of Tesla?") == "market"
        assert _keyword_heuristic_classifier("How is Nifty today?") == "market"

    def test_route_intent_empty_query(self):
        assert route_intent("") == "qa"

    def test_route_intent_uses_llm_when_heuristic_none(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="tax")

        with patch("src.workflow.router.get_llm_for_agent", return_value=mock_llm):
            intent = route_intent("Explain how indexation works for capital gains")
            assert intent == "tax"

    def test_route_intent_defaults_to_qa_on_error(self):
        with patch("src.workflow.router.get_llm_for_agent", side_effect=Exception("LLM down")):
            intent = route_intent("Random query that triggers exception")
            assert intent == "qa"
