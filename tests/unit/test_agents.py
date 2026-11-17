"""
tests/unit/test_agents.py

Comprehensive unit tests for all six specialized finance agents:
  1. FinanceQAAgent
  2. PortfolioAnalysisAgent
  3. MarketAnalysisAgent
  4. GoalPlanningAgent
  5. NewsSynthesizerAgent
  6. TaxEducationAgent
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.goal_agent import (
    GoalPlanningAgent,
    calculate_monthly_savings_needed,
    get_recommended_allocation,
)
from src.agents.market_agent import MarketAnalysisAgent, extract_potential_tickers
from src.agents.news_agent import NewsSynthesizerAgent
from src.agents.portfolio_agent import (
    PortfolioAnalysisAgent,
    calculate_portfolio_metrics,
    parse_holdings_from_text,
)
from src.agents.qa_agent import FinanceQAAgent
from src.agents.tax_agent import TaxEducationAgent
from src.core.state import Holding, initial_state


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Analytical response from finance agent.")
    return llm


@pytest.fixture
def mock_retriever():
    from src.rag.retriever import RetrievalResult

    r = MagicMock()
    r.retrieve.return_value = RetrievalResult(
        documents=[],
        query="test query",
        category_filter=None,
        total_found=0,
    )
    return r


# ---------------------------------------------------------------------------
# 1. FinanceQAAgent Tests
# ---------------------------------------------------------------------------
class TestFinanceQAAgent:
    def test_qa_agent_empty_state_greeting(self):
        with (
            patch("src.agents.qa_agent.get_or_build_vector_store"),
            patch("src.agents.qa_agent.get_llm_for_agent"),
        ):
            agent = FinanceQAAgent()
            state = initial_state()
            out = agent.invoke(state)
            assert "Hello" in out["content"]
            assert out["agent_name"] == "Finance Q&A Agent"

    def test_qa_agent_answers_with_disclaimer(self, mock_llm):
        with (
            patch("src.agents.qa_agent.get_or_build_vector_store"),
            patch("src.agents.qa_agent.get_llm_for_agent", return_value=mock_llm),
        ):
            agent = FinanceQAAgent()
            state = initial_state()
            state["messages"] = [HumanMessage(content="What is an index fund?")]
            out = agent.invoke(state)
            assert "Disclaimer" in out["content"]
            assert out["agent_name"] == "Finance Q&A Agent"


# ---------------------------------------------------------------------------
# 2. PortfolioAnalysisAgent Tests
# ---------------------------------------------------------------------------
class TestPortfolioAnalysisAgent:
    def test_parse_holdings_from_text(self):
        text = "I hold 10 shares of AAPL at 150 and 5 shares of RELIANCE.NS at 2900"
        holdings = parse_holdings_from_text(text)
        assert len(holdings) == 2
        assert holdings[0]["ticker"] == "AAPL"
        assert holdings[0]["quantity"] == 10.0
        assert holdings[0]["avg_cost"] == 150.0

    def test_calculate_portfolio_metrics(self):
        mock_client = MagicMock()
        mock_client.get_quote.return_value = {
            "price": 200.0,
            "currency": "USD",
            "sector": "Technology",
            "status": "success",
        }

        holdings = [Holding(ticker="AAPL", quantity=10, avg_cost=150.0)]
        metrics = calculate_portfolio_metrics(holdings, mock_client)

        assert metrics["total_cost"] == 1500.0
        assert metrics["total_value"] == 2000.0
        assert metrics["total_gain_loss"] == 500.0
        assert metrics["total_gain_loss_pct"] == pytest.approx(33.33, 0.1)
        assert metrics["diversification_score"] > 0
        assert "Technology" in metrics["sector_allocation"]

    def test_portfolio_agent_no_holdings_guidance(self, mock_llm):
        with patch("src.agents.portfolio_agent.get_llm_for_agent", return_value=mock_llm):
            agent = PortfolioAnalysisAgent()
            state = initial_state()
            out = agent.invoke(state)
            assert "don't see any portfolio holdings" in out["content"]

    def test_portfolio_agent_analyzes_holdings(self, mock_llm):
        mock_client = MagicMock()
        mock_client.get_quote.return_value = {"price": 100.0, "currency": "USD", "sector": "Tech"}

        with patch("src.agents.portfolio_agent.get_llm_for_agent", return_value=mock_llm):
            agent = PortfolioAnalysisAgent(market_client=mock_client)
            state = initial_state()
            state["portfolio"] = [Holding(ticker="XYZ", quantity=5, avg_cost=90.0)]
            out = agent.invoke(state)
            assert out["agent_name"] == "Portfolio Analysis Agent"
            assert "metrics" in out["data"]
            assert "Disclaimer" in out["content"]


# ---------------------------------------------------------------------------
# 3. MarketAnalysisAgent Tests
# ---------------------------------------------------------------------------
class TestMarketAnalysisAgent:
    def test_extract_potential_tickers(self):
        text = "What is the price of AAPL and MSFT today?"
        tickers = extract_potential_tickers(text)
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_market_agent_invoke(self, mock_llm):
        mock_client = MagicMock()
        mock_client.get_major_indices.return_value = {
            "S&P 500": {"price": 5000.0, "change_percent": 0.5, "currency": "USD"}
        }
        mock_client.get_quote.return_value = {
            "status": "success",
            "price": 180.0,
            "currency": "USD",
        }

        with (
            patch("src.agents.market_agent.get_llm_for_agent", return_value=mock_llm),
            patch("src.agents.market_agent.get_or_build_vector_store"),
        ):
            agent = MarketAnalysisAgent(market_client=mock_client)
            state = initial_state()
            state["messages"] = [HumanMessage(content="Give me a market overview of AAPL")]
            out = agent.invoke(state)
            assert out["agent_name"] == "Market Analysis Agent"
            assert "Disclaimer" in out["content"]


# ---------------------------------------------------------------------------
# 4. GoalPlanningAgent Tests
# ---------------------------------------------------------------------------
class TestGoalPlanningAgent:
    def test_calculate_monthly_savings_needed(self):
        # Target $100k in 10 years at 8% return
        pmt = calculate_monthly_savings_needed(100_000, 10, 8.0)
        assert pmt > 0
        assert pmt < 1000  # Compound growth reduces required raw cash

    def test_get_recommended_allocation_short_term(self):
        alloc = get_recommended_allocation(2, "conservative")
        assert alloc["Cash / HYSA / Liquid"] == 80
        assert alloc["Equities / Stocks"] == 0

    def test_get_recommended_allocation_long_term(self):
        alloc = get_recommended_allocation(20, "aggressive")
        assert alloc["Equities / Stocks"] >= 85

    def test_goal_agent_invoke(self, mock_llm):
        with (
            patch("src.agents.goal_agent.get_llm_for_agent", return_value=mock_llm),
            patch("src.agents.goal_agent.get_or_build_vector_store"),
        ):
            agent = GoalPlanningAgent()
            state = initial_state()
            state["messages"] = [
                HumanMessage(content="I want to save $500,000 in 15 years for retirement")
            ]
            out = agent.invoke(state)
            assert out["agent_name"] == "Goal Planning Agent"
            assert out["data"]["years"] == 15.0
            assert "Disclaimer" in out["content"]


# ---------------------------------------------------------------------------
# 5. NewsSynthesizerAgent Tests
# ---------------------------------------------------------------------------
class TestNewsSynthesizerAgent:
    def test_news_agent_invoke(self, mock_llm):
        mock_client = MagicMock()
        mock_client.get_ticker_news.return_value = [
            {
                "title": "Company Beats Earnings",
                "publisher": "Bloomberg",
                "link": "https://example.com",
            }
        ]

        with (
            patch("src.agents.news_agent.get_llm_for_agent", return_value=mock_llm),
            patch("src.agents.news_agent.get_or_build_vector_store"),
        ):
            agent = NewsSynthesizerAgent(market_client=mock_client)
            state = initial_state()
            state["messages"] = [HumanMessage(content="What is the latest news on AAPL?")]
            out = agent.invoke(state)
            assert out["agent_name"] == "News Synthesizer Agent"
            assert "Disclaimer" in out["content"]
            assert out["data"]["news_count"] == 1


# ---------------------------------------------------------------------------
# 6. TaxEducationAgent Tests
# ---------------------------------------------------------------------------
class TestTaxEducationAgent:
    def test_tax_agent_empty_state(self):
        with (
            patch("src.agents.tax_agent.get_llm_for_agent"),
            patch("src.agents.tax_agent.get_or_build_vector_store"),
        ):
            agent = TaxEducationAgent()
            state = initial_state()
            out = agent.invoke(state)
            assert "Hello" in out["content"]

    def test_tax_agent_answers_with_tax_disclaimer(self, mock_llm):
        with (
            patch("src.agents.tax_agent.get_llm_for_agent", return_value=mock_llm),
            patch("src.agents.tax_agent.get_or_build_vector_store"),
        ):
            agent = TaxEducationAgent()
            state = initial_state()
            state["messages"] = [HumanMessage(content="Explain Section 80C deductions in India")]
            out = agent.invoke(state)
            assert out["agent_name"] == "Tax Education Agent"
            assert "Tax Disclaimer" in out["content"]
