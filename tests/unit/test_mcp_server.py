"""
tests/unit/test_mcp_server.py

Unit tests for Model Context Protocol (MCP) server tool implementations.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from src.mcp_server.server import (
    _analyze_portfolio_impl,
    _calculate_goal_savings_impl,
    _explain_financial_concept_impl,
    _get_market_indices_impl,
    _get_stock_quote_impl,
)


class TestMCPServerTools:
    def test_get_stock_quote_impl(self):
        fake_quote = {
            "ticker": "AAPL",
            "price": 185.0,
            "currency": "USD",
            "status": "success",
        }
        with patch(
            "src.mcp_server.server.default_market_client.get_quote", return_value=fake_quote
        ):
            res_str = _get_stock_quote_impl("AAPL")
            data = json.loads(res_str)
            assert data["ticker"] == "AAPL"
            assert data["price"] == 185.0

    def test_get_market_indices_impl(self):
        fake_indices = {
            "S&P 500": {"price": 5000.0, "currency": "USD"},
            "Nifty 50": {"price": 24000.0, "currency": "INR"},
        }
        with patch(
            "src.mcp_server.server.default_market_client.get_major_indices",
            return_value=fake_indices,
        ):
            res_str = _get_market_indices_impl("all")
            data = json.loads(res_str)
            assert "S&P 500" in data
            assert "Nifty 50" in data

    def test_analyze_portfolio_impl_valid(self):
        holdings = json.dumps([{"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0}])
        mock_metrics = {
            "total_value": 1850.0,
            "total_cost": 1500.0,
            "total_gain_loss": 350.0,
            "diversification_score": 50,
        }
        with patch("src.mcp_server.server.calculate_portfolio_metrics", return_value=mock_metrics):
            res_str = _analyze_portfolio_impl(holdings)
            data = json.loads(res_str)
            assert data["total_value"] == 1850.0
            assert data["diversification_score"] == 50

    def test_analyze_portfolio_impl_invalid_json(self):
        res_str = _analyze_portfolio_impl("not valid json")
        data = json.loads(res_str)
        assert "error" in data

    def test_calculate_goal_savings_impl(self):
        res_str = _calculate_goal_savings_impl(
            target_amount=100000.0,
            years=10,
            annual_return_pct=8.0,
            currency="USD",
        )
        data = json.loads(res_str)
        assert data["years"] == 10
        assert "required_monthly_savings" in data
        assert "recommended_allocation" in data

    def test_explain_financial_concept_impl(self):
        from src.rag.retriever import RetrievalResult

        mock_retrieval = RetrievalResult(
            documents=[],
            query="test",
            category_filter=None,
            total_found=0,
            sources=[{"title": "Stocks Explained", "category": "basics"}],
        )
        with (
            patch("src.mcp_server.server.get_or_build_vector_store"),
            patch("src.mcp_server.server.FinanceRetriever.retrieve", return_value=mock_retrieval),
        ):
            res_str = _explain_financial_concept_impl("stocks")
            data = json.loads(res_str)
            assert data["query"] == "stocks"
            assert len(data["sources"]) == 1
