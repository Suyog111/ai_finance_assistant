"""
src/mcp_server/server.py

Model Context Protocol (MCP) Server for the AI Finance Assistant.
Exposes specialized financial tools to Claude Desktop and any MCP-compatible LLM client:
  - get_stock_quote: Real-time market quotes (US & India)
  - get_market_indices: Benchmark indices (S&P 500, NASDAQ, Nifty 50, Sensex)
  - analyze_portfolio: Portfolio metrics, sector weights, diversification score
  - calculate_goal_savings: SIP / compounding math and asset allocations
  - explain_financial_concept: Grounded RAG lookup over the financial knowledge base
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]
    except ImportError:
        MCPServer = None  # type: ignore[assignment,misc]

from src.agents.goal_agent import calculate_monthly_savings_needed, get_recommended_allocation
from src.agents.portfolio_agent import calculate_portfolio_metrics
from src.core.state import Holding
from src.rag.retriever import FinanceRetriever
from src.rag.vector_store import get_or_build_vector_store
from src.utils.disclaimers import GENERAL_DISCLAIMER
from src.utils.formatters import format_currency
from src.utils.market_data import default_market_client, normalize_ticker

# Create MCP server instance
server = MCPServer("ai_finance_assistant") if MCPServer else None


def _get_stock_quote_impl(ticker: str) -> str:
    """Core implementation for stock quote tool."""
    sym = normalize_ticker(ticker)
    quote = default_market_client.get_quote(sym)
    return json.dumps(quote, indent=2)


def _get_market_indices_impl(market: str = "all") -> str:
    """Core implementation for market indices tool."""
    indices = default_market_client.get_major_indices(market=market)
    return json.dumps(indices, indent=2)


def _analyze_portfolio_impl(holdings_json: str) -> str:
    """Core implementation for portfolio analysis tool."""
    try:
        data = json.loads(holdings_json)
        holdings = [
            Holding(
                ticker=h["ticker"],
                quantity=float(h["quantity"]),
                avg_cost=float(h["avg_cost"]),
            )
            for h in data
        ]
        metrics = calculate_portfolio_metrics(holdings, default_market_client)
        return json.dumps(metrics, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Invalid holdings format: {str(exc)}"})


def _calculate_goal_savings_impl(
    target_amount: float,
    years: float,
    annual_return_pct: float = 8.0,
    currency: str = "USD",
) -> str:
    """Core implementation for goal savings calculation tool."""
    monthly_needed = calculate_monthly_savings_needed(
        target_amount=target_amount,
        years=years,
        annual_return_pct=annual_return_pct,
    )
    alloc = get_recommended_allocation(years, "moderate")
    res = {
        "target_amount": format_currency(target_amount, currency),
        "years": years,
        "assumed_annual_return": f"{annual_return_pct}%",
        "required_monthly_savings": format_currency(monthly_needed, currency),
        "recommended_allocation": alloc,
        "disclaimer": GENERAL_DISCLAIMER.strip(),
    }
    return json.dumps(res, indent=2)


def _explain_financial_concept_impl(concept: str, category: str = "all") -> str:
    """Core implementation for conceptual RAG lookup tool."""
    try:
        store = get_or_build_vector_store()
        retriever = FinanceRetriever(store)
        cat_filter = None if category.lower() in ("all", "none", "") else category.lower()
        retrieval = retriever.retrieve(concept, category=cat_filter, top_k=3)
        return json.dumps(
            {
                "query": concept,
                "sources": retrieval.sources,
                "context_excerpts": retrieval.context_text()[:1500],
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": f"RAG lookup failed: {str(exc)}"})


# Register tools if MCPServer is available
if server:

    @server.tool()
    def get_stock_quote(ticker: str) -> str:
        """Fetch real-time stock price, 52-week range, P/E ratio, and market cap for US and Indian tickers."""
        return _get_stock_quote_impl(ticker)

    @server.tool()
    def get_market_indices(market: str = "all") -> str:
        """Fetch real-time values for major benchmark indices (S&P 500, NASDAQ, Nifty 50, Sensex)."""
        return _get_market_indices_impl(market)

    @server.tool()
    def analyze_portfolio(holdings_json: str) -> str:
        """Analyze a portfolio given a JSON list of holdings: [{"ticker": "AAPL", "quantity": 10, "avg_cost": 150}]."""
        return _analyze_portfolio_impl(holdings_json)

    @server.tool()
    def calculate_goal_savings(
        target_amount: float,
        years: float,
        annual_return_pct: float = 8.0,
        currency: str = "USD",
    ) -> str:
        """Calculate required monthly savings/SIP to reach a financial goal, plus recommended asset allocation."""
        return _calculate_goal_savings_impl(target_amount, years, annual_return_pct, currency)

    @server.tool()
    def explain_financial_concept(concept: str, category: str = "all") -> str:
        """Retrieve verified educational knowledge on personal finance, investing, taxes, and market concepts."""
        return _explain_financial_concept_impl(concept, category)


def main() -> None:
    """Run the MCP server over stdio for Claude Desktop integration."""
    if server:
        server.run()


if __name__ == "__main__":
    main()
