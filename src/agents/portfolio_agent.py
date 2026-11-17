"""
src/agents/portfolio_agent.py

Portfolio Analysis Agent.
Reviews user-provided portfolios, retrieves live market quotes (US & India),
calculates performance metrics, sector allocations, and diversification scores,
and provides structured analytical guidance.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base_agent import BaseFinanceAgent
from src.core.llm_client import get_llm_for_agent
from src.core.state import AgentOutput, AppState, Holding
from src.utils.disclaimers import PORTFOLIO_DISCLAIMER
from src.utils.formatters import format_currency, format_percentage
from src.utils.market_data import MarketDataClient, default_market_client, normalize_ticker


def calculate_portfolio_metrics(
    holdings: list[Holding],
    market_client: MarketDataClient,
) -> dict[str, Any]:
    """
    Calculate performance and risk metrics for a list of holdings.
    """
    if not holdings:
        return {
            "total_value": 0.0,
            "total_cost": 0.0,
            "total_gain_loss": 0.0,
            "total_gain_loss_pct": 0.0,
            "holdings_detail": [],
            "sector_allocation": {},
            "diversification_score": 0,
            "currency": "USD",
        }

    holdings_detail = []
    total_value = 0.0
    total_cost = 0.0
    sector_values: dict[str, float] = {}
    primary_currency = "USD"

    for h in holdings:
        ticker = normalize_ticker(h["ticker"])
        quantity = float(h.get("quantity", 0))
        avg_cost = float(h.get("avg_cost", 0))
        cost_basis = quantity * avg_cost

        # Live quote lookup
        quote = market_client.get_quote(ticker)
        current_price = quote.get("price", 0.0) or avg_cost
        curr = quote.get("currency", "INR" if ticker.endswith((".NS", ".BO")) else "USD")
        sector = quote.get("sector", "General")

        if ticker.endswith((".NS", ".BO")):
            primary_currency = "INR"

        current_value = quantity * current_price
        gain_loss = current_value - cost_basis
        gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0.0

        total_value += current_value
        total_cost += cost_basis
        sector_values[sector] = sector_values.get(sector, 0.0) + current_value

        holdings_detail.append(
            {
                "ticker": ticker,
                "quantity": quantity,
                "avg_cost": avg_cost,
                "current_price": current_price,
                "cost_basis": round(cost_basis, 2),
                "current_value": round(current_value, 2),
                "gain_loss": round(gain_loss, 2),
                "gain_loss_pct": round(gain_loss_pct, 2),
                "currency": curr,
                "sector": sector,
            }
        )

    # Calculate weights and sector allocation
    sector_allocation: dict[str, float] = {}
    for h in holdings_detail:
        weight = (h["current_value"] / total_value * 100) if total_value > 0 else 0.0
        h["weight_pct"] = round(weight, 2)

    for sector, s_val in sector_values.items():
        s_weight = (s_val / total_value * 100) if total_value > 0 else 0.0
        sector_allocation[sector] = round(s_weight, 2)

    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0.0

    # Calculate diversification score (0 to 100)
    # Higher score for more holdings, lower single-stock concentration, more sectors
    num_holdings = len(holdings_detail)
    num_sectors = len(sector_values)
    max_weight = max([h["weight_pct"] for h in holdings_detail]) if holdings_detail else 100.0

    # Penalize single holding > 30%, reward 5+ holdings across 3+ sectors
    score = min(
        100,
        int((min(num_holdings, 10) * 4) + (min(num_sectors, 6) * 6) + max(0, (50 - max_weight))),
    )
    score = max(10, min(100, score))

    return {
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_gain_loss": round(total_gain_loss, 2),
        "total_gain_loss_pct": round(total_gain_loss_pct, 2),
        "holdings_detail": holdings_detail,
        "sector_allocation": sector_allocation,
        "diversification_score": score,
        "currency": primary_currency,
    }


def parse_holdings_from_text(text: str) -> list[Holding]:
    """
    Heuristic regex parser to extract holdings from informal user text.
    Matches formats like:
      - '10 shares of AAPL at 150'
      - 'AAPL: 10 @ 150'
      - 'RELIANCE 50 shares bought at 2800'
    """
    holdings: list[Holding] = []
    # Pattern 1: TICKER [colon/dash] QUANTITY [at/@] COST
    pattern1 = re.compile(
        r"([A-Za-z0-9\.\^]+)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:shares?|qty)?\s*(?:at|@)\s*(?:[\$₹Rs\.]*)?\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    # Pattern 2: QUANTITY [shares of] TICKER [at/@] COST
    pattern2 = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:shares?|units?)\s*(?:of)?\s*([A-Za-z0-9\.\^]+)\s*(?:at|@|bought at)\s*(?:[\$₹Rs\.]*)?\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )

    for match in pattern1.finditer(text):
        ticker, qty, cost = match.groups()
        if ticker.lower() not in ("buy", "sell", "holding", "have", "i"):
            holdings.append(
                Holding(ticker=ticker.upper(), quantity=float(qty), avg_cost=float(cost))
            )

    if not holdings:
        for match in pattern2.finditer(text):
            qty, ticker, cost = match.groups()
            holdings.append(
                Holding(ticker=ticker.upper(), quantity=float(qty), avg_cost=float(cost))
            )

    return holdings


class PortfolioAnalysisAgent(BaseFinanceAgent):
    """Specialized agent for portfolio review, risk analysis, and asset allocation."""

    name: str = "Portfolio Analysis Agent"
    description: str = "Analyzes holdings, sector exposure, risk metrics, and rebalancing needs."

    def __init__(self, market_client: MarketDataClient | None = None) -> None:
        super().__init__()
        self.market_client = market_client or default_market_client
        self.llm = get_llm_for_agent("portfolio_agent", temperature=0.0)

    def invoke(self, state: AppState) -> AgentOutput:
        """Analyze portfolio from state or user input, returning commentary and metrics."""
        holdings = state.get("portfolio", [])

        # If no portfolio in state, check if user provided one in the latest message
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        user_text = getattr(last_message, "content", "") if last_message else ""

        if not holdings and user_text:
            parsed = parse_holdings_from_text(user_text)
            if parsed:
                holdings = parsed
                # Note: state updates in LangGraph happen via state returns in node

        if not holdings:
            return AgentOutput(
                agent_name=self.name,
                content=(
                    "I don't see any portfolio holdings in your profile yet! "
                    "You can enter your holdings like this:\n\n"
                    "- *10 shares of AAPL at 150*\n"
                    "- *5 shares of RELIANCE.NS at 2900*\n"
                    "- *50 units of VTI at 220*\n\n"
                    "Or use the Portfolio tab in the application to add your assets."
                    + PORTFOLIO_DISCLAIMER
                ),
                sources=[],
                data={"metrics": {}},
            )

        # Calculate metrics using live market quotes
        metrics = calculate_portfolio_metrics(holdings, self.market_client)

        # Build prompt for LLM narrative commentary
        currency = metrics.get("currency", "USD")
        formatted_val = format_currency(metrics["total_value"], currency)
        formatted_gl = format_currency(metrics["total_gain_loss"], currency)
        formatted_pct = format_percentage(metrics["total_gain_loss_pct"])

        summary_table = "\n".join(
            [
                f"- {h['ticker']}: {h['quantity']} shares | Current: {format_currency(h['current_price'], h['currency'])} | "
                f"Value: {format_currency(h['current_value'], h['currency'])} | P&L: {format_percentage(h['gain_loss_pct'])} "
                f"({h['weight_pct']}% of portfolio) | Sector: {h['sector']}"
                for h in metrics["holdings_detail"]
            ]
        )

        analysis_prompt = (
            f"You are a professional portfolio analyst. Review this portfolio objectively:\n\n"
            f"Total Value: {formatted_val}\n"
            f"Total P&L: {formatted_gl} ({formatted_pct})\n"
            f"Diversification Score: {metrics['diversification_score']}/100\n"
            f"Sector Allocations: {json.dumps(metrics['sector_allocation'])}\n\n"
            f"Holdings:\n{summary_table}\n\n"
            f"Provide a structured analysis covering:\n"
            f"1. **Performance Overview**: Brief assessment of overall gains/losses.\n"
            f"2. **Concentration & Sector Risk**: Highlight if any single asset or sector exceeds 25-30%.\n"
            f"3. **Diversification & Recommendations**: Constructive suggestions (e.g. adding index funds, international exposure, or fixed income).\n"
            f"Be encouraging, objective, and concise. Avoid recommending specific single stock buys."
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [
                SystemMessage(content="You are an expert investment portfolio analyst."),
                HumanMessage(content=analysis_prompt),
            ]
        )

        content = str(response.content) + PORTFOLIO_DISCLAIMER

        return AgentOutput(
            agent_name=self.name,
            content=content,
            sources=[
                {
                    "title": "Principles of Strategic Asset Allocation",
                    "category": "portfolio",
                    "source": "Knowledge Base",
                }
            ],
            data={"metrics": metrics},
            error=None,
        )
