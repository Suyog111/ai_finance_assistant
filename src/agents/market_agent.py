"""
src/agents/market_agent.py

Market Analysis Agent.
Provides real-time market insights, live quotes (US & India),
major index tracking (S&P 500, NASDAQ, Nifty 50, Sensex), and market analysis.
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.base_agent import BaseFinanceAgent
from src.core.llm_client import get_llm_for_agent
from src.core.state import AgentOutput, AppState
from src.rag.retriever import FinanceRetriever
from src.rag.vector_store import get_or_build_vector_store
from src.utils.disclaimers import GENERAL_DISCLAIMER
from src.utils.formatters import format_currency, format_large_number, format_percentage
from src.utils.market_data import MarketDataClient, default_market_client, normalize_ticker


def extract_potential_tickers(text: str) -> list[str]:
    """Extract stock symbols or common company names from user text."""
    # Look for capitalized ticker symbols ($AAPL, MSFT, RELIANCE, TCS)
    tokens = re.findall(r"\b\$?([A-Za-z\.\^]{2,12})\b", text)
    stopwords = {
        "WHAT",
        "HOW",
        "IS",
        "THE",
        "TODAY",
        "STOCK",
        "SHARE",
        "PRICE",
        "MARKET",
        "LOOK",
        "SHOW",
        "TELL",
        "ABOUT",
        "FOR",
    }
    candidates = []
    for t in tokens:
        clean = t.replace("$", "").upper()
        if clean not in stopwords and len(clean) >= 2:
            candidates.append(normalize_ticker(clean))
    return candidates


class MarketAnalysisAgent(BaseFinanceAgent):
    """Specialized agent for real-time market data, stock quotes, and index trends."""

    name: str = "Market Analysis Agent"
    description: str = (
        "Provides live market quotes, index overviews (US & India), and market analysis."
    )

    def __init__(
        self,
        market_client: MarketDataClient | None = None,
        vector_store: Any | None = None,
    ) -> None:
        super().__init__()
        self.market_client = market_client or default_market_client
        self.llm = get_llm_for_agent("market_agent", temperature=0.1)

        try:
            store = vector_store or get_or_build_vector_store()
            self.retriever = FinanceRetriever(store)
        except Exception:
            self.retriever = None  # type: ignore[assignment]

    def invoke(self, state: AppState) -> AgentOutput:
        """Process user market query and return real-time data + analysis."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        user_query = getattr(last_message, "content", "") if last_message else "Market Overview"

        # Check for specific ticker mentions
        candidates = extract_potential_tickers(user_query)
        quotes_data: dict[str, Any] = {}

        # 1. If user asks about major indices
        is_index_query = any(
            w in user_query.lower()
            for w in [
                "index",
                "indices",
                "nifty",
                "sensex",
                "s&p",
                "market overview",
                "nasdaq",
                "dow",
            ]
        )
        index_data = {}
        if is_index_query or not candidates:
            # Determine market preference
            if any(w in user_query.lower() for w in ["india", "nifty", "sensex", "nse", "bse"]):
                index_data = self.market_client.get_major_indices(market="india")
            elif any(w in user_query.lower() for w in ["us", "s&p", "nasdaq", "dow"]):
                index_data = self.market_client.get_major_indices(market="us")
            else:
                index_data = self.market_client.get_major_indices(market="all")

        # 2. If specific tickers found, fetch live quotes
        for sym in candidates[:3]:  # Limit to top 3 candidates
            q = self.market_client.get_quote(sym)
            if q.get("status") == "success" and q.get("price", 0) > 0:
                quotes_data[sym] = q

        # 3. Pull conceptual market context from RAG if query asks educational or macro questions
        rag_context = ""
        sources = []
        if self.retriever and any(
            w in user_query.lower()
            for w in [
                "why",
                "how",
                "what",
                "valuation",
                "p/e",
                "inflation",
                "recession",
                "fed",
                "rbi",
            ]
        ):
            retrieval = self.retriever.retrieve(user_query, category="market", top_k=3)
            rag_context = retrieval.context_text()
            sources = retrieval.sources

        # Build prompt for LLM synthesis
        market_summary_parts = []
        if index_data:
            market_summary_parts.append("Major Indices:")
            for name, q in index_data.items():
                curr = q.get("currency", "")
                market_summary_parts.append(
                    f"  - {name}: {format_currency(q.get('price', 0), curr)} "
                    f"({format_percentage(q.get('change_percent', 0))})"
                )

        if quotes_data:
            market_summary_parts.append("\nLive Ticker Quotes:")
            for sym, q in quotes_data.items():
                curr = q.get("currency", "")
                mc = format_large_number(q.get("market_cap") or 0)
                pe = q.get("pe_ratio") or "N/A"
                market_summary_parts.append(
                    f"  - {q.get('name', sym)} ({sym}): {format_currency(q.get('price', 0), curr)} | "
                    f"Change: {format_percentage(q.get('change_percent', 0))} | P/E: {pe} | "
                    f"52W Range: {format_currency(q.get('fifty_two_week_low', 0), curr)} - {format_currency(q.get('fifty_two_week_high', 0), curr)} | "
                    f"Market Cap: {mc}"
                )

        market_info_text = "\n".join(market_summary_parts)

        prompt = (
            f"You are a market analyst. Answer the user's question clearly and concisely:\n\n"
            f"User Question: {user_query}\n\n"
            f"Live Market Data:\n{market_info_text}\n\n"
            f"Market Knowledge Excerpts:\n{rag_context}\n\n"
            f"Guidelines:\n"
            f"- Present the live quotes clearly using bullet points.\n"
            f"- Explain what the movements or metrics signify in beginner-friendly terms.\n"
            f"- Maintain an objective educational tone without recommending buying/selling."
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [
                SystemMessage(content="You are an expert market analyst."),
                HumanMessage(content=prompt),
            ]
        )

        content = str(response.content) + GENERAL_DISCLAIMER

        return AgentOutput(
            agent_name=self.name,
            content=content,
            sources=sources,
            data={"quotes": quotes_data, "indices": index_data},
            error=None,
        )
