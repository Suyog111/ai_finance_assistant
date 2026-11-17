"""
src/agents/news_agent.py

News Synthesizer Agent.
Summarizes real-time financial headlines for tickers or market sectors,
contextualizes news items using foundational financial principles (via RAG),
and helps investors separate market noise from fundamental trends.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseFinanceAgent
from src.agents.market_agent import extract_potential_tickers
from src.core.llm_client import get_llm_for_agent
from src.core.state import AgentOutput, AppState
from src.rag.retriever import FinanceRetriever
from src.rag.vector_store import get_or_build_vector_store
from src.utils.disclaimers import GENERAL_DISCLAIMER
from src.utils.market_data import MarketDataClient, default_market_client


class NewsSynthesizerAgent(BaseFinanceAgent):
    """Specialized agent for summarizing news and contextualizing corporate events."""

    name: str = "News Synthesizer Agent"
    description: str = "Summarizes financial news, earnings headlines, and macro announcements."

    def __init__(
        self,
        market_client: MarketDataClient | None = None,
        vector_store: Any | None = None,
    ) -> None:
        super().__init__()
        self.market_client = market_client or default_market_client
        self.llm = get_llm_for_agent("news_agent", temperature=0.1)

        try:
            store = vector_store or get_or_build_vector_store()
            self.retriever = FinanceRetriever(store)
        except Exception:
            self.retriever = None  # type: ignore[assignment]

    def invoke(self, state: AppState) -> AgentOutput:
        """Fetch headlines for mentioned tickers, contextualize, and summarize."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        user_query = getattr(last_message, "content", "") if last_message else "Financial News"

        candidates = extract_potential_tickers(user_query)
        target_ticker = candidates[0] if candidates else "^GSPC"  # Default to S&P 500

        # If user asks about Indian news, default to Nifty 50 or Reliance
        if any(w in user_query.lower() for w in ["india", "nifty", "sensex", "reliance", "tcs"]):
            if not candidates or candidates[0] == "^GSPC":
                target_ticker = "^NSEI"

        # Fetch live news items
        news_items = self.market_client.get_ticker_news(target_ticker, limit=5)

        # Pull educational context from RAG (category="news")
        rag_context = ""
        sources = []
        if self.retriever:
            retrieval = self.retriever.retrieve(user_query, category="news", top_k=3)
            rag_context = retrieval.context_text()
            sources = retrieval.sources

        news_text_list = []
        for i, item in enumerate(news_items, 1):
            title = item.get("title", "")
            pub = item.get("publisher", "")
            link = item.get("link", "")
            news_text_list.append(f"{i}. **{title}** ({pub})\n   Link: {link}")

        news_text = (
            "\n\n".join(news_text_list)
            if news_text_list
            else "No recent headlines retrieved from market feed."
        )

        prompt = (
            f"You are a financial news synthesizer and analyst. Help the user understand recent news:\n\n"
            f"User Query: {user_query}\n"
            f"Subject Ticker / Asset: {target_ticker}\n\n"
            f"Recent News Headlines:\n{news_text}\n\n"
            f"Educational News Interpretation Concepts:\n{rag_context}\n\n"
            f"Please provide:\n"
            f"1. **News Summary**: Synthesize the key developments in 2-3 clear bullet points.\n"
            f"2. **Market Context**: Explain what these developments mean for long-term investors vs. short-term traders.\n"
            f"3. **Educational Takeaway**: Remind the user how to filter out media noise and stay focused on core fundamentals."
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [
                SystemMessage(content="You are an expert financial news analyst and educator."),
                HumanMessage(content=prompt),
            ]
        )

        content = str(response.content) + GENERAL_DISCLAIMER

        return AgentOutput(
            agent_name=self.name,
            content=content,
            sources=sources,
            data={"ticker": target_ticker, "news_count": len(news_items), "headlines": news_items},
            error=None,
        )
