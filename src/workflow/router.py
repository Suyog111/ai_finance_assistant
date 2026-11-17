"""
src/workflow/router.py

Intent classification and workflow routing.
Classifies user queries into one of the specialized financial domains:
  - 'qa': General financial education, terms, concepts
  - 'portfolio': Portfolio review, holdings, asset allocation, diversification
  - 'market': Live quotes, ticker metrics, index overviews (Nifty, S&P 500)
  - 'goals': Goal setting, retirement planning, SIP/monthly savings calculations
  - 'news': Corporate news, earnings releases, macro headlines
  - 'tax': Tax laws (US IRS / Indian Income Tax), deductions (80C, NPS), capital gains
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.core.llm_client import get_llm_for_agent
from src.core.logging_config import get_logger

logger = get_logger(__name__)

IntentType = Literal["qa", "portfolio", "market", "goals", "news", "tax"]

VALID_INTENTS: set[str] = {"qa", "portfolio", "market", "goals", "news", "tax"}

ROUTER_SYSTEM_PROMPT = """You are an intent classification router for an AI Financial Assistant.
Classify the user's latest query into EXACTLY ONE of the following 6 intent categories:

1. 'portfolio': User wants to analyze, review, or input their holdings, check diversification, asset allocation, or calculate portfolio gain/loss.
2. 'market': User asks for real-time stock quotes, price checks (e.g., Apple, Reliance, TCS), index performance (S&P 500, Nifty 50, Sensex), or market valuation.
3. 'goals': User asks about financial planning, retirement corpus, 4% rule, saving for a house/college, or calculating monthly savings / SIP needed.
4. 'news': User asks for recent news headlines, earnings reports, Fed announcements, or corporate events.
5. 'tax': User asks about tax rules, 401(k), IRA, Indian Section 80C, capital gains tax (LTCG / STCG), New vs Old Tax Regime, or tax deductions.
6. 'qa': General financial education, investing basics for beginners (what is a stock, bond, ETF, compound interest, diversification definition).

Output ONLY the category name in lowercase (e.g. 'tax' or 'market' or 'portfolio'). Do not include any explanations or punctuation.
"""


def _keyword_heuristic_classifier(text: str) -> str | None:
    """Fast rule-based regex heuristics for unambiguous queries."""
    lower = text.lower()

    # Portfolio triggers
    if any(
        k in lower
        for k in [
            "my portfolio",
            "my holdings",
            "analyze my",
            "shares of",
            "bought at",
            "cost basis",
        ]
    ):
        return "portfolio"

    # Tax triggers
    if any(
        k in lower
        for k in [
            "80c",
            "ltcg",
            "stcg",
            "tax regime",
            "capital gains tax",
            "deduction",
            "401k",
            "401(k)",
            "roth ira",
            "traditional ira",
            "tax loss harvesting",
            "tax bracket",
            "tds",
        ]
    ):
        return "tax"

    # Goals triggers
    if any(
        k in lower
        for k in [
            "4% rule",
            "retire in",
            "retirement goal",
            "sip needed",
            "monthly savings",
            "emergency fund",
            "saving for a house",
            "down payment",
            "fire movement",
            "financial independence",
        ]
    ):
        return "goals"

    # News triggers
    if any(
        k in lower
        for k in [
            "news on",
            "latest news",
            "earnings report",
            "earnings release",
            "fed announcement",
            "headline",
            "headlines",
        ]
    ):
        return "news"

    # Market quote triggers
    if any(
        k in lower
        for k in [
            "stock price",
            "share price",
            "trading at",
            "nifty today",
            "sensex today",
            "s&p 500 today",
            "nasdaq today",
            "market quote",
        ]
    ):
        return "market"

    return None


def route_intent(query: str, history: list[BaseMessage] | None = None) -> IntentType:
    """
    Determine the appropriate agent intent for the user query.

    Args:
        query: Latest user message string.
        history: Optional list of previous conversation messages for context.

    Returns:
        One of 'qa', 'portfolio', 'market', 'goals', 'news', 'tax'.
    """
    clean_query = query.strip()
    if not clean_query:
        return "qa"

    # 1. Check deterministic heuristics first (fast, zero latency, zero token cost)
    heuristic = _keyword_heuristic_classifier(clean_query)
    if heuristic and heuristic in VALID_INTENTS:
        logger.info("Intent classified via heuristic", query=clean_query[:50], intent=heuristic)
        return heuristic  # type: ignore[return-value]

    # 2. Use LLM zero-shot classification
    try:
        llm = get_llm_for_agent("router", temperature=0.0)
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=clean_query),
        ]
        response = llm.invoke(messages)
        raw_intent = str(response.content).strip().lower()
        # Clean any trailing punctuation or markdown
        raw_intent = re.sub(r"[^a-z]", "", raw_intent)

        if raw_intent in VALID_INTENTS:
            logger.info("Intent classified via LLM", query=clean_query[:50], intent=raw_intent)
            return raw_intent  # type: ignore[return-value]
        else:
            logger.warning(
                "Unknown intent returned by LLM, defaulting to 'qa'", raw_intent=raw_intent
            )
            return "qa"

    except Exception as exc:
        logger.error("Intent routing failed, defaulting to 'qa'", error=str(exc))
        return "qa"
