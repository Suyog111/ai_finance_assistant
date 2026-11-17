"""
src/agents/goal_agent.py

Goal Planning Agent.
Assists with financial goal setting, retirement calculation (4% rule),
SIP / monthly savings planning, inflation adjustments (US & India),
and recommended asset allocation based on risk tolerance and timeline.
"""

from __future__ import annotations

import math
import re
from typing import Any

from src.agents.base_agent import BaseFinanceAgent
from src.core.llm_client import get_llm_for_agent
from src.core.state import AgentOutput, AppState
from src.rag.retriever import FinanceRetriever
from src.rag.vector_store import get_or_build_vector_store
from src.utils.disclaimers import GENERAL_DISCLAIMER
from src.utils.formatters import format_currency


def calculate_monthly_savings_needed(
    target_amount: float,
    years: float,
    annual_return_pct: float = 8.0,
    current_savings: float = 0.0,
) -> float:
    """
    Calculate the monthly contribution needed to reach a target amount.
    Formula: PMT = [FV - PV * (1 + r)^n] * (r / ((1 + r)^n - 1))
    """
    if years <= 0:
        return max(0.0, target_amount - current_savings)

    n_months = int(years * 12)
    monthly_rate = (annual_return_pct / 100.0) / 12.0

    if monthly_rate == 0:
        return max(0.0, (target_amount - current_savings) / n_months)

    future_val_of_current = current_savings * math.pow(1 + monthly_rate, n_months)
    remaining_target = max(0.0, target_amount - future_val_of_current)

    pmt = remaining_target * (monthly_rate / (math.pow(1 + monthly_rate, n_months) - 1))
    return round(pmt, 2)


def get_recommended_allocation(years: float, risk_tolerance: str = "moderate") -> dict[str, int]:
    """
    Determine recommended asset allocation based on time horizon and risk profile.
    """
    risk = risk_tolerance.lower()

    if years < 3:
        return {"Equities / Stocks": 0, "Fixed Income / Bonds": 20, "Cash / HYSA / Liquid": 80}
    elif years < 7:
        if risk == "conservative":
            return {"Equities / Stocks": 25, "Fixed Income / Bonds": 55, "Cash / HYSA": 20}
        elif risk == "aggressive":
            return {"Equities / Stocks": 60, "Fixed Income / Bonds": 35, "Cash / HYSA": 5}
        else:  # moderate
            return {"Equities / Stocks": 40, "Fixed Income / Bonds": 50, "Cash / HYSA": 10}
    elif years < 15:
        if risk == "conservative":
            return {"Equities / Stocks": 50, "Fixed Income / Bonds": 40, "Cash / HYSA": 10}
        elif risk == "aggressive":
            return {"Equities / Stocks": 85, "Fixed Income / Bonds": 15, "Cash / HYSA": 0}
        else:
            return {"Equities / Stocks": 70, "Fixed Income / Bonds": 25, "Cash / HYSA": 5}
    else:  # 15+ years
        if risk == "conservative":
            return {"Equities / Stocks": 65, "Fixed Income / Bonds": 30, "Cash / HYSA": 5}
        elif risk == "aggressive":
            return {"Equities / Stocks": 90, "Fixed Income / Bonds": 10, "Cash / HYSA": 0}
        else:
            return {"Equities / Stocks": 80, "Fixed Income / Bonds": 20, "Cash / HYSA": 0}


class GoalPlanningAgent(BaseFinanceAgent):
    """Specialized agent for goal setting, retirement calculations, and milestone plans."""

    name: str = "Goal Planning Agent"
    description: str = (
        "Calculates monthly savings/SIP requirements, retirement targets, and asset allocations."
    )

    def __init__(self, vector_store: Any | None = None) -> None:
        super().__init__()
        self.llm = get_llm_for_agent("goal_agent", temperature=0.1)
        try:
            store = vector_store or get_or_build_vector_store()
            self.retriever = FinanceRetriever(store)
        except Exception:
            self.retriever = None  # type: ignore[assignment]

    def invoke(self, state: AppState) -> AgentOutput:
        """Process goal query, calculate compound math, and formulate plan."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        user_query = getattr(last_message, "content", "") if last_message else ""

        # Profile attributes
        profile = state.get("user_profile", {})
        risk_tolerance = profile.get("risk_tolerance", "moderate")

        # Determine currency context (USD vs INR)
        is_india_context = any(
            w in user_query.lower()
            for w in [
                "inr",
                "rupee",
                "rupees",
                "₹",
                "lakh",
                "crore",
                "lakhs",
                "crores",
                "india",
                "sip",
                "epf",
            ]
        )
        currency = "INR" if is_india_context else "USD"
        expected_inflation = 6.0 if is_india_context else 2.5
        expected_return = 12.0 if is_india_context else 8.0  # nominal equity return

        # Parse potential target numbers and years using regex
        # Look for e.g. "$1,000,000", "50 lakhs", "1 crore", "500000"
        target_amount = 0.0
        years = 10.0

        # Check for Indian units (Lakhs / Crores)
        lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|l)s?", user_query, re.IGNORECASE)
        crore_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|cr)s?", user_query, re.IGNORECASE)
        dollar_match = re.search(r"(?:[\$₹]\s*)(\d[\d,]+(?:\.\d+)?)", user_query)
        year_match = re.search(r"(\d+)\s*(?:years?|yrs?)", user_query, re.IGNORECASE)

        if crore_match:
            target_amount = float(crore_match.group(1)) * 10_000_000
        elif lakh_match:
            target_amount = float(lakh_match.group(1)) * 100_000
        elif dollar_match:
            target_amount = float(dollar_match.group(1).replace(",", ""))

        if year_match:
            years = float(year_match.group(1))

        # If user asks about the 4% rule / retirement without explicit target:
        if "4%" in user_query or "retire" in user_query.lower():
            if target_amount == 0:
                # Default baseline retirement demonstration: 25x of $60,000 or ₹12 Lakhs
                target_amount = 25 * (1_200_000 if is_india_context else 60_000)
                years = 20.0

        # Fallback default if not parsed
        if target_amount <= 0:
            target_amount = 100_000 if currency == "USD" else 5_000_000  # $100k or ₹50 Lakhs
            years = 5.0

        # Calculate monthly savings / SIP needed
        monthly_needed = calculate_monthly_savings_needed(
            target_amount=target_amount,
            years=years,
            annual_return_pct=expected_return,
            current_savings=0.0,
        )

        allocation = get_recommended_allocation(years, risk_tolerance)

        # Pull relevant RAG context
        rag_context = ""
        sources = []
        if self.retriever:
            retrieval = self.retriever.retrieve(user_query, category="goals", top_k=3)
            rag_context = retrieval.context_text()
            sources = retrieval.sources

        prompt = (
            f"You are a certified financial goal planner. Assist the user with their goal:\n\n"
            f"User Query: {user_query}\n"
            f"Currency: {currency}\n"
            f"Target Amount: {format_currency(target_amount, currency)}\n"
            f"Time Horizon: {years} years\n"
            f"Assumed Annualized Nominal Return: {expected_return}%\n"
            f"Estimated Monthly Savings / SIP Needed: {format_currency(monthly_needed, currency)}\n"
            f"Recommended Asset Allocation: {allocation}\n"
            f"Expected Long-Term Inflation: ~{expected_inflation}%\n\n"
            f"Knowledge Base References:\n{rag_context}\n\n"
            f"Provide a clear, encouraging, step-by-step financial plan:\n"
            f"1. **Goal Breakdown**: Clarify target amount and time horizon.\n"
            f"2. **Savings Roadmap**: Explain the monthly contribution/SIP required to compound to that amount.\n"
            f"3. **Suggested Asset Allocation**: Outline how to distribute the funds across stocks, bonds, and cash.\n"
            f"4. **Actionable Tips**: Suggest automating contributions, using tax-advantaged accounts, or stepping up contributions annually."
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [
                SystemMessage(content="You are an expert financial goal planner."),
                HumanMessage(content=prompt),
            ]
        )

        content = str(response.content) + GENERAL_DISCLAIMER

        return AgentOutput(
            agent_name=self.name,
            content=content,
            sources=sources,
            data={
                "target_amount": target_amount,
                "years": years,
                "monthly_needed": monthly_needed,
                "allocation": allocation,
                "currency": currency,
            },
            error=None,
        )
