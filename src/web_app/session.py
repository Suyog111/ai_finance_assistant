"""
src/web_app/session.py

Session state management and initialization for the Streamlit UI.
Ensures persistent state across user interactions and page reloads.
"""

from __future__ import annotations

import uuid

import streamlit as st

from src.core.state import Holding

DEFAULT_PORTFOLIO: list[Holding] = [
    Holding(ticker="AAPL", quantity=10, avg_cost=165.0),
    Holding(ticker="RELIANCE.NS", quantity=15, avg_cost=2850.0),
    Holding(ticker="TCS.NS", quantity=8, avg_cost=3900.0),
    Holding(ticker="MSFT", quantity=5, avg_cost=380.0),
    Holding(ticker="VTI", quantity=20, avg_cost=230.0),
]


def init_session_state() -> None:
    """Initialize default session state keys if not already set."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Welcome to the AI Finance Assistant!**\n\n"
                    "I am your multi-agent financial literacy guide. I can help you:\n"
                    "- 📚 **Learn Investing Fundamentals**: Stocks, bonds, ETFs, compound interest, index funds\n"
                    "- 📊 **Analyze Your Portfolio**: Holdings review, diversification scoring, sector allocation\n"
                    "- 📈 **Track Real-Time Markets**: Live quotes & indices across US and Indian markets (NSE/BSE)\n"
                    "- 🎯 **Plan Financial Goals**: Retirement (4% rule), house down payments, SIP calculations\n"
                    "- 🧾 **Understand Tax Concepts**: US 401(k)/IRAs, Indian Section 80C, ELSS, and Budget 2024 capital gains rules\n"
                    "- 📰 **Synthesize Market News**: Earnings reports, central bank announcements (Fed/RBI)\n\n"
                    "*What financial question or goal would you like to explore today?*"
                ),
                "agent_name": "Finance Q&A Agent",
                "sources": [],
            }
        ]

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = list(DEFAULT_PORTFOLIO)

    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "knowledge_level": "beginner",
            "risk_tolerance": "moderate",
            "primary_market": "Both (US & India)",
            "currency": "USD",
        }

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Chat"


def clear_chat_history() -> None:
    """Reset chat history to default greeting and generate new thread_id."""
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    if "messages" in st.session_state:
        del st.session_state["messages"]
    init_session_state()
