"""
src/web_app/app.py

Main entry point for the AI Finance Assistant Streamlit Application.
Launches the responsive multi-tab interface with sidebar controls and state management.
Run as: streamlit run src/web_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st

from src.core.settings import get_settings
from src.web_app.session import clear_chat_history, init_session_state
from src.web_app.views.chat import render_chat_page
from src.web_app.views.goals import render_goals_page
from src.web_app.views.market import render_market_page
from src.web_app.views.portfolio import render_portfolio_page

# Configure Streamlit page layout
st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
init_session_state()
settings = get_settings()


# ---------------------------------------------------------------------------
# Sidebar Controls & User Profile
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💹 AI Finance Assistant")
    st.caption("Applied Agentic AI · Capstone Project")
    st.markdown("---")

    st.markdown("### ⚙️ System Status")
    st.markdown(f"**LLM Model:** `{settings.llm.model_name}`")
    st.markdown("**Vector Store:** `FAISS (67 articles, 561 chunks)`")
    st.markdown("**Markets:** `US & Indian Equities`")
    st.markdown("---")

    st.markdown("### 👤 User Preferences")
    knowledge = st.selectbox(
        "Knowledge Level",
        ["Beginner", "Intermediate", "Advanced"],
        index=["beginner", "intermediate", "advanced"].index(
            st.session_state.user_profile.get("knowledge_level", "beginner")
        ),
    )
    risk = st.selectbox(
        "Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=["conservative", "moderate", "aggressive"].index(
            st.session_state.user_profile.get("risk_tolerance", "moderate")
        ),
    )
    pref_market = st.selectbox(
        "Focus Market",
        ["Both (US & India)", "Indian Markets (NSE/BSE)", "US Markets"],
        index=0,
    )

    # Sync profile to session_state
    st.session_state.user_profile["knowledge_level"] = knowledge.lower()
    st.session_state.user_profile["risk_tolerance"] = risk.lower()
    st.session_state.user_profile["primary_market"] = pref_market

    st.markdown("---")
    st.markdown("### 🔄 Session Controls")
    st.caption(f"Session Thread ID: `{st.session_state.thread_id}`")
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        clear_chat_history()
        st.success("Conversation history reset.")
        st.rerun()

    st.markdown("---")
    st.caption("Built with LangGraph, LangChain, FAISS, yFinance, and Streamlit.")


# ---------------------------------------------------------------------------
# Main Application Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_portfolio, tab_market, tab_goals = st.tabs(
    [
        "💬 Chat & Advisory",
        "📊 Portfolio Dashboard",
        "📈 Live Market Overview",
        "🎯 Goal Planner",
    ]
)

with tab_chat:
    render_chat_page()

with tab_portfolio:
    render_portfolio_page()

with tab_market:
    render_market_page()

with tab_goals:
    render_goals_page()
