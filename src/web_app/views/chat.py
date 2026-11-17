"""
src/web_app/views/chat.py

Interactive multi-turn conversational interface for the AI Finance Assistant.
Features:
  - Agent badge attribution (Finance Q&A, Portfolio, Market, Goals, News, Tax)
  - Collapsible source citations from RAG
  - Quick starter prompt buttons (US and Indian financial questions)
  - Full session persistence and conversation reset
"""

from __future__ import annotations

import streamlit as st

from src.web_app.components.disclaimer_footer import render_disclaimer_footer
from src.workflow.graph import run_finance_turn

SAMPLE_PROMPTS = [
    "What are the new capital gains tax rules in India after Budget 2024?",
    "Explain the difference between a traditional 401(k) and a Roth IRA",
    "How does the Three-Fund Portfolio work for beginners?",
    "What is Apple (AAPL) and Reliance (RELIANCE.NS) trading at today?",
    "How does the 4% rule work for retirement planning?",
    "What is the difference between New Tax Regime and Old Tax Regime in India?",
]


def render_chat_page() -> None:
    st.subheader("💬 Financial Advisory & Education Chat")

    # Quick Starter Prompts
    st.caption("💡 **Quick Questions to Try:**")
    cols = st.columns(3)
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        col = cols[i % 3]
        if col.button(prompt, key=f"starter_{i}", use_container_width=True):
            st.session_state.pending_query = prompt

    st.markdown("---")

    # Render Conversation History
    for _idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        content = msg["content"]
        agent_name = msg.get("agent_name", "AI Finance Assistant")
        sources = msg.get("sources", [])

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="💹"):
                # Agent badge
                st.caption(f"🤖 **Handled by:** `{agent_name}`")
                st.markdown(content)

                # Show sources expander if sources exist
                if sources:
                    with st.expander(
                        f"📚 Cited Knowledge Base Sources ({len(sources)})", expanded=False
                    ):
                        for s in sources:
                            title = s.get("title", "Article")
                            cat = s.get("category", "General").title()
                            src_name = s.get("source", "Knowledge Base")
                            st.markdown(f"- **{title}** — *{cat}* (`{src_name}`)")

    # Chat input & submission handler
    user_query = st.chat_input("Ask any financial, investing, market, or tax question...")

    # Handle pending query from button clicks if present
    if "pending_query" in st.session_state and st.session_state.pending_query:
        user_query = st.session_state.pending_query
        del st.session_state.pending_query

    if user_query:
        # 1. Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_query})

        # 2. Display immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        # 3. Invoke LangGraph workflow
        with st.chat_message("assistant", avatar="💹"):
            with st.spinner("Analyzing financial query across specialized agents..."):
                try:
                    result_state = run_finance_turn(
                        user_input=user_query,
                        thread_id=st.session_state.thread_id,
                        user_profile=st.session_state.user_profile,
                        portfolio=st.session_state.portfolio,
                    )

                    final_resp = result_state.get(
                        "final_response", "I'm sorry, I could not generate a response."
                    )
                    outputs = result_state.get("agent_outputs", [])
                    agent_name = (
                        outputs[0].get("agent_name", "Finance Assistant")
                        if outputs
                        else "Finance Assistant"
                    )
                    sources = outputs[0].get("sources", []) if outputs else []

                    # If portfolio was updated by agent, sync session_state
                    if result_state.get("portfolio"):
                        st.session_state.portfolio = result_state["portfolio"]

                    # Display response
                    st.caption(f"🤖 **Handled by:** `{agent_name}`")
                    st.markdown(final_resp)

                    if sources:
                        with st.expander(
                            f"📚 Cited Knowledge Base Sources ({len(sources)})", expanded=False
                        ):
                            for s in sources:
                                st.markdown(
                                    f"- **{s.get('title')}** — *{s.get('category', '').title()}*"
                                )

                    # Append to session state
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": final_resp,
                            "agent_name": agent_name,
                            "sources": sources,
                        }
                    )

                except Exception as exc:
                    st.error(f"Error processing message: {str(exc)}")

    render_disclaimer_footer()
