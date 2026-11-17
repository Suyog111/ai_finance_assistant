"""
src/web_app/pages/portfolio.py

Portfolio Analysis Dashboard.
Features:
  - Live metric KPI cards (Total Value, Cost Basis, Unrealized P&L, Diversification Score)
  - Holdings table with real-time quotes (US & Indian equities)
  - Interactive Plotly visualizations (Sector Donut Chart, Holding Return Bar Chart)
  - Interactive holding management (Add / Remove securities)
  - On-demand AI Portfolio Analysis commentary
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agents.portfolio_agent import PortfolioAnalysisAgent, calculate_portfolio_metrics
from src.core.state import Holding, initial_state
from src.utils.formatters import format_currency, format_percentage
from src.utils.market_data import default_market_client, normalize_ticker
from src.web_app.components.charts import plot_holding_gains, plot_sector_allocation
from src.web_app.components.disclaimer_footer import render_disclaimer_footer


def render_portfolio_page() -> None:
    st.subheader("📊 Portfolio Analysis & Holdings Dashboard")

    holdings = st.session_state.portfolio

    # Calculate live metrics
    with st.spinner("Fetching live market prices..."):
        metrics = calculate_portfolio_metrics(holdings, default_market_client)

    currency = metrics.get("currency", "USD")

    # 1. Top KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Current Value",
            value=format_currency(metrics["total_value"], currency),
        )
    with col2:
        st.metric(
            label="Total Cost Basis",
            value=format_currency(metrics["total_cost"], currency),
        )
    with col3:
        p_pct = metrics["total_gain_loss_pct"]
        st.metric(
            label="Unrealized P&L",
            value=format_currency(metrics["total_gain_loss"], currency),
            delta=format_percentage(p_pct),
        )
    with col4:
        score = metrics["diversification_score"]
        color = "🟢" if score >= 70 else ("🟡" if score >= 40 else "🔴")
        st.metric(
            label="Diversification Score",
            value=f"{score} / 100",
            delta=f"{color} Rating",
        )

    st.markdown("---")

    # 2. Holdings Breakdown Table
    st.markdown("#### 📋 Current Holdings")
    if metrics["holdings_detail"]:
        table_rows = []
        for h in metrics["holdings_detail"]:
            table_rows.append(
                {
                    "Ticker": h["ticker"],
                    "Shares": h["quantity"],
                    "Avg Cost": format_currency(h["avg_cost"], h["currency"]),
                    "Live Price": format_currency(h["current_price"], h["currency"]),
                    "Total Value": format_currency(h["current_value"], h["currency"]),
                    "Gain / Loss": format_currency(h["gain_loss"], h["currency"]),
                    "Return %": format_percentage(h["gain_loss_pct"]),
                    "Weight %": f"{h['weight_pct']}%",
                    "Sector": h["sector"],
                }
            )
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No holdings found. Add securities using the form below.")

    # 3. Interactive Visualizations
    st.markdown("#### 📈 Visualizations")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.plotly_chart(
            plot_sector_allocation(metrics["sector_allocation"]), use_container_width=True
        )
    with vcol2:
        st.plotly_chart(plot_holding_gains(metrics["holdings_detail"]), use_container_width=True)

    st.markdown("---")

    # 4. Holding Manager (Add / Remove)
    st.markdown("#### ⚙️ Manage Holdings")
    mcol1, mcol2 = st.columns([2, 1])

    with mcol1:
        with st.expander("➕ Add New Holding", expanded=False):
            with st.form("add_holding_form", clear_on_submit=True):
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    new_ticker = (
                        st.text_input("Ticker Symbol (e.g. AAPL, RELIANCE.NS, VTI)").strip().upper()
                    )
                with fcol2:
                    new_qty = st.number_input(
                        "Shares / Units", min_value=0.01, value=10.0, step=1.0
                    )
                with fcol3:
                    new_cost = st.number_input(
                        "Average Purchase Price", min_value=0.01, value=100.0, step=1.0
                    )

                submitted = st.form_submit_button("Add to Portfolio")
                if submitted and new_ticker:
                    clean_sym = normalize_ticker(new_ticker)
                    st.session_state.portfolio.append(
                        Holding(ticker=clean_sym, quantity=float(new_qty), avg_cost=float(new_cost))
                    )
                    st.success(f"Added {new_qty} shares of {clean_sym} to portfolio!")
                    st.rerun()

    with mcol2, st.expander("🗑️ Remove Holding", expanded=False):
        if holdings:
            ticker_options = [h["ticker"] for h in holdings]
            remove_sym = st.selectbox("Select holding to remove", ticker_options)
            if st.button("Delete Holding", type="secondary"):
                st.session_state.portfolio = [h for h in holdings if h["ticker"] != remove_sym]
                st.warning(f"Removed {remove_sym} from portfolio.")
                st.rerun()

    st.markdown("---")

    # 5. On-Demand AI Portfolio Analysis Commentary
    st.markdown("#### 🤖 AI Portfolio Analysis & Rebalancing Recommendations")
    if st.button(
        "Generate Comprehensive AI Portfolio Review", type="primary", use_container_width=True
    ):
        with st.spinner(
            "AI Portfolio Analysis Agent is analyzing your asset allocation and concentration risks..."
        ):
            temp_state = initial_state()
            temp_state["portfolio"] = st.session_state.portfolio
            agent = PortfolioAnalysisAgent(market_client=default_market_client)
            analysis_output = agent.invoke(temp_state)

            st.markdown(analysis_output["content"])

    render_disclaimer_footer()
