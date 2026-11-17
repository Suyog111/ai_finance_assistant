"""
src/web_app/pages/goals.py

Goal Planning & Wealth Accumulation Simulator.
Features:
  - Financial goal configuration (Retirement, Home Down Payment, Education, FIRE)
  - Compound interest math (SIP / Monthly savings calculation)
  - Interactive multi-year growth simulation (Plotly stacked area chart)
  - Time-horizon & risk-adjusted asset allocation recommendation
  - On-demand AI Milestone Plan generation
"""

from __future__ import annotations

import math

import streamlit as st

from src.agents.goal_agent import (
    GoalPlanningAgent,
    calculate_monthly_savings_needed,
    get_recommended_allocation,
)
from src.core.state import initial_state
from src.utils.formatters import format_currency
from src.web_app.components.charts import plot_asset_allocation_bar, plot_compound_growth
from src.web_app.components.disclaimer_footer import render_disclaimer_footer


def render_goals_page() -> None:
    st.subheader("🎯 Financial Goal Planner & Compounding Simulator")

    # 1. Goal Configuration Controls
    col1, col2 = st.columns([1, 1])

    with col1:
        goal_name = st.text_input("Goal Name", value="Retirement Corpus")
        currency = st.selectbox("Currency", ["INR (₹)", "USD ($)"], index=0)
        curr_symbol = "₹" if "INR" in currency else "$"
        curr_code = "INR" if "INR" in currency else "USD"

        if curr_code == "INR":
            default_target = 5000000.0  # ₹50 Lakhs
            target_amount = st.number_input(
                f"Target Amount ({curr_symbol})",
                min_value=10000.0,
                value=default_target,
                step=500000.0,
                help="E.g., ₹50,00,000 (50 Lakhs) or ₹1,00,00,000 (1 Crore)",
            )
            st.caption(
                f"Target: **{format_currency(target_amount, 'INR', use_indian_units=True)}**"
            )
        else:
            default_target = 500000.0  # $500k
            target_amount = st.number_input(
                f"Target Amount ({curr_symbol})",
                min_value=1000.0,
                value=default_target,
                step=25000.0,
            )

        initial_savings = st.number_input(
            f"Current Existing Savings ({curr_symbol})",
            min_value=0.0,
            value=0.0,
            step=10000.0,
        )

    with col2:
        years = st.slider("Time Horizon (Years)", min_value=1, max_value=35, value=10, step=1)
        expected_return = st.slider(
            "Expected Annualized Return (%)",
            min_value=3.0,
            max_value=16.0,
            value=12.0 if curr_code == "INR" else 8.0,
            step=0.5,
        )
        inflation_rate = st.slider(
            "Assumed Annual Inflation Rate (%)",
            min_value=1.0,
            max_value=10.0,
            value=6.0 if curr_code == "INR" else 2.5,
            step=0.5,
        )
        risk_tolerance = st.selectbox(
            "Your Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1
        )

    # 2. Mathematical Calculations
    monthly_needed = calculate_monthly_savings_needed(
        target_amount=target_amount,
        years=years,
        annual_return_pct=expected_return,
        current_savings=initial_savings,
    )

    total_months = years * 12
    total_principal_invested = initial_savings + (monthly_needed * total_months)
    total_compound_interest = max(0.0, target_amount - total_principal_invested)
    inflation_adjusted = target_amount * math.pow(1 + (inflation_rate / 100.0), years)

    allocation = get_recommended_allocation(years, risk_tolerance)

    st.markdown("---")

    # 3. KPI Metrics Summary Cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric(
            label="Required Monthly Contribution / SIP",
            value=format_currency(monthly_needed, curr_code),
        )
    with mcol2:
        st.metric(
            label="Target Goal Value",
            value=format_currency(target_amount, curr_code),
            help=f"Future purchasing power target at {inflation_rate}% inflation: {format_currency(inflation_adjusted, curr_code)}",
        )
    with mcol3:
        st.metric(
            label="Total Principal to Invest",
            value=format_currency(total_principal_invested, curr_code),
        )
    with mcol4:
        st.metric(
            label="Compounded Growth Earned",
            value=format_currency(total_compound_interest, curr_code),
            delta="Power of Compounding",
        )

    st.markdown("---")

    # 4. Interactive Compounding Growth Simulation
    st.markdown("#### 📈 Compounding Roadmap Over Time")
    fig = plot_compound_growth(
        years=years,
        monthly_contrib=monthly_needed,
        annual_return_pct=expected_return,
        initial_investment=initial_savings,
        currency_symbol=curr_symbol,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5. Asset Allocation Breakdown Bar
    st.markdown("#### ⚖️ Recommended Target Asset Allocation")
    st.plotly_chart(plot_asset_allocation_bar(allocation), use_container_width=True)

    st.markdown("---")

    # 6. AI Milestone Planning Action
    st.markdown("#### 🤖 AI Goal Milestone Plan")
    if st.button(
        "Generate AI Milestone Roadmap & Action Plan", type="primary", use_container_width=True
    ):
        with st.spinner(
            "Goal Planning Agent is assembling your customized milestones and roadmap..."
        ):
            query = (
                f"Create a milestone roadmap to reach {format_currency(target_amount, curr_code)} for '{goal_name}' "
                f"in {years} years with monthly savings of {format_currency(monthly_needed, curr_code)} "
                f"under a {risk_tolerance.lower()} risk profile."
            )
            state = initial_state()
            state["user_profile"] = {"risk_tolerance": risk_tolerance.lower()}
            from langchain_core.messages import HumanMessage

            state["messages"] = [HumanMessage(content=query)]

            agent = GoalPlanningAgent()
            output = agent.invoke(state)

            st.markdown(output["content"])

    render_disclaimer_footer()
