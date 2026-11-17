"""
src/web_app/components/charts.py

Plotly visualization components for the Streamlit UI:
  - Sector allocation donut chart
  - Individual holding P&L bar chart
  - Multi-year compound growth projection (Invested Capital vs Interest Growth)
  - Asset allocation breakdown
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def plot_sector_allocation(sector_data: dict[str, float]) -> go.Figure:
    """Create an interactive donut chart showing portfolio sector distribution."""
    if not sector_data:
        fig = go.Figure()
        fig.update_layout(title="No sector data available")
        return fig

    labels = list(sector_data.keys())
    values = list(sector_data.values())

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                textinfo="label+percent",
                marker=dict(
                    colors=[
                        "#3B82F6",
                        "#10B981",
                        "#F59E0B",
                        "#EF4444",
                        "#8B5CF6",
                        "#EC4899",
                        "#14B8A6",
                        "#6366F1",
                    ]
                ),
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        title="<b>Sector Allocation Breakdown</b>",
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=320,
    )
    return fig


def plot_holding_gains(holdings_detail: list[dict[str, Any]]) -> go.Figure:
    """Create a horizontal bar chart of percentage return per holding."""
    if not holdings_detail:
        fig = go.Figure()
        fig.update_layout(title="No holdings data")
        return fig

    tickers = [h["ticker"] for h in holdings_detail]
    pcts = [h["gain_loss_pct"] for h in holdings_detail]
    colors = ["#10B981" if p >= 0 else "#EF4444" for p in pcts]

    fig = go.Figure(
        go.Bar(
            x=pcts,
            y=tickers,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{p:+.1f}%" for p in pcts],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>Unrealized Return by Holding (%)</b>",
        xaxis_title="Gain / Loss (%)",
        margin=dict(t=40, b=40, l=60, r=40),
        height=320,
    )
    return fig


def plot_compound_growth(
    years: int,
    monthly_contrib: float,
    annual_return_pct: float = 10.0,
    initial_investment: float = 0.0,
    currency_symbol: str = "$",
) -> go.Figure:
    """
    Generate an interactive stacked area chart demonstrating compound interest over time:
    Principal Invested vs Accumulated Compound Returns.
    """
    labels = []
    invested_series = []
    interest_series = []

    monthly_rate = (annual_return_pct / 100.0) / 12.0
    current_corpus = initial_investment
    total_invested = initial_investment

    for yr in range(1, int(years) + 1):
        for _ in range(12):
            current_corpus = (current_corpus + monthly_contrib) * (1 + monthly_rate)
            total_invested += monthly_contrib

        labels.append(f"Year {yr}")
        invested_series.append(round(total_invested, 2))
        interest_earned = max(0.0, current_corpus - total_invested)
        interest_series.append(round(interest_earned, 2))

    fig = go.Figure()

    # Principal Invested area
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=invested_series,
            mode="lines",
            name="Principal Invested",
            stackgroup="one",
            fillcolor="rgba(59, 130, 246, 0.6)",
            line=dict(color="#3B82F6", width=2),
        )
    )

    # Compound Interest area
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=interest_series,
            mode="lines",
            name="Compound Interest Earned",
            stackgroup="one",
            fillcolor="rgba(16, 185, 129, 0.6)",
            line=dict(color="#10B981", width=2),
        )
    )

    total_final = invested_series[-1] + interest_series[-1] if invested_series else 0.0

    fig.update_layout(
        title=f"<b>Wealth Accumulation Roadmap: Projected {currency_symbol}{total_final:,.0f}</b>",
        xaxis_title="Investment Horizon",
        yaxis_title=f"Portfolio Value ({currency_symbol})",
        hovermode="x unified",
        margin=dict(t=40, b=40, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360,
    )
    return fig


def plot_asset_allocation_bar(allocation: dict[str, int]) -> go.Figure:
    """Create a horizontal stacked bar for target asset allocation."""
    fig = go.Figure()
    palette = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]

    for i, (asset, pct) in enumerate(allocation.items()):
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Bar(
                name=asset,
                y=["Target Allocation"],
                x=[pct],
                orientation="h",
                text=f"{asset}: {pct}%",
                textposition="inside",
                marker=dict(color=color),
            )
        )

    fig.update_layout(
        barmode="stack",
        title="<b>Recommended Asset Allocation (%)</b>",
        xaxis=dict(range=[0, 100], title="Percentage Allocation (%)"),
        margin=dict(t=40, b=30, l=30, r=30),
        height=160,
        showlegend=False,
    )
    return fig
