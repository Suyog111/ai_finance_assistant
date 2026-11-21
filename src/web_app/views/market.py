"""
src/web_app/views/market.py

Real-time Market Overview & Stock Quotes.
Features:
  - Major index grid (US & Indian markets: S&P 500, NASDAQ, Dow, Nifty 50, Sensex, Bank Nifty)
  - Real-time ticker search & live fundamental metrics (P/E, 52W High/Low, Market Cap)
  - Real-time financial headlines feed
"""

from __future__ import annotations

import streamlit as st

from src.utils.formatters import format_currency, format_large_number, format_percentage
from src.utils.market_data import default_market_client, normalize_ticker
from src.web_app.components.disclaimer_footer import render_disclaimer_footer


def render_market_page() -> None:
    st.subheader("📈 Real-Time Market Overview & Live Quotes")

    # 1. Market Scope Selector
    market_filter = st.radio(
        "Select Market View:",
        ["All Markets", "Indian Markets (NSE/BSE)", "US Markets"],
        horizontal=True,
    )

    market_param = "all"
    if "Indian" in market_filter:
        market_param = "india"
    elif "US" in market_filter:
        market_param = "us"

    # 2. Major Indices Grid
    st.markdown("#### 🌐 Major Benchmark Indices")
    with st.spinner("Fetching index quotes..."):
        indices = default_market_client.get_major_indices(market=market_param)

    if indices:
        cols = st.columns(len(indices))
        for i, (name, quote) in enumerate(indices.items()):
            col = cols[i]
            price = quote.get("price", 0.0)
            chg_pct = quote.get("change_percent", 0.0)
            curr = quote.get("currency", "USD")

            col.metric(
                label=name,
                value=format_currency(price, curr),
                delta=format_percentage(chg_pct),
            )

    st.markdown("---")

    # 3. Live Ticker Search
    st.markdown("#### 🔍 Real-Time Stock & ETF Lookup")
    scol1, scol2 = st.columns([3, 1])
    with scol1:
        searched_sym = (
            st.text_input(
                "Enter Ticker Symbol (e.g., BEL, TATAPOWER, AAPL, NVDA, RELIANCE, TCS, SPY)",
                value="AAPL",
                help="Supports US equities, Indian stocks (with or without .NS/.BO, e.g. BEL, TATAPOWER, RELIANCE.NS), and ETFs.",
            )
            .strip()
            .upper()
        )

    with scol2:
        st.write("")
        st.write("")
        st.button("Get Live Quote", type="primary", use_container_width=True)

    if searched_sym:
        norm_sym = normalize_ticker(searched_sym, market=market_param)
        with st.spinner(f"Fetching real-time data for {norm_sym}..."):
            quote = default_market_client.get_quote(norm_sym)

        if quote.get("status") == "success" and quote.get("price", 0) > 0:
            resolved_sym = quote.get("ticker", norm_sym)
            curr = quote.get("currency", "USD")
            st.markdown(f"### {quote.get('name', resolved_sym)} (`{resolved_sym}`)")

            # Metric Cards
            qcol1, qcol2, qcol3, qcol4, qcol5 = st.columns(5)
            with qcol1:
                st.metric(
                    label="Current Price",
                    value=format_currency(quote["price"], curr),
                    delta=format_percentage(quote["change_percent"]),
                )
            with qcol2:
                st.metric(
                    label="P/E Ratio",
                    value=str(quote.get("pe_ratio") or "N/A"),
                )
            with qcol3:
                mc = format_large_number(quote.get("market_cap") or 0)
                st.metric(label="Market Cap", value=mc)
            with qcol4:
                low_52 = format_currency(quote.get("fifty_two_week_low") or 0, curr)
                high_52 = format_currency(quote.get("fifty_two_week_high") or 0, curr)
                st.metric(label="52-Week Range", value=f"{low_52} - {high_52}")
            with qcol5:
                st.metric(label="Sector", value=quote.get("sector", "General"))

            st.markdown("---")

            # 4. Recent News for Searched Ticker
            st.markdown(f"#### 📰 Recent Headlines for {resolved_sym}")
            news_items = default_market_client.get_ticker_news(resolved_sym, limit=4)
            if news_items:
                for item in news_items:
                    st.markdown(f"- **[{item['title']}]({item['link']})** — *{item['publisher']}*")
            else:
                st.caption(f"No recent news items found for {resolved_sym}.")

        else:
            st.warning(
                f"Could not retrieve live market quote for symbol `{norm_sym}`. "
                "Tip: For Indian equities, symbols work with or without suffix (e.g. `BEL`, `TATAPOWER`, `BEL.NS`, or `BEL.BO`). "
                "Please verify the ticker symbol."
            )

    render_disclaimer_footer()
