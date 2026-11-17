"""
src/web_app/components/disclaimer_footer.py

Standardized educational disclaimer footer rendered across the Streamlit app.
"""

from __future__ import annotations

import streamlit as st


def render_disclaimer_footer() -> None:
    """Render footer disclaimer banner."""
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** AI Finance Assistant is designed strictly for educational and informational purposes. "
        "It does not provide certified financial, investment, legal, or tax advice. Market investments are subject to "
        "risk. Past performance does not guarantee future results. Consult a qualified professional (e.g. SEBI-registered "
        "advisor in India or CFP/CPA in the US) before making financial decisions."
    )
