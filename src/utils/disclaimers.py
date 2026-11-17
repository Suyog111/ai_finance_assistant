"""
src/utils/disclaimers.py

Centralized regulatory disclaimers.
Mandatory for fintech compliance to distinguish educational assistance from certified financial/tax advice.
"""

from __future__ import annotations

GENERAL_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Educational Disclaimer:** This content is generated for informational and educational "
    "purposes only and does not constitute certified financial, investment, or legal advice. "
    "Market investments are subject to risk. Always conduct independent research or consult a "
    "licensed financial advisor (e.g., SEBI-registered RIA in India or CFP/FINRA-registered advisor in the US) "
    "before executing investment decisions."
)

TAX_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Tax Disclaimer:** Tax provisions (including US IRS codes and Indian Income Tax rules under "
    "the Union Budget 2024) are subject to periodic legislative changes and depend on individual income profiles. "
    "This explanation is purely educational. Please consult a qualified Chartered Accountant (CA) or certified "
    "CPA/tax professional for personalized tax filings."
)

PORTFOLIO_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Portfolio Disclaimer:** The metrics and allocation reviews provided are purely analytical simulations "
    "based on historical data and user-provided numbers. Past performance is no guarantee of future returns. "
    "Diversification reduces but cannot eliminate the risk of capital loss."
)
