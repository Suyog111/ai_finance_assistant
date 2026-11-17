"""
src/utils/formatters.py

Formatting utilities for financial figures, percentages,
and currency symbols (USD $ and INR ₹), including Lakhs and Crores support.
"""

from __future__ import annotations


def format_currency(amount: float, currency: str = "USD", use_indian_units: bool = False) -> str:
    """
    Format a numeric amount as currency string.

    Args:
        amount: Numerical value.
        currency: 'USD' ($) or 'INR' (₹).
        use_indian_units: If True and currency is INR, format in Lakhs and Crores.

    Returns:
        Formatted currency string, e.g. '$1,250.00' or '₹1.50 Lakhs'.
    """
    symbol = "₹" if currency.upper() in ("INR", "₹") else "$"

    if use_indian_units or currency.upper() in ("INR", "₹"):
        abs_amt = abs(amount)
        sign = "-" if amount < 0 else ""
        if abs_amt >= 10_000_000:  # 1 Crore = 10,000,000
            crores = abs_amt / 10_000_000
            return f"{sign}{symbol}{crores:,.2f} Cr"
        elif abs_amt >= 100_000:  # 1 Lakh = 100,000
            lakhs = abs_amt / 100_000
            return f"{sign}{symbol}{lakhs:,.2f} L"

    return f"{symbol}{amount:,.2f}"


def format_percentage(value: float, include_sign: bool = True) -> str:
    """Format float as percentage string e.g. +12.45% or -3.20%."""
    if include_sign and value > 0:
        return f"+{value:.2f}%"
    return f"{value:.2f}%"


def format_large_number(num: float) -> str:
    """Format large numbers in international K/M/B/T format."""
    abs_num = abs(num)
    sign = "-" if num < 0 else ""
    if abs_num >= 1e12:
        return f"{sign}{abs_num / 1e12:.2f}T"
    if abs_num >= 1e9:
        return f"{sign}{abs_num / 1e9:.2f}B"
    if abs_num >= 1e6:
        return f"{sign}{abs_num / 1e6:.2f}M"
    if abs_num >= 1e3:
        return f"{sign}{abs_num / 1e3:.2f}K"
    return f"{sign}{abs_num:.2f}"
