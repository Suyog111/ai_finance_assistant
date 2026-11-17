"""
tests/unit/test_formatters.py

Unit tests for src/utils/formatters.py
"""

from __future__ import annotations

from src.utils.formatters import format_currency, format_large_number, format_percentage


class TestFormatters:
    def test_format_currency_usd(self):
        assert format_currency(1250.50, "USD") == "$1,250.50"
        assert format_currency(0.0, "USD") == "$0.00"

    def test_format_currency_inr_regular(self):
        # Without indian units
        assert format_currency(50000.0, "INR", use_indian_units=False) == "₹50,000.00"

    def test_format_currency_inr_lakhs(self):
        # 250,000 = 2.50 Lakhs
        formatted = format_currency(250000.0, "INR", use_indian_units=True)
        assert "2.50 L" in formatted
        assert "₹" in formatted

    def test_format_currency_inr_crores(self):
        # 15,000,000 = 1.50 Crores
        formatted = format_currency(15000000.0, "INR", use_indian_units=True)
        assert "1.50 Cr" in formatted
        assert "₹" in formatted

    def test_format_percentage(self):
        assert format_percentage(12.345) == "+12.35%"
        assert format_percentage(-4.567) == "-4.57%"
        assert format_percentage(0.0) == "0.00%"
        assert format_percentage(5.0, include_sign=False) == "5.00%"

    def test_format_large_number(self):
        assert format_large_number(2_500_000_000_000) == "2.50T"
        assert format_large_number(1_200_000_000) == "1.20B"
        assert format_large_number(45_000_000) == "45.00M"
        assert format_large_number(8_500) == "8.50K"
        assert format_large_number(500) == "500.00"
