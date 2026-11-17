"""
src/utils/__init__.py
Exports utility functions and market data client.
"""

from src.utils.disclaimers import GENERAL_DISCLAIMER, PORTFOLIO_DISCLAIMER, TAX_DISCLAIMER
from src.utils.formatters import format_currency, format_large_number, format_percentage
from src.utils.market_data import MarketDataClient, default_market_client, normalize_ticker

__all__ = [
    "GENERAL_DISCLAIMER",
    "TAX_DISCLAIMER",
    "PORTFOLIO_DISCLAIMER",
    "format_currency",
    "format_percentage",
    "format_large_number",
    "MarketDataClient",
    "default_market_client",
    "normalize_ticker",
]
