"""
src/utils/market_data.py

Real-time market data client backed by yFinance.
Provides:
  - Live quotes with in-memory TTL caching (default 60s)
  - Seamless support for US (AAPL, MSFT, SPY) and Indian tickers (RELIANCE.NS, TCS.NS, ^NSEI)
  - Index overviews (S&P 500, NASDAQ, Nifty 50, Sensex)
  - Recent company news and headlines
  - Graceful error handling and offline fallbacks
"""

from __future__ import annotations

import time
from typing import Any

from cachetools import TTLCache

from src.core.logging_config import get_logger
from src.core.settings import get_settings

logger = get_logger(__name__)

# In-memory cache for live quotes (max 500 items, TTL from settings)
_cache: TTLCache = TTLCache(maxsize=500, ttl=get_settings().market_data.cache_ttl_seconds)

# Common Indian ticker aliases mapped to NSE (.NS) or index symbols
INDIAN_TICKER_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFC": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "ITC": "ITC.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "WIPRO": "WIPRO.NS",
    "NIFTYBEES": "NIFTYBEES.NS",
    "GOLDBEES": "GOLDBEES.NS",
}


def normalize_ticker(raw_ticker: str) -> str:
    """Normalize raw ticker string, converting known Indian aliases to NSE tickers."""
    cleaned = raw_ticker.strip().upper()
    return INDIAN_TICKER_MAP.get(cleaned, cleaned)


class MarketDataClient:
    """Client for querying real-time market data with caching and fault tolerance."""

    def __init__(self, cache_ttl: int | None = None) -> None:
        self.settings = get_settings()
        ttl = cache_ttl or self.settings.market_data.cache_ttl_seconds
        self.cache = TTLCache(maxsize=500, ttl=ttl)

    def get_quote(self, ticker: str) -> dict[str, Any]:
        """
        Fetch real-time quote and fundamental metrics for a given ticker.

        Returns a dictionary containing:
          ticker, name, price, change, change_percent, currency,
          previous_close, open_price, day_high, day_low, 52w_high, 52w_low,
          volume, pe_ratio, market_cap, sector, timestamp, status.
        """
        symbol = normalize_ticker(ticker)
        cache_key = f"quote:{symbol}"

        if cache_key in self.cache:
            logger.debug("Quote returned from cache", ticker=symbol)
            return self.cache[cache_key]

        try:
            import yfinance as yf

            t = yf.Ticker(symbol)
            info = t.info or {}

            # Determine price from available fields
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
                or 0.0
            )
            prev_close = info.get("previousClose") or price or 1.0
            change = price - prev_close if price and prev_close else 0.0
            change_percent = (change / prev_close * 100) if prev_close else 0.0

            currency = info.get("currency", "INR" if symbol.endswith((".NS", ".BO")) else "USD")

            quote_data = {
                "ticker": symbol,
                "name": info.get("shortName") or info.get("longName") or symbol,
                "price": round(float(price), 2),
                "change": round(float(change), 2),
                "change_percent": round(float(change_percent), 2),
                "currency": currency,
                "previous_close": round(float(prev_close), 2),
                "open_price": round(float(info.get("open") or price), 2),
                "day_high": round(float(info.get("dayHigh") or price), 2),
                "day_low": round(float(info.get("dayLow") or price), 2),
                "fifty_two_week_high": round(float(info.get("fiftyTwoWeekHigh") or 0.0), 2),
                "fifty_two_week_low": round(float(info.get("fiftyTwoWeekLow") or 0.0), 2),
                "volume": info.get("volume") or 0,
                "pe_ratio": (
                    round(float(info.get("trailingPE") or 0.0), 2)
                    if info.get("trailingPE")
                    else None
                ),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector", "General"),
                "timestamp": int(time.time()),
                "status": "success",
            }

            self.cache[cache_key] = quote_data
            logger.info("Fetched live quote", ticker=symbol, price=price, currency=currency)
            return quote_data

        except Exception as exc:
            logger.error("Failed to fetch quote", ticker=symbol, error=str(exc))
            return {
                "ticker": symbol,
                "name": symbol,
                "price": 0.0,
                "change": 0.0,
                "change_percent": 0.0,
                "currency": "INR" if symbol.endswith((".NS", ".BO")) else "USD",
                "status": "error",
                "error": f"Unable to fetch market data for {symbol}: {str(exc)}",
            }

    def get_multiple_quotes(self, tickers: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch quotes for a batch of tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_quote(ticker)
        return results

    def get_major_indices(self, market: str = "all") -> dict[str, dict[str, Any]]:
        """
        Fetch quotes for major indices.

        Args:
            market: 'all' | 'us' | 'india'
        """
        us_indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI",
        }
        india_indices = {
            "Nifty 50": "^NSEI",
            "BSE Sensex": "^BSESN",
            "Bank Nifty": "^NSEBANK",
        }

        targets: dict[str, str] = {}
        if market.lower() in ("us", "all"):
            targets.update(us_indices)
        if market.lower() in ("india", "all"):
            targets.update(india_indices)

        results = {}
        for display_name, symbol in targets.items():
            quote = self.get_quote(symbol)
            quote["display_name"] = display_name
            results[display_name] = quote

        return results

    def get_ticker_news(self, ticker: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Fetch recent news items for a given ticker via yFinance.
        """
        symbol = normalize_ticker(ticker)
        cache_key = f"news:{symbol}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            import yfinance as yf

            t = yf.Ticker(symbol)
            news_items = t.news or []

            parsed_news = []
            for item in news_items[:limit]:
                # yFinance news structure: title, publisher, link, providerPublishTime
                content_obj = item.get("content", {})
                title = item.get("title") or content_obj.get("title", "")
                link = item.get("link") or content_obj.get("canonicalUrl", {}).get("url", "")
                pub = item.get("publisher") or content_obj.get("provider", {}).get(
                    "displayName", "Finance News"
                )
                pub_time = item.get("providerPublishTime") or content_obj.get("pubDate", "")

                if title:
                    parsed_news.append(
                        {
                            "title": title,
                            "publisher": pub,
                            "link": link,
                            "published_at": str(pub_time),
                            "ticker": symbol,
                        }
                    )

            self.cache[cache_key] = parsed_news
            return parsed_news

        except Exception as exc:
            logger.error("Failed to fetch news", ticker=symbol, error=str(exc))
            return []


# Global singleton client
default_market_client = MarketDataClient()
