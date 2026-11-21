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
    # Major Benchmark Indices
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "^NSEMDCP50",
    "INDIAVIX": "^INDIAVIX",

    # Key ETFs
    "NIFTYBEES": "NIFTYBEES.NS",
    "BANKBEES": "BANKBEES.NS",
    "GOLDBEES": "GOLDBEES.NS",
    "SILVERBEES": "SILVERBEES.NS",
    "ITBEES": "ITBEES.NS",
    "JUNIORBEES": "JUNIORBEES.NS",
    "MON100": "MON100.NS",

    # Defense & Aerospace (specifically BEL, HAL, etc.)
    "BEL": "BEL.NS",
    "BHARAT ELECTRONICS": "BEL.NS",
    "HAL": "HAL.NS",
    "HINDUSTAN AERONAUTICS": "HAL.NS",
    "BDL": "BDL.NS",
    "BHARAT DYNAMICS": "BDL.NS",
    "MAZDOCK": "MAZDOCK.NS",
    "MAZAGON DOCK": "MAZDOCK.NS",
    "COCHINSHIP": "COCHINSHIP.NS",
    "COCHIN SHIPYARD": "COCHINSHIP.NS",
    "GRSE": "GRSE.NS",
    "PARAS": "PARAS.NS",
    "DATAPATTNS": "DATAPATTNS.NS",

    # Tata Group (specifically TATAPOWER, etc.)
    "TATAPOWER": "TATAPOWER.NS",
    "TATA POWER": "TATAPOWER.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "TATACONSUMER": "TATACONSUM.NS",
    "TATA CONSUMER": "TATACONSUM.NS",
    "TATACHEM": "TATACHEM.NS",
    "TATA CHEMICALS": "TATACHEM.NS",
    "TATAELXSI": "TATAELXSI.NS",
    "TATA ELXSI": "TATAELXSI.NS",
    "TITAN": "TITAN.NS",
    "VOLTAS": "VOLTAS.NS",
    "TRENT": "TRENT.NS",

    # Reliance & Energy/PSUs
    "RELIANCE": "RELIANCE.NS",
    "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS",
    "POWERGRID": "POWERGRID.NS",
    "POWER GRID": "POWERGRID.NS",
    "COALINDIA": "COALINDIA.NS",
    "COAL INDIA": "COALINDIA.NS",
    "IOC": "IOC.NS",
    "INDIAN OIL": "IOC.NS",
    "BPCL": "BPCL.NS",
    "GAIL": "GAIL.NS",
    "BHEL": "BHEL.NS",
    "SAIL": "SAIL.NS",
    "NHPC": "NHPC.NS",
    "SJVN": "SJVN.NS",
    "IREDA": "IREDA.NS",
    "PFC": "PFC.NS",
    "REC": "RECLTD.NS",
    "RECLTD": "RECLTD.NS",
    "IRFC": "IRFC.NS",
    "RVNL": "RVNL.NS",
    "IRCTC": "IRCTC.NS",
    "SUZLON": "SUZLON.NS",

    # Banking & Financial Services
    "HDFCBANK": "HDFCBANK.NS",
    "HDFC": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "KOTAK": "KOTAKBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "AXIS": "AXISBANK.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "PNB": "PNB.NS",
    "BANKBARODA": "BANKBARODA.NS",
    "CANBK": "CANBK.NS",
    "UNIONBANK": "UNIONBANK.NS",
    "IDFCFIRSTB": "IDFCFIRSTB.NS",
    "YESBANK": "YESBANK.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "BAJAJ FINSERV": "BAJAJFINSV.NS",
    "JIOFIN": "JIOFIN.NS",
    "JIO FINANCIAL": "JIOFIN.NS",
    "MUTHOOTFIN": "MUTHOOTFIN.NS",
    "CHOLAFIN": "CHOLAFIN.NS",
    "CDSL": "CDSL.NS",
    "BSE": "BSE.NS",
    "ANGELONE": "ANGELONE.NS",

    # IT & Technology
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    "LTIM": "LTIM.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "COFORGE": "COFORGE.NS",
    "KPITTECH": "KPITTECH.NS",
    "MPHASIS": "MPHASIS.NS",
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    "SWIGGY": "SWIGGY.NS",
    "NYKAA": "NYKAA.NS",

    # Auto & Manufacturing
    "MARUTI": "MARUTI.NS",
    "M&M": "M&M.NS",
    "MAHINDRA": "M&M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "TVSMOTOR": "TVSMOTOR.NS",
    "MOTHERSON": "MOTHERSON.NS",
    "BOSCHLTD": "BOSCHLTD.NS",
    "MRF": "MRF.NS",
    "BALKRISIND": "BALKRISIND.NS",

    # Consumer, FMCG, Infra & Metals
    "ITC": "ITC.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "HUL": "HINDUNILVR.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "BERGEPAINT": "BERGEPAINT.NS",
    "PIDILITIND": "PIDILITIND.NS",
    "DMART": "DMART.NS",
    "LT": "LT.NS",
    "L&T": "LT.NS",
    "LARSEN": "LT.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "GRASIM": "GRASIM.NS",
    "HINDALCO": "HINDALCO.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "VEDL": "VEDL.NS",
    "VEDANTA": "VEDL.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "ADANIPOWER": "ADANIPOWER.NS",
    "ADANIGREEN": "ADANIGREEN.NS",
    "SIEMENS": "SIEMENS.NS",
    "ABB": "ABB.NS",
    "HAVELLS": "HAVELLS.NS",
    "POLYCAB": "POLYCAB.NS",
    "DLF": "DLF.NS",
    "GODREJPROP": "GODREJPROP.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "CIPLA": "CIPLA.NS",
    "DRREDDY": "DRREDDY.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "MAXHEALTH": "MAXHEALTH.NS",
}


# Major Benchmark Indices
INDEX_ALIASES = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "^NSEMDCP50",
    "INDIAVIX": "^INDIAVIX",
    "SP500": "^GSPC",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "DOW JONES": "^DJI",
}

# 24-hour in-memory cache for dynamically resolved tickers
_resolution_cache: TTLCache = TTLCache(maxsize=2000, ttl=86400)


def resolve_ticker_dynamic(query: str, market: str | None = None) -> str:
    """
    Dynamically resolve any company name, raw symbol, or fuzzy input to a valid Yahoo Finance ticker
    using Yahoo Finance's live search API (yf.Search) and algorithmic suffix probing.
    Results are cached for 24 hours.
    """
    cleaned = query.strip()
    if not cleaned:
        return cleaned

    upper_clean = cleaned.upper()

    # 1. Fast-path: Index aliases (NIFTY, SENSEX, BANKNIFTY, etc.)
    if upper_clean in INDEX_ALIASES:
        return INDEX_ALIASES[upper_clean]
    no_space = upper_clean.replace(" ", "")
    if no_space in INDEX_ALIASES:
        return INDEX_ALIASES[no_space]

    # 2. Already fully qualified (has .NS, .BO, index ^, or currency =)
    if upper_clean.endswith((".NS", ".BO")) or upper_clean.startswith("^") or "=" in upper_clean:
        return upper_clean

    # 3. Check 24-hour resolution cache
    cache_key = f"{upper_clean}|{market or 'all'}"
    if cache_key in _resolution_cache:
        return _resolution_cache[cache_key]

    resolved: str | None = None

    # 4. Tier 1: Dynamic Yahoo Finance Search API
    try:
        import yfinance as yf

        s = yf.Search(cleaned, max_results=6)
        quotes = getattr(s, "quotes", []) or []

        if quotes:
            market_lower = (market or "").lower()
            if "india" in market_lower:
                # Prioritize NSE (.NS) and BSE (.BO) quotes
                for q in quotes:
                    sym = q.get("symbol", "")
                    ex = q.get("exchange", "")
                    if sym.endswith((".NS", ".BO")) or ex in ("NSI", "BSE"):
                        resolved = sym
                        break
            elif "us" in market_lower:
                # Prioritize US exchanges (NYSE, NASDAQ, AMEX, BATS)
                for q in quotes:
                    sym = q.get("symbol", "")
                    ex = q.get("exchange", "")
                    if ex in ("NYQ", "NMS", "NGM", "NCM", "ASE", "BATS") or "." not in sym:
                        resolved = sym
                        break
            else:
                # "All markets": check if any quote symbol's base matches input symbol
                for q in quotes:
                    sym = q.get("symbol", "")
                    base = sym.split(".")[0].upper()
                    if base == no_space:
                        resolved = sym
                        break
                # If no direct base match, pick top equity/ETF candidate
                if not resolved:
                    for q in quotes:
                        if q.get("quoteType") in ("EQUITY", "ETF", "INDEX"):
                            resolved = q.get("symbol")
                            break
                    if not resolved:
                        resolved = quotes[0].get("symbol")

    except Exception as exc:
        logger.debug("Dynamic search resolution exception", query=cleaned, error=str(exc))

    # 5. Tier 2: Algorithmic fallback if search returned no candidates or network failed
    if not resolved:
        if upper_clean in INDIAN_TICKER_MAP:
            resolved = INDIAN_TICKER_MAP[upper_clean]
        elif no_space in INDIAN_TICKER_MAP:
            resolved = INDIAN_TICKER_MAP[no_space]
        elif market and "india" in market.lower():
            resolved = f"{no_space}.NS"
        else:
            resolved = upper_clean

    _resolution_cache[cache_key] = resolved
    logger.debug("Resolved ticker dynamically", raw=cleaned, market=market, resolved=resolved)
    return resolved


def normalize_ticker(raw_ticker: str, market: str | None = None) -> str:
    """Normalize raw ticker or company name using dynamic search resolution."""
    return resolve_ticker_dynamic(raw_ticker, market=market)


class MarketDataClient:
    """Client for querying real-time market data with caching and fault tolerance."""

    def __init__(self, cache_ttl: int | None = None) -> None:
        self.settings = get_settings()
        ttl = cache_ttl or self.settings.market_data.cache_ttl_seconds
        self.cache = TTLCache(maxsize=500, ttl=ttl)

    def _fetch_single_quote(self, symbol: str) -> dict[str, Any]:
        """Fetch quote directly for a specific symbol without fallback."""
        try:
            import yfinance as yf

            t = yf.Ticker(symbol)
            info = t.info or {}

            # Determine price from available fields
            price = 0.0
            for candidate in (
                info.get("currentPrice"),
                info.get("regularMarketPrice"),
                info.get("previousClose"),
            ):
                if candidate is not None:
                    try:
                        val = float(candidate)
                        if val > 0:
                            price = val
                            break
                    except (ValueError, TypeError):
                        pass

            # Fallback to fast_info if price is still missing or 0
            if price <= 0 and hasattr(t, "fast_info") and t.fast_info is not None:
                try:
                    for attr in ("last_price", "previous_close"):
                        candidate = getattr(t.fast_info, attr, None)
                        if candidate is not None:
                            try:
                                val = float(candidate)
                                if val > 0:
                                    price = val
                                    break
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    pass

            if price <= 0:
                return {
                    "ticker": symbol,
                    "name": symbol,
                    "price": 0.0,
                    "status": "error",
                    "error": f"No price data available for {symbol}",
                }

            prev_close = 0.0
            raw_prev = info.get("previousClose")
            if raw_prev is not None:
                try:
                    prev_close = float(raw_prev)
                except (ValueError, TypeError):
                    pass
            if prev_close <= 0 and hasattr(t, "fast_info") and t.fast_info is not None:
                try:
                    raw_fc = getattr(t.fast_info, "previous_close", None)
                    if raw_fc is not None:
                        prev_close = float(raw_fc)
                except (ValueError, TypeError):
                    pass
            if prev_close <= 0:
                prev_close = price or 1.0

            change = float(price) - float(prev_close) if price and prev_close else 0.0
            change_percent = (change / float(prev_close) * 100) if prev_close else 0.0

            currency = "INR" if symbol.endswith((".NS", ".BO")) else "USD"
            raw_curr = info.get("currency")
            if raw_curr and isinstance(raw_curr, str):
                currency = raw_curr
            elif hasattr(t, "fast_info") and t.fast_info is not None:
                try:
                    fc_curr = getattr(t.fast_info, "currency", None)
                    if fc_curr and isinstance(fc_curr, str):
                        currency = fc_curr
                except Exception:
                    pass

            def _to_float(v: Any, default: float = 0.0) -> float:
                try:
                    return float(v) if v is not None else default
                except (ValueError, TypeError):
                    return default

            def _to_int(v: Any, default: int = 0) -> int:
                try:
                    return int(v) if v is not None else default
                except (ValueError, TypeError):
                    return default

            open_val = info.get("open") or (getattr(t.fast_info, "open", None) if hasattr(t, "fast_info") else None) or price
            high_val = info.get("dayHigh") or (getattr(t.fast_info, "day_high", None) if hasattr(t, "fast_info") else None) or price
            low_val = info.get("dayLow") or (getattr(t.fast_info, "day_low", None) if hasattr(t, "fast_info") else None) or price
            high_52 = info.get("fiftyTwoWeekHigh") or (getattr(t.fast_info, "year_high", None) if hasattr(t, "fast_info") else None) or 0.0
            low_52 = info.get("fiftyTwoWeekLow") or (getattr(t.fast_info, "year_low", None) if hasattr(t, "fast_info") else None) or 0.0
            vol = info.get("volume") or (getattr(t.fast_info, "last_volume", None) if hasattr(t, "fast_info") else None) or 0
            mcap = info.get("marketCap") or (getattr(t.fast_info, "market_cap", None) if hasattr(t, "fast_info") else None)

            pe = None
            if info.get("trailingPE") is not None:
                try:
                    pe = round(float(info["trailingPE"]), 2)
                except (ValueError, TypeError):
                    pe = None

            quote_data = {
                "ticker": symbol,
                "name": info.get("shortName") or info.get("longName") or symbol,
                "price": round(float(price), 2),
                "change": round(float(change), 2),
                "change_percent": round(float(change_percent), 2),
                "currency": currency,
                "previous_close": round(float(prev_close), 2),
                "open_price": round(_to_float(open_val, price), 2),
                "day_high": round(_to_float(high_val, price), 2),
                "day_low": round(_to_float(low_val, price), 2),
                "fifty_two_week_high": round(_to_float(high_52, 0.0), 2),
                "fifty_two_week_low": round(_to_float(low_52, 0.0), 2),
                "volume": _to_int(vol, 0),
                "pe_ratio": pe,
                "market_cap": _to_int(mcap, 0) if mcap is not None else None,
                "sector": info.get("sector", "General"),
                "timestamp": int(time.time()),
                "status": "success",
            }
            return quote_data

        except Exception as exc:
            return {
                "ticker": symbol,
                "name": symbol,
                "price": 0.0,
                "status": "error",
                "error": str(exc),
            }

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

        quote = self._fetch_single_quote(symbol)

        # Automatic fallback for symbols without exchange suffix (e.g., Indian equities on NSE/BSE)
        if quote.get("status") != "success" or quote.get("price", 0.0) <= 0:
            if not symbol.endswith((".NS", ".BO")) and not symbol.startswith("^"):
                # Try NSE (.NS) first
                nse_symbol = f"{symbol}.NS"
                nse_quote = self._fetch_single_quote(nse_symbol)
                if nse_quote.get("status") == "success" and nse_quote.get("price", 0.0) > 0:
                    self.cache[cache_key] = nse_quote
                    self.cache[f"quote:{nse_symbol}"] = nse_quote
                    logger.info("Resolved bare ticker to NSE", raw=symbol, resolved=nse_symbol)
                    return nse_quote

                # Try BSE (.BO) fallback
                bse_symbol = f"{symbol}.BO"
                bse_quote = self._fetch_single_quote(bse_symbol)
                if bse_quote.get("status") == "success" and bse_quote.get("price", 0.0) > 0:
                    self.cache[cache_key] = bse_quote
                    self.cache[f"quote:{bse_symbol}"] = bse_quote
                    logger.info("Resolved bare ticker to BSE", raw=symbol, resolved=bse_symbol)
                    return bse_quote

        if quote.get("status") == "success":
            self.cache[cache_key] = quote
            logger.info("Fetched live quote", ticker=symbol, price=quote.get("price"), currency=quote.get("currency"))
            return quote

        logger.error("Failed to fetch quote", ticker=symbol, error=quote.get("error", "Unknown error"))
        return {
            "ticker": symbol,
            "name": symbol,
            "price": 0.0,
            "change": 0.0,
            "change_percent": 0.0,
            "currency": "INR" if symbol.endswith((".NS", ".BO")) else "USD",
            "status": "error",
            "error": f"Unable to fetch market data for {symbol}: {quote.get('error', 'Symbol not found')}",
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
