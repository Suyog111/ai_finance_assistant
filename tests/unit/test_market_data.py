"""
tests/unit/test_market_data.py

Unit tests for src/utils/market_data.py
Uses mocking for yfinance calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.utils.market_data import (
    MarketDataClient,
    normalize_ticker,
    resolve_ticker_dynamic,
    _resolution_cache,
)


class TestMarketData:
    def test_normalize_ticker(self):
        assert normalize_ticker("aapl") == "AAPL"
        assert normalize_ticker("RELIANCE") == "RELIANCE.NS"
        assert normalize_ticker("tcs") == "TCS.NS"
        assert normalize_ticker("nifty") == "^NSEI"
        assert normalize_ticker("sensex") == "^BSESN"
        # Test BEL and TATAPOWER aliases
        assert normalize_ticker("BEL") == "BEL.NS"
        assert normalize_ticker("bel") == "BEL.NS"
        assert normalize_ticker("TATAPOWER") == "TATAPOWER.NS"
        assert normalize_ticker("tatapower") == "TATAPOWER.NS"
        assert normalize_ticker("TATA POWER") == "TATAPOWER.NS"
        assert normalize_ticker("BHARAT ELECTRONICS") == "BEL.NS"
        # Test market parameter
        assert normalize_ticker("SWIGGY", market="india") == "SWIGGY.NS"
        assert normalize_ticker("AAPL", market="us") == "AAPL"

    def test_get_quote_success(self):
        client = MarketDataClient(cache_ttl=60)
        mock_info = {
            "currentPrice": 180.50,
            "previousClose": 175.00,
            "currency": "USD",
            "sector": "Technology",
            "trailingPE": 28.5,
            "shortName": "Apple Inc.",
            "volume": 50000000,
        }

        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            quote = client.get_quote("AAPL")
            assert quote["status"] == "success"
            assert quote["price"] == 180.50
            assert quote["change"] == 5.50
            assert quote["currency"] == "USD"
            assert quote["pe_ratio"] == 28.5

    def test_get_quote_cached(self):
        client = MarketDataClient(cache_ttl=60)
        mock_info = {"currentPrice": 100.0, "previousClose": 100.0}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("yfinance.Ticker", return_value=mock_ticker) as mock_yf:
            quote1 = client.get_quote("TEST")
            quote2 = client.get_quote("TEST")
            assert quote1["price"] == 100.0
            assert quote2["price"] == 100.0
            assert mock_yf.call_count == 1  # Called only once due to cache

    def test_get_quote_error_fallback(self):
        client = MarketDataClient(cache_ttl=60)

        with patch("yfinance.Ticker", side_effect=Exception("Network error")):
            quote = client.get_quote("FAILTICKER")
            assert quote["status"] == "error"
            assert quote["price"] == 0.0

    def test_get_quote_automatic_nse_fallback(self):
        client = MarketDataClient(cache_ttl=60)

        def mock_ticker_side_effect(sym):
            mock = MagicMock()
            if sym == "UNMAPPED":
                mock.info = {}
                mock.fast_info = None
            elif sym == "UNMAPPED.NS":
                mock.info = {
                    "currentPrice": 250.0,
                    "previousClose": 240.0,
                    "currency": "INR",
                    "shortName": "Unmapped India Ltd",
                }
                mock.fast_info = None
            else:
                mock.info = {}
                mock.fast_info = None
            return mock

        with patch("yfinance.Ticker", side_effect=mock_ticker_side_effect):
            quote = client.get_quote("UNMAPPED")
            assert quote["status"] == "success"
            assert quote["price"] == 250.0
            assert quote["ticker"] == "UNMAPPED.NS"
            assert quote["currency"] == "INR"

    def test_get_quote_fast_info_fallback(self):
        client = MarketDataClient(cache_ttl=60)
        mock_ticker = MagicMock()
        mock_ticker.info = {}  # Empty info dict
        mock_fast = MagicMock()
        mock_fast.last_price = 312.45
        mock_fast.previous_close = 310.0
        mock_fast.currency = "INR"
        mock_fast.day_high = 315.0
        mock_fast.day_low = 308.0
        mock_fast.year_high = 350.0
        mock_fast.year_low = 200.0
        mock_fast.last_volume = 1200000
        mock_fast.market_cap = 50000000000
        mock_ticker.fast_info = mock_fast

        with patch("yfinance.Ticker", return_value=mock_ticker):
            quote = client.get_quote("FASTINFO")
            assert quote["status"] == "success"
            assert quote["price"] == 312.45
            assert quote["day_high"] == 315.0

    def test_get_major_indices(self):
        client = MarketDataClient(cache_ttl=60)
        fake_quote = {
            "price": 5000.0,
            "change_percent": 1.2,
            "currency": "USD",
            "status": "success",
        }

        with patch.object(client, "get_quote", return_value=fake_quote):
            indices = client.get_major_indices(market="us")
            assert "S&P 500" in indices
            assert "NASDAQ" in indices
            assert "Dow Jones" in indices

    def test_get_ticker_news(self):
        client = MarketDataClient(cache_ttl=60)
        mock_news = [
            {
                "title": "Tech Rally Continues",
                "publisher": "Reuters",
                "link": "https://reuters.com/1",
            },
            {
                "title": "Fed Holds Rates",
                "publisher": "Bloomberg",
                "link": "https://bloomberg.com/2",
            },
        ]
        mock_ticker = MagicMock()
        mock_ticker.news = mock_news

        with patch("yfinance.Ticker", return_value=mock_ticker):
            news = client.get_ticker_news("AAPL", limit=2)
            assert len(news) == 2
            assert news[0]["title"] == "Tech Rally Continues"

    def test_resolve_ticker_dynamic_with_search_quotes(self):
        _resolution_cache.clear()
        mock_search = MagicMock()
        mock_search.quotes = [
            {"symbol": "TATAPOWER.NS", "exchange": "NSI", "quoteType": "EQUITY", "shortname": "TATA POWER CO"}
        ]

        with patch("yfinance.Search", return_value=mock_search) as mock_s:
            resolved = resolve_ticker_dynamic("Tata Power")
            assert resolved == "TATAPOWER.NS"
            mock_s.assert_called_once()

    def test_resolve_ticker_dynamic_market_context_disambiguation(self):
        _resolution_cache.clear()
        mock_search = MagicMock()
        mock_search.quotes = [
            {"symbol": "HAL", "exchange": "NYQ", "quoteType": "EQUITY"},
            {"symbol": "HAL.NS", "exchange": "NSI", "quoteType": "EQUITY"},
        ]

        with patch("yfinance.Search", return_value=mock_search):
            resolved_india = resolve_ticker_dynamic("HAL", market="india")
            assert resolved_india == "HAL.NS"

            _resolution_cache.clear()
            resolved_us = resolve_ticker_dynamic("HAL", market="us")
            assert resolved_us == "HAL"

    def test_resolve_ticker_dynamic_unmapped_ipo(self):
        _resolution_cache.clear()
        mock_search = MagicMock()
        mock_search.quotes = [
            {"symbol": "SWIGGY.NS", "exchange": "NSI", "quoteType": "EQUITY", "shortname": "SWIGGY LTD"}
        ]

        with patch("yfinance.Search", return_value=mock_search):
            resolved = resolve_ticker_dynamic("Swiggy")
            assert resolved == "SWIGGY.NS"

    def test_resolve_ticker_dynamic_caches_result(self):
        _resolution_cache.clear()
        mock_search = MagicMock()
        mock_search.quotes = [
            {"symbol": "ZOMATO.NS", "exchange": "NSI", "quoteType": "EQUITY"}
        ]

        with patch("yfinance.Search", return_value=mock_search) as mock_s:
            r1 = resolve_ticker_dynamic("Zomato")
            r2 = resolve_ticker_dynamic("Zomato")
            assert r1 == "ZOMATO.NS"
            assert r2 == "ZOMATO.NS"
            assert mock_s.call_count == 1  # Only 1 API call due to cache

    def test_resolve_ticker_dynamic_search_exception_fallback(self):
        _resolution_cache.clear()
        with patch("yfinance.Search", side_effect=Exception("Network error")):
            # Fallback for Indian market context
            resolved_in = resolve_ticker_dynamic("NEWSTOCK", market="india")
            assert resolved_in == "NEWSTOCK.NS"

            # Fallback for US context
            resolved_us = resolve_ticker_dynamic("NEWSTOCK", market="us")
            assert resolved_us == "NEWSTOCK"
