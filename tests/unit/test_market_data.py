"""
tests/unit/test_market_data.py

Unit tests for src/utils/market_data.py
Uses mocking for yfinance calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.utils.market_data import MarketDataClient, normalize_ticker


class TestMarketData:
    def test_normalize_ticker(self):
        assert normalize_ticker("aapl") == "AAPL"
        assert normalize_ticker("RELIANCE") == "RELIANCE.NS"
        assert normalize_ticker("tcs") == "TCS.NS"
        assert normalize_ticker("nifty") == "^NSEI"
        assert normalize_ticker("sensex") == "^BSESN"

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
