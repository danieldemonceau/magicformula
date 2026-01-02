"""Integration tests for data fetchers."""

import pytest

from src.data.fetchers.yfinance_fetcher import YFinanceFetcher


@pytest.mark.integration
class TestYFinanceFetcher:
    """Integration tests for YFinance fetcher."""

    def test_fetch_ticker_data_single(self) -> None:
        """Test fetching data for a single ticker."""
        fetcher = YFinanceFetcher(sleep_time=0.5)
        ticker_data = fetcher.fetch_ticker_data("AAPL")

        assert ticker_data.symbol == "AAPL"
        assert ticker_data.price is not None
        assert ticker_data.price > 0

    def test_fetch_multiple_tickers(self) -> None:
        """Test fetching data for multiple tickers."""
        fetcher = YFinanceFetcher(sleep_time=0.5)
        symbols = ["AAPL", "MSFT"]
        ticker_data_list = fetcher.fetch_multiple_tickers(symbols)

        assert len(ticker_data_list) == 2
        assert all(td.symbol in symbols for td in ticker_data_list)

    def test_fetch_invalid_ticker(self) -> None:
        """Test fetching data for invalid ticker."""
        fetcher = YFinanceFetcher(sleep_time=0.5)
        ticker_data = fetcher.fetch_ticker_data("INVALIDTICKER123")

        assert ticker_data.symbol == "INVALIDTICKER123"
        assert ticker_data.price is None or ticker_data.price == 0
