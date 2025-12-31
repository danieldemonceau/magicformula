"""End-to-end integration tests."""

import pytest

from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
from src.strategies.acquirers_multiple import AcquirersMultipleStrategy
from src.strategies.magic_formula import MagicFormulaStrategy


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_magic_formula_end_to_end(self) -> None:
        """Test complete Magic Formula workflow."""
        fetcher = YFinanceFetcher(sleep_time=0.5)
        ticker_data = fetcher.fetch_multiple_tickers(["AAPL", "MSFT"])

        strategy = MagicFormulaStrategy()
        results = strategy.calculate(ticker_data)

        assert len(results) > 0
        assert all("symbol" in r for r in results)
        assert all("magic_formula_score" in r for r in results)

    def test_acquirers_multiple_end_to_end(self) -> None:
        """Test complete Acquirer's Multiple workflow."""
        fetcher = YFinanceFetcher(sleep_time=0.5)
        ticker_data = fetcher.fetch_multiple_tickers(["AAPL", "MSFT"])

        strategy = AcquirersMultipleStrategy()
        results = strategy.calculate(ticker_data)

        assert len(results) > 0
        assert all("symbol" in r for r in results)
        assert all("acquirers_multiple_rank" in r for r in results)

