"""Abstract base class for data fetchers."""

from abc import ABC, abstractmethod

from src.data.models.ticker_data import TickerData


class BaseFetcher(ABC):
    """Abstract base class for fetching financial data."""

    @abstractmethod
    def fetch_ticker_data(self, symbol: str) -> TickerData:
        """Fetch financial data for a single ticker.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            TickerData object with financial metrics.

        Raises:
            DataFetchError: If data cannot be fetched.
        """
        pass

    @abstractmethod
    def fetch_multiple_tickers(self, symbols: list[str]) -> list[TickerData]:
        """Fetch financial data for multiple tickers.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            List of TickerData objects.
        """
        pass

